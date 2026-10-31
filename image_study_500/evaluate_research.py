"""Exploratory robustness studies. Existing test data is not a new external campaign."""
import json
import tempfile
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from threadpoolctl import threadpool_limits
from image_pipeline import ROOT, SEED, METALS, TARGETS, MAXIMA, FEATURES, SPOTS, preprocess_image
from reliability import load_frame, reliability_partition, small_spaces, load_reliable, RGB_FEATURES, estimate_checked
from quality_control import ImageRejected

OUT = ROOT/'results/robustness'

def metric_row(truth, prediction):
    return {'MAE':mean_absolute_error(truth,prediction),'RMSE':mean_squared_error(truth,prediction)**.5,'R2':r2_score(truth,prediction)}

def nested_repeats(proper):
    rows, assignments = [], []
    for repeat in range(3):
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+100+repeat).split(proper,groups=proper.session_id))
        train,valid = proper.iloc[a],proper.iloc[b]
        assert not set(train.session_id)&set(valid.session_id)
        assert not set(train.sample_id)&set(valid.sample_id)
        for role, part in [('outer_train', train),('outer_validation',valid)]:
            assignments.extend({'repeat':repeat,'image_id':r.image_id,'session_id':r.session_id,'role':role} for r in part.itertuples())
        folds=list(GroupKFold(n_splits=3).split(train,groups=train.session_id))
        for metal,target in zip(METALS,TARGETS):
            for family,(model,grid) in small_spaces().items():
                print(f'Nested evaluation {repeat+1}/3: {metal} / {family}',flush=True)
                search=GridSearchCV(model,grid,cv=folds,scoring='neg_root_mean_squared_error',n_jobs=4,error_score='raise')
                with threadpool_limits(limits=1),joblib.parallel_backend('threading'):
                    search.fit(train[FEATURES],train[target])
                pred=np.asarray(search.predict(valid[FEATURES])).reshape(-1)
                rows.append({'repeat':repeat,'metal':metal,'model':family,'outer_train_sessions':train.session_id.nunique(),
                             'outer_validation_sessions':valid.session_id.nunique(),**metric_row(valid[target],pred)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'nested_grouped_repeats.csv',index=False)
    pd.DataFrame(assignments).to_csv(OUT/'nested_assignments.csv',index=False)
    frame.groupby(['metal','model'])[['MAE','RMSE','R2']].agg(['mean','std']).to_csv(OUT/'nested_repeat_summary.csv')

def bootstrap_original_test():
    data=pd.read_csv(ROOT/'results/test_predictions.csv')
    rows=[]
    for (metal,family),part in data.groupby(['metal','model']):
        part=part.copy();part['ae']=abs(part.actual_ug_L_simulated-part.predicted_ug_L_simulated)
        part['se']=(part.actual_ug_L_simulated-part.predicted_ug_L_simulated)**2
        part['y2']=part.actual_ug_L_simulated**2
        totals=part.groupby('session_id').agg(n=('ae','size'),ae=('ae','sum'),se=('se','sum'),y=('actual_ug_L_simulated','sum'),y2=('y2','sum'))
        count=len(totals);weights=np.random.default_rng(SEED+14).multinomial(count,np.ones(count)/count,size=1000)
        agg=weights@totals.to_numpy();n,ae,se,y,y2=agg.T
        values=np.c_[ae/n,np.sqrt(se/n),1-se/(y2-y*y/n)]
        for i,name in enumerate(['MAE','RMSE','R2']):
            lo,hi=np.percentile(values[:,i],[2.5,97.5])
            rows.append({'metal':metal,'model':family,'metric':name,'lower_95':lo,'upper_95':hi,'test_sessions':count,'resamples':1000})
    pd.DataFrame(rows).to_csv(OUT/'original_test_session_bootstrap.csv',index=False)

def reliable_test(test):
    config,models,_=load_reliable();rows=[];predictions=[]
    for metal,target,maximum in zip(METALS,TARGETS,MAXIMA):
        pred=np.asarray(models[metal].predict(test[FEATURES])).reshape(-1);truth=test[target].to_numpy()
        radius=config['metals'][metal]['interval_radius_ug_L_simulated'];covered=abs(pred-truth)<=radius
        grouped=pd.DataFrame({'session':test.session_id,'covered':covered}).groupby('session').covered.all()
        rows.append({'metal':metal,'model':config['metals'][metal]['family'],**metric_row(truth,pred),
                     'NRMSE_percent_full_scale':100*np.sqrt(np.mean((pred-truth)**2))/maximum,
                     'empirical_image_coverage':covered.mean(),'empirical_session_coverage':grouped.mean(),
                     'interval_radius':radius,'negative_predictions':int((pred<0).sum())})
        predictions.extend({'image_id':r.image_id,'session_id':r.session_id,'metal':metal,'actual':a,'prediction':p,'lower':max(0,p-radius),'upper':p+radius}
                           for r,a,p in zip(test.itertuples(),truth,pred))
    pd.DataFrame(rows).to_csv(OUT/'reliable_test_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'reliable_test_predictions.csv',index=False)

def ablations(proper,test):
    _,models,_=load_reliable()
    raw=[]
    for row in pd.concat([proper,test]).itertuples():
        _,rectified,_,_,_=preprocess_image(ROOT/row.image_path)
        image=cv2.medianBlur(rectified,3);yy,xx=np.ogrid[:280,:560];features={}
        for metal,(x,y) in zip(METALS,SPOTS):
            rgb=np.median(image[(xx-x)**2+(yy-y)**2<=16**2],axis=0)
            features.update({f'{metal}_{c}':v for c,v in zip(['R','G','B'],rgb)})
        raw.append({'image_id':row.image_id,**features})
    raw=pd.DataFrame(raw).set_index('image_id');rows=[]
    for variant in ['corrected_RGB_HSV_LAB','corrected_RGB','uncorrected_RGB']:
        for metal,target in zip(METALS,TARGETS):
            if variant=='uncorrected_RGB':
                Xtr=raw.loc[proper.image_id,RGB_FEATURES];Xte=raw.loc[test.image_id,RGB_FEATURES]
            else:
                columns=FEATURES if variant=='corrected_RGB_HSV_LAB' else RGB_FEATURES
                Xtr=proper[columns];Xte=test[columns]
            model=clone(models[metal])
            with threadpool_limits(limits=1): model.fit(Xtr,proper[target])
            pred=np.asarray(model.predict(Xte)).reshape(-1)
            rows.append({'variant':variant,'metal':metal,**metric_row(test[target],pred)})
    pd.DataFrame(rows).to_csv(OUT/'feature_preprocessing_ablation.csv',index=False)

def stress_tests(test):
    # One fixed image per test session. Stress scenes are diagnostic, not extra training rows.
    samples=test.groupby('session_id').head(1);rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for row in samples.itertuples():
            base=cv2.imread(str(ROOT/row.image_path))
            _,_,_,points,_=preprocess_image(ROOT/row.image_path)
            from image_pipeline import FIDUCIALS
            transform=cv2.getPerspectiveTransform(FIDUCIALS,points)
            point=cv2.perspectiveTransform(np.array([[[65,127]]],np.float32),transform)[0,0].astype(int)
            glare=base.copy();cv2.circle(glare,tuple(point),20,(255,255,255),-1)
            blocked=base.copy();cv2.circle(blocked,tuple(point),20,(0,0,0),-1)
            cropped=base.copy();cropped[:150,:200]=20
            cases={'clean':base,'heavy_blur':cv2.GaussianBlur(base,(31,31),8),
                   'upside_down':cv2.rotate(base,cv2.ROTATE_180),'glare':glare,'occluded_pad':blocked,
                   'missing_marker':cropped,'severe_underexposure':np.clip(base*.22,0,255).astype(np.uint8)}
            for name,image in cases.items():
                path=Path(tmp)/'stress.png';cv2.imwrite(str(path),image)
                try:
                    result,_=estimate_checked(path);accepted=True;reason=''
                except ImageRejected as error:
                    accepted=False;reason='; '.join(error.reasons)
                rows.append({'image_id':row.image_id,'session_id':row.session_id,'condition':name,'accepted':accepted,'reason':reason})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'image_stress_tests.csv',index=False)
    frame.groupby('condition').accepted.agg(['count','sum','mean']).to_csv(OUT/'stress_summary.csv')

def main():
    OUT.mkdir(exist_ok=True)
    proper,calibration,test=reliability_partition(load_frame())
    bootstrap_original_test();reliable_test(test)
    nested_repeats(proper)
    print('Running feature/preprocessing ablations...',flush=True);ablations(proper,test)
    print('Running image stress tests...',flush=True);stress_tests(test)
    print('Research checks complete.',flush=True)

if __name__=='__main__': main()
