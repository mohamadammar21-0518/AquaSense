"""Deterministic, synthetic-only seven-pad image study. No real chemical calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
SEED = 20261024
METALS = ['Pb', 'Hg', 'Cd', 'As', 'Cr', 'Cu', 'Zn']
MAXIMA = np.array([150., 30., 50., 120., 180., 1500., 5000.])
TARGETS = [f'{m}_ug_L_simulated' for m in METALS]
CHANNELS = ['R', 'G', 'B', 'H', 'S', 'V', 'L_star', 'a_star', 'b_star']
FEATURES = [f'{m}_{c}' for m in METALS for c in CHANNELS]
CARD_W, CARD_H = 560, 280
FIDUCIALS = np.array([[22, 22], [538, 22], [538, 258], [22, 258]], dtype=np.float32)
SPOTS = [(65 + 71*i, 127) for i in range(7)]
REF_CENTERS = [(175, 211), (280, 211), (385, 211)]
REF_VALUES = np.array([235., 128., 35.])
N_SESSIONS, PROFILES_PER_SESSION, CAPTURES = 50, 5, 2


def prepare_folders():
    for name in ['design', 'datasets/images', 'datasets/preprocessed', 'charts', 'models', 'results']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def lab_from_rgb(rgb):
    value = np.asarray(rgb, dtype=float) / 255.
    linear = np.where(value <= .04045, value / 12.92, ((value+.055)/1.055)**2.4)
    xyz = linear @ np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
    xyz /= np.array([.95047, 1., 1.08883])
    delta = 6/29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    return np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])


def assay_colors(fractions, ph, temperature, lot_shift):
    # INVENTED dominant per-metal responses. These are not measured reagent colors.
    directions = np.array([[-80,-125,-30],[-20,-90,-140],[-10,-135,-85],[-115,-25,-105],[-120,-70,-15],[-10,-85,-135],[-85,-100,-20]])
    response = 1 - np.exp(-1.8 * fractions)
    cross = .045*np.roll(fractions, 1) + .025*np.roll(fractions, -1)
    rgb = np.array([238.,233.,222.]) + (response+cross)[:,None]*directions
    rgb += (ph-7)*np.array([1.3,-1.,.8]) + (temperature-25)*np.array([-.15,.1,.12])
    return np.clip(rgb+lot_shift, 40, 245)


def render_card(colors, rng):
    card = np.full((CARD_H,CARD_W,3), 242., dtype=np.float32)
    card += rng.normal(0,.9,(CARD_H,CARD_W,1)).astype(np.float32)
    card = np.clip(card,0,255).astype(np.uint8)
    for x,y in FIDUCIALS.astype(int):
        cv2.rectangle(card,(x-10,y-10),(x+10,y+10),(12,12,12),-1)
    cv2.putText(card,'AquaSense - proposed seven-spot strip',(68,57),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,70,85),1,cv2.LINE_AA)
    yy,xx = np.ogrid[:CARD_H,:CARD_W]
    for i,(x,y) in enumerate(SPOTS):
        mask = (xx-x)**2+(yy-y)**2 <= 23**2
        grain = rng.normal(0,1.8,(mask.sum(),1))
        radial = (((xx-x)**2+(yy-y)**2)[mask]/23**2)[:,None]
        card[mask] = colors[i] + grain + radial*1.8
        cv2.circle(card,(x,y),24,(180,180,180),1,cv2.LINE_AA)
        cv2.putText(card,METALS[i],(x-12,174),cv2.FONT_HERSHEY_SIMPLEX,.48,(45,45,45),1,cv2.LINE_AA)
    for (x,y),value in zip(REF_CENTERS,REF_VALUES):
        cv2.rectangle(card,(x-32,y-13),(x+32,y+13),(float(value),)*3,-1)
    cv2.putText(card,'SYNTHETIC ONLY - assumed chemistry',(119,246),cv2.FONT_HERSHEY_SIMPLEX,.39,(80,80,80),1,cv2.LINE_AA)
    return np.clip(card,0,255).astype(np.uint8)


def camera_capture(card, rng, session_gain):
    angle = rng.uniform(-10,10)
    scale = rng.uniform(.88,1.)
    matrix = cv2.getRotationMatrix2D((CARD_W/2,CARD_H/2),angle,scale)
    matrix[:,2] += [320-CARD_W/2+rng.uniform(-12,12),240-CARD_H/2+rng.uniform(-12,12)]
    corners = np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]],np.float32)
    dst = np.c_[corners,np.ones(4)] @ matrix.T + rng.uniform(-5,5,(4,2))
    homography = cv2.getPerspectiveTransform(corners,dst.astype(np.float32))
    canvas = np.empty((480,640,3),np.float32)
    canvas[:] = rng.uniform([25,35,42],[40,50,62])
    canvas += rng.normal(0,1.5,(480,640,1)).astype(np.float32)
    warped = cv2.warpPerspective(card,homography,(640,480)).astype(np.float32)
    mask = cv2.warpPerspective(np.full((CARD_H,CARD_W),255,np.uint8),homography,(640,480))
    alpha = mask[:,:,None]/255.
    canvas = canvas*(1-alpha)+warped*alpha
    yy,xx = np.mgrid[:480,:640]
    slope_x,slope_y = rng.uniform(-.08,.08,2)
    light = rng.uniform(.91,1.04)*(1+slope_x*(xx/640-.5)+slope_y*(yy/480-.5))
    gain = session_gain*rng.normal(1,.012,3)
    canvas = canvas*light[:,:,None]*gain
    gamma = rng.uniform(.96,1.04)
    canvas = 255*np.clip(canvas/255,0,1)**gamma
    noise = rng.uniform(.6,2.)
    canvas += rng.normal(0,noise,canvas.shape)
    sigma = rng.uniform(.15,.65)
    canvas = cv2.GaussianBlur(np.clip(canvas,0,255).astype(np.uint8),(3,3),sigma)
    quality = int(rng.integers(78,96))
    ok,buffer = cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,quality])
    assert ok
    return buffer, {'angle_deg':float(angle),'scale':float(scale),'gamma':float(gamma),'jpeg_quality':quality,'noise_sd':float(noise),'homography':homography.tolist()}


def generate_images():
    prepare_folders()
    rng = np.random.default_rng(SEED)
    rows, metadata = [], {}
    for session in range(N_SESSIONS):
        session_gain = rng.normal(1,.025,3)
        lot_shift = rng.normal(0,1.5,(7,3))
        for sample in range(PROFILES_PER_SESSION):
            sample_id = f'SIM-S{session:02d}-P{sample:02d}'
            fraction = .88*rng.beta(1.2,1.5,7)+.12*rng.beta(2,2)
            fraction[rng.random(7)<.06] = rng.uniform(0,.015)
            ph,temperature = rng.uniform(6,8),rng.uniform(20,30)
            colors = assay_colors(fraction,ph,temperature,lot_shift)
            # Same underlying strip texture/chemistry for both captures of a profile.
            card = render_card(colors,rng)
            for capture in range(CAPTURES):
                image_id = f'{sample_id}-C{capture}'
                buffer,render_info = camera_capture(card,rng,session_gain)
                path = f'datasets/images/{image_id}.jpg'
                buffer.tofile(str(ROOT/path))
                rows.append({'image_id':image_id,'sample_id':sample_id,'session_id':f'SIM-S{session:02d}',
                             'capture':capture,'image_path':path,'is_synthetic':True,
                             **dict(zip(TARGETS,fraction*MAXIMA))})
                metadata[image_id] = {'ph_simulated':float(ph),'temperature_C_simulated':float(temperature),**render_info}
    labels = pd.DataFrame(rows)
    tr,te = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(labels,groups=labels.session_id))
    labels['partition'] = 'train'
    labels.loc[te,'partition'] = 'test'
    assert len(labels)==500 and labels.sample_id.nunique()==250
    labels.to_csv(ROOT/'datasets/labels.csv',index=False,float_format='%.9f')
    (ROOT/'datasets/render_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    template = render_card(assay_colors(np.full(7,.45),7,25,np.zeros((7,3))),np.random.default_rng(SEED))
    cv2.imwrite(str(ROOT/'design/numerical_template.png'),cv2.cvtColor(template,cv2.COLOR_RGB2BGR))
    print('Generated 500 JPEGs, 250 profiles, 50 acquisition sessions.',flush=True)
    return labels


def preprocess_image(path):
    """Read pixels only. No labels, simulated pose or rendering metadata are accepted."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f'Unreadable image: {path}')
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    _,binary = cv2.threshold(gray,65,255,cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 130 < area < 650:
            continue
        (_, _),(w,h),_ = cv2.minAreaRect(contour)
        if min(w,h)<10 or max(w,h)/min(w,h)>1.35 or area/(w*h)<.75:
            continue
        moments = cv2.moments(contour)
        candidates.append([moments['m10']/moments['m00'],moments['m01']/moments['m00']])
    points = np.array(candidates,dtype=np.float32)
    if len(points)<4:
        raise ValueError(f'Only {len(points)} fiducials found in {path.name}')
    # Known modest rotation: extreme corner centers determine the four marks.
    sums = points.sum(axis=1)
    diff = points[:,0]-points[:,1]
    indices = [np.argmin(sums),np.argmax(diff),np.argmax(sums),np.argmin(diff)]
    if len(set(indices))!=4:
        raise ValueError('Ambiguous fiducial layout')
    ordered = points[indices]
    if cv2.contourArea(ordered)<60000:
        raise ValueError('Detected quadrilateral too small')
    transform = cv2.getPerspectiveTransform(ordered,FIDUCIALS)
    rectified = cv2.cvtColor(cv2.warpPerspective(bgr,transform,(CARD_W,CARD_H)),cv2.COLOR_BGR2RGB)
    # Small fixed median filter; extraction additionally uses robust inner-region medians.
    denoised = cv2.medianBlur(rectified,3).astype(np.float32)
    refs = np.array([np.median(denoised[y-7:y+8,x-23:x+24],axis=(0,1)) for x,y in REF_CENTERS])
    span = refs[0]-refs[2]
    if np.any(span<100):
        raise ValueError('Reference range too low')
    corrected = np.clip((denoised-refs[2])*(200/span)+35,0,255).astype(np.uint8)
    check_gray = np.median(corrected[204:219,257:304],axis=(0,1))
    features = {}
    yy,xx = np.ogrid[:CARD_H,:CARD_W]
    for metal,(x,y) in zip(METALS,SPOTS):
        roi = (xx-x)**2+(yy-y)**2 <= 16**2
        rgb = np.median(corrected[roi],axis=0)
        hsv = cv2.cvtColor((rgb/255).astype(np.float32).reshape(1,1,3),cv2.COLOR_RGB2HSV)[0,0]
        hsv[1:] *= 100
        values = np.r_[rgb,hsv,lab_from_rgb(rgb)]
        features.update({f'{metal}_{c}':float(v) for c,v in zip(CHANNELS,values)})
    qc = {'fiducial_candidates':len(points),'gray_reference_abs_error':float(np.abs(check_gray-128).mean()),'minimum_reference_span':float(span.min())}
    return features,rectified,corrected,ordered,qc


def extract_dataset(labels):
    rows,qc_rows = [],[]
    for row in labels.itertuples():
        features,rectified,corrected,points,qc = preprocess_image(ROOT/row.image_path)
        rows.append({'image_id':row.image_id,'sample_id':row.sample_id,'session_id':row.session_id,'partition':row.partition,**features})
        qc_rows.append({'image_id':row.image_id,'status':'ok',**qc})
        cv2.imwrite(str(ROOT/'datasets/preprocessed'/f'{row.image_id}.png'),cv2.cvtColor(corrected,cv2.COLOR_RGB2BGR))
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/'datasets/features.csv',index=False,float_format='%.9f')
    pd.DataFrame(qc_rows).to_csv(ROOT/'datasets/preprocessing_qc.csv',index=False)
    print(f'Preprocessed {len(frame)} images and extracted {len(FEATURES)} features each.',flush=True)
    return frame


def model_spaces():
    estimators = {
        'PLS':(PLSRegression(scale=False,max_iter=1500),{'n_components':[2,4,8]}),
        'RandomForest':(RandomForestRegressor(n_estimators=150,random_state=SEED,n_jobs=1),{'max_depth':[None,8],'min_samples_leaf':[1,3]}),
        'XGBoost':(XGBRegressor(n_estimators=200,learning_rate=.05,random_state=SEED,n_jobs=1,objective='reg:squarederror'),{'max_depth':[2,3],'reg_lambda':[1.,5.]}),
        'RBF_SVR':(SVR(kernel='rbf',epsilon=.05),{'C':[1.,10.,100.],'gamma':['scale',.02]})}
    out = {}
    for name,(estimator,grid) in estimators.items():
        pipeline = Pipeline([('scale',StandardScaler()),('model',estimator)])
        fitted = TransformedTargetRegressor(regressor=pipeline,transformer=StandardScaler())
        out[name] = (fitted,{f'regressor__model__{k}':v for k,v in grid.items()})
    return out


def train_models(labels,features):
    frame = features.merge(labels[['image_id',*TARGETS]],on='image_id',validate='one_to_one')
    tr,te = frame[frame.partition=='train'],frame[frame.partition=='test']
    assert len(tr)==400 and len(te)==100
    assert not set(tr.session_id)&set(te.session_id)
    assert not set(tr.sample_id)&set(te.sample_id)
    folds = list(GroupKFold(n_splits=5).split(tr[FEATURES],groups=tr.session_id))
    fold_rows = []
    for fold,(a,b) in enumerate(folds):
        assert not set(tr.iloc[a].session_id)&set(tr.iloc[b].session_id)
        fold_rows.extend({'image_id':image_id,'validation_fold':fold} for image_id in tr.iloc[b].image_id)
    pd.DataFrame(fold_rows).to_csv(ROOT/'results/validation_folds.csv',index=False)
    metrics,predictions,searches,baseline = [],[],[],[]
    for metal,target,maximum in zip(METALS,TARGETS,MAXIMA):
        truth = te[target].to_numpy()
        base = np.full(len(te),tr[target].mean())
        baseline.append({'metal':metal,'MAE':mean_absolute_error(truth,base),'RMSE':mean_squared_error(truth,base)**.5,'R2':r2_score(truth,base)})
        for name,(model,grid) in model_spaces().items():
            print(f'Training {metal}: {name}',flush=True)
            search = GridSearchCV(model,grid,scoring='neg_root_mean_squared_error',cv=folds,n_jobs=4,error_score='raise')
            with joblib.parallel_backend('threading'):
                search.fit(tr[FEATURES],tr[target])
            prediction = np.asarray(search.predict(te[FEATURES])).reshape(-1)
            rmse = mean_squared_error(truth,prediction)**.5
            metrics.append({'metal':metal,'model':name,'n_test_images':len(te),'n_test_profiles':te.sample_id.nunique(),
                            'MAE':mean_absolute_error(truth,prediction),'RMSE':rmse,'R2':r2_score(truth,prediction),
                            'NRMSE_percent_full_scale':100*rmse/maximum,'validation_RMSE':-search.best_score_,
                            'negative_predictions':int((prediction<0).sum()),'is_synthetic':True})
            predictions.extend({'image_id':image_id,'sample_id':sample_id,'session_id':session_id,'metal':metal,'model':name,
                                'actual_ug_L_simulated':actual,'predicted_ug_L_simulated':pred}
                               for image_id,sample_id,session_id,actual,pred in zip(te.image_id,te.sample_id,te.session_id,truth,prediction))
            searches.append({'metal':metal,'model':name,'best_parameters':json.dumps(search.best_params_,sort_keys=True),'validation_RMSE':-search.best_score_})
            cv_frame = pd.DataFrame(search.cv_results_)
            cv_frame.to_csv(ROOT/'results'/f'cv_{metal}_{name}.csv',index=False)
            joblib.dump(search.best_estimator_,ROOT/'models'/f'{metal}_{name}.joblib')
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(ROOT/'results/test_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(ROOT/'results/test_predictions.csv',index=False)
    pd.DataFrame(searches).to_csv(ROOT/'results/best_parameters.csv',index=False)
    pd.DataFrame(baseline).to_csv(ROOT/'results/training_mean_baseline.csv',index=False)
    summary = metrics.groupby('model').agg(mean_R2=('R2','mean'),mean_NRMSE_percent_full_scale=('NRMSE_percent_full_scale','mean')).sort_values('mean_NRMSE_percent_full_scale')
    summary.to_csv(ROOT/'results/model_summary.csv')
    print(summary.to_string(),flush=True)
    return metrics


def predict_image(image_path):
    features,_,_,_,qc = preprocess_image(Path(image_path))
    X = pd.DataFrame([features],columns=FEATURES)
    rows = []
    for metal in METALS:
        for family in ['PLS','RandomForest','XGBoost','RBF_SVR']:
            model = joblib.load(ROOT/'models'/f'{metal}_{family}.joblib')
            rows.append({'metal':metal,'model':family,'predicted_ug_L_simulated':float(np.asarray(model.predict(X)).reshape(-1)[0]),'training_data':'SYNTHETIC ONLY'})
    return pd.DataFrame(rows),qc


def make_charts(labels,metrics=None):
    # Illustrations are fixed examples, not selected for favorable predictions.
    selected = labels.iloc[np.linspace(0,len(labels)-1,12,dtype=int)]
    fig,axes = plt.subplots(3,4,figsize=(16,10))
    for ax,row in zip(axes.ravel(),selected.itertuples()):
        ax.imshow(cv2.cvtColor(cv2.imread(str(ROOT/row.image_path)),cv2.COLOR_BGR2RGB))
        ax.set_title(row.image_id,fontsize=9); ax.axis('off')
    fig.suptitle('500-image study: synthetic camera scenes — assumed seven-pad chemistry',fontsize=15)
    fig.tight_layout(); fig.savefig(ROOT/'charts/image_contact_sheet.png',dpi=140); plt.close(fig)
    example = labels.iloc[0]
    _,rectified,corrected,points,_ = preprocess_image(ROOT/example.image_path)
    raw = cv2.cvtColor(cv2.imread(str(ROOT/example.image_path)),cv2.COLOR_BGR2RGB)
    detected = raw.copy()
    cv2.polylines(detected,[points.astype(int)],True,(255,65,20),2)
    for p in points.astype(int): cv2.circle(detected,tuple(p),13,(255,65,20),2)
    regions = corrected.copy()
    for x,y in SPOTS: cv2.circle(regions,(x,y),16,(230,25,25),2)
    fig,axes = plt.subplots(2,2,figsize=(13,8))
    for ax,img,title in zip(axes.ravel(),[detected,rectified,corrected,regions],['1. JPEG: detect four markers','2. Perspective correction','3. Reference normalization + denoising','4. Extract 9 color features from each of 7 regions']):
        ax.imshow(img); ax.set_title(title,fontsize=12); ax.axis('off')
    fig.suptitle('Synthetic image preprocessing — no concentration labels used',fontsize=15)
    fig.tight_layout(); fig.savefig(ROOT/'charts/preprocessing_steps.png',dpi=150); plt.close(fig)
    if metrics is not None:
        fig,axes = plt.subplots(1,2,figsize=(14,5))
        for ax,metric,title in zip(axes,['NRMSE_percent_full_scale','R2'],['Test RMSE (% simulated full scale)','Test R²']):
            metrics.pivot(index='metal',columns='model',values=metric).loc[METALS].plot.bar(ax=ax)
            ax.set_title(title); ax.set_xlabel('Metal'); ax.legend(fontsize=8)
            ax.text(.5,.5,'SYNTHETIC ONLY',transform=ax.transAxes,ha='center',alpha=.16,fontsize=24,rotation=20)
        fig.suptitle('Seven-spot image simulation: held-out session evaluation')
        fig.tight_layout(); fig.savefig(ROOT/'charts/model_comparison.png',dpi=160); plt.close(fig)


def verify_study(labels,features):
    tr,te = labels[labels.partition=='train'],labels[labels.partition=='test']
    assert len(tr)==400 and len(te)==100
    assert tr.sample_id.nunique()==200 and te.sample_id.nunique()==50
    assert tr.session_id.nunique()==40 and te.session_id.nunique()==10
    assert not set(tr.session_id)&set(te.session_id)
    assert not set(tr.sample_id)&set(te.sample_id)
    assert len(list((ROOT/'datasets/images').glob('*.jpg')))==500
    assert len(list((ROOT/'datasets/preprocessed').glob('*.png')))==500
    assert features[FEATURES].shape==(500,63) and np.isfinite(features[FEATURES].to_numpy()).all()
    assert not set(FEATURES)&set(TARGETS)
    for row in labels.itertuples():
        assert cv2.imread(str(ROOT/row.image_path)).shape==(480,640,3)
    # Re-extract a held-out image from its pixels and compare with the saved feature record.
    row = labels[labels.partition=='test'].iloc[0]
    extracted,*_ = preprocess_image(ROOT/row.image_path)
    saved = features.loc[features.image_id==row.image_id,FEATURES].iloc[0].to_numpy(dtype=float)
    assert np.allclose(saved,[extracted[c] for c in FEATURES],atol=1e-7)
    metric_path = ROOT/'results/test_metrics.csv'
    if metric_path.exists():
        metrics = pd.read_csv(metric_path)
        predictions = pd.read_csv(ROOT/'results/test_predictions.csv')
        assert len(metrics)==28 and len(predictions)==2800
        for (metal,model),part in predictions.groupby(['metal','model']):
            actual,pred = part.actual_ug_L_simulated,part.predicted_ug_L_simulated
            m = metrics[(metrics.metal==metal)&(metrics.model==model)].iloc[0]
            assert np.allclose([mean_absolute_error(actual,pred),mean_squared_error(actual,pred)**.5,r2_score(actual,pred)],m[['MAE','RMSE','R2']].to_numpy(dtype=float))
        assert len(list((ROOT/'models').glob('*.joblib')))==28
        inference,_ = predict_image(ROOT/row.image_path)
        expected = predictions[predictions.image_id==row.image_id].merge(inference,on=['metal','model'])
        assert len(expected)==28 and np.allclose(expected.predicted_ug_L_simulated_x,expected.predicted_ug_L_simulated_y)
    print('Verification passed: image counts, shapes, feature extraction and available prediction metrics.',flush=True)


def write_manifest():
    from importlib.metadata import version
    files = [p for folder in ['datasets','models','results','charts','design'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='manifest.json']
    manifest = {'synthetic_only':True,'seed':SEED,'images':500,'independent_profiles':250,'session_groups':50,
                'train_images':400,'test_images':100,'image_size':[640,480],'features':FEATURES,
                'concentration_maxima_ug_L_simulated':dict(zip(METALS,MAXIMA)),
                'python':platform.python_version(),
                'versions':{name:version(name) for name in ['numpy','pandas','scikit-learn','xgboost','joblib','matplotlib']},
                'opencv':cv2.__version__,'concept_image_role':'visual reference only, not source pixels for the numerical renderer',
                'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'results/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def run_all(train=True):
    labels = generate_images()
    features = extract_dataset(labels)
    make_charts(labels)
    verify_study(labels,features)
    metrics = train_models(labels,features) if train else None
    make_charts(labels,metrics)
    verify_study(labels,features)
    write_manifest()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-only',action='store_true')
    parser.add_argument('--train-existing',action='store_true')
    args = parser.parse_args()
    if args.train_existing:
        labels = pd.read_csv(ROOT/'datasets/labels.csv')
        features = pd.read_csv(ROOT/'datasets/features.csv')
        metrics = train_models(labels,features)
        make_charts(labels,metrics)
        verify_study(labels,features)
        write_manifest()
    else:
        run_all(train=not args.dataset_only)
