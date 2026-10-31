"""Local AquaSense prototype UI and ESP32 image receiver."""
import argparse
import csv
import io
import json
import os
import secrets
import sqlite3
import tempfile
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote
import cv2
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request, send_file, Response
from werkzeug.exceptions import HTTPException
from image_pipeline import ROOT
from improved_model import estimate_improved, load_improved
from quality_control import ImageRejected
from interpretation import interpret_prediction

def create_app(storage=None, token=None, predictor=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 3*1024*1024
    folder = Path(storage or ROOT/'runtime')
    folder.mkdir(parents=True, exist_ok=True)
    uploads = folder/'captures'; uploads.mkdir(exist_ok=True)
    database = folder/'measurements.sqlite3'
    access_token = token if token is not None else os.environ.get('AQUASENSE_API_TOKEN', '')
    predict = predictor or estimate_improved
    with closing(sqlite3.connect(database)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS measurements (id TEXT PRIMARY KEY, time_utc TEXT, sample_name TEXT, source TEXT, status TEXT, response TEXT, image_path TEXT, preview_path TEXT)')

    @app.before_request
    def check_access():
        if request.path.startswith('/api/'):
            if access_token and not secrets.compare_digest(request.headers.get('X-API-Key',''), access_token):
                return jsonify(error='An API token is required.'), 401
            origin = request.headers.get('Origin')
            if origin and urlparse(origin).netloc != request.host:
                return jsonify(error='Cross-origin requests are not enabled.'), 403

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get('/')
    def home():
        return render_template('index.html', token_required=bool(access_token))

    @app.get('/api/health')
    def health():
        try:
            config, _, _ = load_improved()
        except (FileNotFoundError, RuntimeError):
            return jsonify(status='unavailable', error='Build the improved model bundle first.'), 503
        return jsonify(status='ready', model_version=config['version'], mode='synthetic-only prototype', camera_target='ESP32-S3', real_world_validated=False)

    @app.get('/api/demo-image')
    def demo():
        import pandas as pd
        row = pd.read_csv(ROOT/'datasets/labels.csv').query("partition == 'test'").iloc[0]
        return send_file(ROOT/row.image_path, mimetype='image/jpeg')

    @app.post('/api/predict')
    def predict_route():
        if request.mimetype not in ['image/jpeg','image/png']:
            return jsonify(error='Send JPEG or PNG bytes with the matching Content-Type.'), 415
        body = request.get_data()
        try:
            with Image.open(io.BytesIO(body)) as img:
                if img.format not in ['JPEG','PNG'] or img.size != (640,480):
                    return jsonify(error='A JPEG or PNG capture at 640 × 480 is required.'), 422
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error='The file is not a valid supported image.'), 422
        sample_name = unquote(request.headers.get('X-Sample-Name','Untitled capture'))[:80]
        source = 'esp32-s3' if request.headers.get('X-Device') == 'esp32-s3' else 'upload'
        identifier = str(uuid.uuid4())
        # Validate through a temporary file; retain a normalized JPEG for history.
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.img', delete=False) as tmp:
            tmp.write(body); temp_path = Path(tmp.name)
        corrected = None
        try:
            result, corrected = predict(temp_path)
            code = 200
        except ImageRejected as error:
            result = {'status':'rejected','reasons':error.reasons,'qc':error.qc,'predictions':[], 'training_data':'SYNTHETIC ONLY'}
            code = 422
        except (FileNotFoundError, RuntimeError) as error:
            app.logger.error('Prediction bundle unavailable: %s', error)
            return jsonify(error='Prediction models are unavailable. Run the improved model build first.'), 503
        finally:
            temp_path.unlink(missing_ok=True)
        result['interpretation'] = interpret_prediction(result)
        raw_path = uploads/f'{identifier}.jpg'
        with Image.open(io.BytesIO(body)) as img:
            img.convert('RGB').save(raw_path, format='JPEG', quality=95)
        preview_path = None
        if corrected is not None:
            preview_path = uploads/f'{identifier}_corrected.png'
            if not cv2.imwrite(str(preview_path), cv2.cvtColor(corrected,cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not save corrected image')
        result.update(id=identifier,time_utc=datetime.now(timezone.utc).isoformat(),sample_name=sample_name,source=source)
        with closing(sqlite3.connect(database)) as db, db:
            db.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?)',
                       (identifier,result['time_utc'],sample_name,source,result['status'],json.dumps(result,allow_nan=False),str(raw_path),str(preview_path) if preview_path else None))
        return jsonify(result), code

    @app.get('/api/history')
    def history():
        with closing(sqlite3.connect(database)) as db, db:
            values = db.execute('SELECT response FROM measurements ORDER BY time_utc DESC LIMIT 100').fetchall()
        return jsonify([json.loads(value[0]) for value in values])

    @app.get('/api/measurements/<uuid:identifier>/preview')
    def preview(identifier):
        with closing(sqlite3.connect(database)) as db, db:
            row = db.execute('SELECT preview_path FROM measurements WHERE id=?',(str(identifier),)).fetchone()
        if not row or not row[0]: return jsonify(error='No corrected preview for this capture.'),404
        return send_file(row[0],mimetype='image/png')

    @app.get('/api/history.csv')
    def export():
        with closing(sqlite3.connect(database)) as db, db:
            records = [json.loads(row[0]) for row in db.execute('SELECT response FROM measurements ORDER BY time_utc')]
        stream = io.StringIO(); writer = csv.writer(stream)
        writer.writerow(['id','time_utc','sample_name','image_status','metal','model','estimate_ug_L_simulated','lower_90','upper_90','prediction_status','flags','training_data'])
        for record in records:
            name = record['sample_name']
            if name.lstrip().startswith(('=','+','-','@')): name = "'"+name
            for row in record.get('predictions') or [{}]:
                writer.writerow([record['id'],record['time_utc'],name,record['status'],row.get('metal'),row.get('model'),row.get('estimate_ug_L_simulated'),row.get('lower_90'),row.get('upper_90'),row.get('status'),'; '.join(row.get('flags',record.get('reasons',[]))),'SYNTHETIC ONLY'])
        return Response(stream.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=aquasense_history.csv'})

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8765)
    args = parser.parse_args()
    if args.host not in ['127.0.0.1','localhost','::1'] and not os.environ.get('AQUASENSE_API_TOKEN'):
        parser.error('Set AQUASENSE_API_TOKEN before enabling LAN access for the ESP32.')
    load_improved()
    create_app().run(host=args.host,port=args.port,debug=False,use_reloader=False)
