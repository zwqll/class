from concurrent.futures import ThreadPoolExecutor
from flask import request
from flask import Response
from flask import Flask
import json
from text_predict import *
from text_train import train as text_train

app = Flask(__name__)
predictor_factory = PredictorFactory()


@app.route('/PredictOne', methods=['POST'])
def predict_one():
    print('Enter Text CNN Predict')
    try:
        text = request.form.get('text')
        model = request.form.get('model')
        predictor = predictor_factory.get_predictor(model)
        return_data = predictor.predict(text)

    except Exception as e:
        print(e)
        return_data = {
            'status': 'error',
            'id': '',
            'error': format(e)
        }
    js = json.dumps(return_data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


@app.route('/PredictOneBi', methods=['POST'])
def predict_one_bi():
    print('Enter Text CNN Predict')
    try:
        # model = request.form.get('model')
        text = request.form.get('text')
        return_data = predict_bi(text)
    except Exception as e:
        print(e)
        return_data = {
            'status': 'error',
            'id': '',
            'error': format(e)
        }
    js = json.dumps(return_data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


@app.route('/Train', methods=['POST'])
def train():
    print('Enter Text CNN Predict')
    try:
        model_name = request.form.get('model')
        executor = ThreadPoolExecutor(1)
        executor.submit(text_train, model_name)
        return_data = {
            'status': '200',
            'id': ''
        }
    except Exception as e:
        print(e)
        return_data = {
            'status': 'error',
            'id': '',
            'error': format(e)
        }
    js = json.dumps(return_data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


app.run(debug=False, host='0.0.0.0', port=9988)
