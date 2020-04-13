
import numpy as np
import asyncio
from flask import Flask, request, jsonify, render_template

import get_prediction

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction = "~~~~~~~~~~here"
    data = request.get_json(force=True)
    url = data['url']
    # return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
    return jsonify(get_prediction.getImage(url))


@app.route('/get-prediction', methods=['POST'])
def results():
    print('data')
    return jsonify(get_prediction.runUsingModel())


if __name__ == "__main__":
    app.run(debug=True)
