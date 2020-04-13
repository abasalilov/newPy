
import numpy as np
from flask import Flask, request, jsonify, render_template

from getPrediction import getImage, getPrediction

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction = {}
    url = request.get_json(force=True)
    print('url', url)
    imageReady = getImage(url)
    prediction = getPrediction()
    output = prediction

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = 'model.predict([np.array(list(data.values()))])'

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
