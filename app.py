from flask import Flask, request, jsonify
import pickle
import sklearn
import numpy as np

model = pickle.load(open('RF_58.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods = ['POST'])
def predict():
    pulse_BPM = request.form.get('pulse_BPM')
    Temperature = request.form.get('Temperature')
    GSR = request.form.get('GSR')

    input_query = np.array([[pulse_BPM, Temperature, GSR]])

    result = model.predict(input_query)[0]

    return jsonify({'emotion':str(result)})

if __name__ == '__main__':
    app.run(debug=True)