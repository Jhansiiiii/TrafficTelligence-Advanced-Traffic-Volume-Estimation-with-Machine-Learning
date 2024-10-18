import numpy as np 
import pickle
import matplotlib.pyplot as plt 
import time 
import pandas 
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and encoder
with open(r"C:\Users\DELL\Downloads\Bootslander (1)\Bootslander\model.pk1", 'rb') as file:
    model = pickle.load(file)

with open(r"C:\Users\DELL\Downloads\Bootslander (1)\Bootslander\encoder.pk1", 'rb') as file:
    scale = pickle.load(file)
print("Model's feature names:", model.feature_names_in_)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    names = [['holiday','temp', 'rain', 'snow', 'weather', 'year', 'month', 'day','hours', 'minutes', 'seconds']]
    data = pandas.DataFrame(features_values,columns=names)
     # predictions using the loaded model file
    prediction=model.predict(data)
    print(prediction)
    text = "Estimated Traffic Volume is :"
    return render_template("output.html",result = text + str(prediction) + "units")
     # showing the prediction results in a UI
if __name__ == "__main__":
    print("Model's feature names:", model.feature_names_in_)
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)