import numpy as np
from flask import Flask, request, jsonify ,render_template
import pickle
import pandas as pd

# get the model saved earlier
model = pickle.load(open("Diabetes.pkl", "rb"))  ##loading model





app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html')



@app.route('/detection_function',methods=['POST','GET'])
def detection_function():
    print("hello")
    if request.method=='POST':
        print("entered the chamber")
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        blood_pressure = request.form['bp']
        skin_thickness = request.form['skinthickness']
        bmi = request.form['bmi']
        insulin = request.form['insulin']
        dpfunc = request.form['dpfunc']
        age=request.form['age']
        row_df = pd.DataFrame([pd.Series([pregnancies,glucose,blood_pressure,skin_thickness,bmi,insulin,dpfunc,age])])  ### Creating a dataframe using all the values
        prediction=model.predict_proba(row_df) ## Predicting the output
        output='{0:.{1}f}'.format(prediction[0][1], 2)    ## Formating output
        if output>str(0.5):
            return render_template('index.html',pred='You have chance of having diabetes.') ## Returning the message for use on the same index.html page
        else:
            return render_template('index.html',pred='You are safe.') 







    return render_template('detection.html')
  
if __name__ == '__main__':
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run()