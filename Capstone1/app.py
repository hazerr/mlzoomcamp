import json
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the model

modelRegre = pickle.load(open('modelRegression.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    new_data_json = request.json['data']
    data2 = new_data_json.get('data', {})
    df_new = pd.DataFrame([data2])
    print(df_new)
    
    output = modelRegre.predict(df_new)
    print(output[0])
    
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    
    # Use preprocessor instead of fitting it again
    final_input = np.array([data])
    print(final_input)
    
    output = modelRegre.predict(final_input)[0]
    
    return render_template("home.html", prediction_text="The prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)