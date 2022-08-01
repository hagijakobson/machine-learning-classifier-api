import pandas as pd
import pickle
from flask import Flask, request
from data_prep.DataPrep import DataPrep
import os

# load model
model = pickle.load(open('model/model.pkl', 'rb'))

# instantiate flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()
    
    # collect data
    if test_json:
        if isinstance(test_json, dict): # unique value
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    # instantiate data preparation
    pipeline = DataPrep()

    # data data preparation
    df_processed = pipeline.data_preparation(df_raw.copy())
    
    # prediction
    pred = model.predict(df_processed)
    
    df_raw['prediction'] = pred
    
    return df_raw.to_json(orient='records')
    
if __name__ == '__main__':
    # start flask
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port, debug=False)
    