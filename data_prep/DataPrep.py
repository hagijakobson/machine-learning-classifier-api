import pickle
import pandas as pd

class DataPrep(object):
    def __init__(self):
        self.feature_scaler = pickle.load(open('parameters/feature_scaler.pkl','rb'))
    
    def data_preparation(self, df):
        # feature scaling
        df = pd.DataFrame(self.feature_scaler.transform(df), index=df.index, columns=df.columns)
        return df    
