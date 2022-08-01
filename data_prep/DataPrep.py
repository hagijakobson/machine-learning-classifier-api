import pickle
import pandas as pd
import numpy as np

class DataPrep(object):
    def __init__(self):
        self.feature_scaler = pickle.load(open('parameters/feature_scaler.pkl','rb'))
    
    def data_preparation(self, df):
        # Turning features more Guassian
        # feature4
        df.loc[:,'feature4'] =  np.log1p(-1*df.loc[:,'feature4']).values

        # All features except 3 and 4
        columns = ['feature0', 'feature1', 'feature2', 'feature5',
               'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
               'feature11', 'feature12', 'feature13', 'feature14', 'feature15']
        i=0
        for column in df[columns]: 
            df.loc[:,column] = np.log1p(df.loc[:,column]).values
            i += 1
        
        # Drop highly correlated features
        df = df.drop(columns=['feature2','feature4','feature14'])

        # feature scaling
        df = pd.DataFrame(self.feature_scaler.transform(df), index=df.index, columns=df.columns)
        
        return df    
