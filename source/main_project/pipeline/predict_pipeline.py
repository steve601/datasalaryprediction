import sys
import pandas as pd
from source.commons import load_object
from source.exception import UserException
from source.logger import logging

class PredicPipeline:
    def __init__(self):
        pass
    
    logging.info('Preprocessing user input and making predictions')
    def predict(self,features):
        model_path = 'elements\model.pkl'
        preprocessor_path = 'elements\preprocessor.pkl'
        # loaeding objects
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        data_processed = preprocessor.transform(features)
        prediction = model.predict(data_processed)
        
        return prediction
logging.info('This class is responsible for mapping all the inputs from html to flask')
class UserData:
    def __init__(self,designation,age,unit,ratings,experience):
        self.des = designation
        self.age = age
        self.unit = unit
        self.rate = ratings
        self.exp = experience
        
    # let's write a function that returns the user input as a pandas dataframe
    def get_data_as_df(self):
        try:
            user_data = {
                "designation":[self.des],
                "age":[self.age],
                "unit":[self.unit],
                "ratings":[self.rate],
                "experience":[self.exp],
            }
            return pd.DataFrame(user_data)
        except Exception as e:
            raise UserException(e,sys)
        