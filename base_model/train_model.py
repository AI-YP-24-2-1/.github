import pandas as pd
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import time
from sklearn.metrics import r2_score, mean_squared_error as MSE, root_mean_squared_error as RMSE


class BaseModel:
    def __init__(self):
        self.models = {
            'pollen': {
                'regressor': {'alder_pollen': SGDRegressor(),
                    'birch_pollen': SGDRegressor(),
                    'grass_pollen': SGDRegressor(),
                    'mugwort_pollen': SGDRegressor(),
                    'olive_pollen': SGDRegressor(),
                    'ragweed_pollen': SGDRegressor()
                    },
                'scaler': {'pollen_scaler': StandardScaler()},
                'path': {
                    'train_x_path': 'csv_data/train_x_pollen.csv',
                    'train_y_path': 'csv_data/train_y_pollen.csv',
                    'test_x_path': 'csv_data/test_x_pollen.csv',
                    'test_y_path': 'csv_data/test_y_pollen.csv',
                    'pickle_path': 'pickle_files/european_aqi_model.pickle'
                }
            },
            'aqi': {
                'regressor': {'european_aqi': SGDRegressor()},
                'scaler': {'pollen_scaler': StandardScaler()},
                'path': {
                    'train_x_path': 'csv_data/train_x_aqi.csv',
                    'train_y_path': 'csv_data/train_y_aqi.csv',
                    'test_x_path': 'csv_data/test_x_aqi.csv',
                    'test_y_path': 'csv_data/test_y_aqi.csv',
                    'pickle_path': ''
                }
                }
        }
    
    def log(self, mode, text, *args):
        text = f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: ({mode}) {text.format(*args)}'

        with open('log.txt', 'a') as log_file:
             log_file.write(text + '\n')
        print(text)
    
    def train_model(self, chunksize, mode='aqi'):
        rows = 1

        while True:
            self.log(mode, 'Reading train data')
            train_x = pd.read_csv(self.models[mode]['path']['train_x_path'], skiprows=range(1, rows), nrows=chunksize, index_col=0, header=0).select_dtypes(['float', 'int'])
            train_y = pd.read_csv(self.models[mode]['path']['train_y_path'], skiprows=range(1, rows), nrows=chunksize)
            self.log(mode, 'Data read')

            for column in train_x.columns:
                median_value = train_x[column].median() if not train_x[column].isna().all() else 0
                train_x[column] = train_x[column].fillna(median_value)
            self.log(mode, 'np.nan filled')

            train_x = pd.DataFrame(self.models[mode]['scaler']['pollen_scaler'].fit_transform(train_x), columns=train_x.columns)

            for model_name in self.models[mode]['regressor'].keys():
                self.log(mode, 'training  model')
                model = self.models[mode]['regressor'][model_name]
                train_y_model = np.array(train_y[model_name].fillna(0)).ravel()
                model.partial_fit(train_x, train_y_model)

            rows += train_y.shape[0]
            self.log(mode, '{} processed', rows-1)

            if train_y.shape[0] != chunksize:
                break
        
        for model_name in self.models[mode]['regressor'].keys():
            joblib.dump(self.models[mode]['regressor'][model_name], f'pickle_files/{model_name}_model.pickle')


    def model_quality(self, mode) -> None:
        self.log(mode, 'Reading train data')
        test_x = pd.read_csv(self.models[mode]['path']['test_x_path']).select_dtypes(['int', 'float'])
        test_y = pd.read_csv(self.models[mode]['path']['test_y_path']).select_dtypes(['int', 'float'])
        self.log(mode, 'Data read')

        for column in test_x.columns:
            median_value = test_x[column].median() if not test_x[column].isna().all() else 0
            test_x[column] = test_x[column].fillna(median_value)

        for column in test_y.columns:
            test_y[column] = test_y[column].fillna(0) if mode=='pollen' else test_y[column].fillna(test_y[column].median())

        self.log(mode, 'np.nan filled')

        for model_name in self.models[mode]['regressor'].keys():
            model = joblib.load(f'pickle_files/{model_name}_model.pickle')
            pred = model.predict(test_x)
            self.log(mode, '{} predicted', model_name)

            r2 = r2_score(test_y[model_name], pred)
            mse = MSE(test_y[model_name], pred)
            rmse = RMSE(test_y[model_name], pred)
            self.log(mode, '\n---------------RESULTS:\nModel: {}\nr2: {}\nMSE: {}\nRMSE: {}\n---------------', model_name, r2, mse, rmse)


base_model = BaseModel()
base_model.train_model(30000000, mode='aqi')
base_model.model_quality(mode='aqi')

base_model.train_model(30000000, mode='pollen')
base_model.model_quality(mode='pollen')
