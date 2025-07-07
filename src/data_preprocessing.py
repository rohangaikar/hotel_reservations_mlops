import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data,read_yaml

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self,df):
        try:
            logger.info('Started Pre-Processing of Data...')

            logger.info('Dropping the columns')
            df.drop('Booking_ID',axis = 1,inplace = True)

            logger.info('Dropping Duplicates...')
            df.drop_duplicates(inplace = True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info('Applying Label Encoding')
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}

            logger.info('Label Mappings are :')
            for label,code in mappings.items():
                logger.info(f'{label}:{code} \n')

            logger.info('Log Scaling the data for skewness handling')
            skewness_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x: x.skew())
            for col in skewness[skewness>skewness_threshold].index:
                df[col] = np.log1p(df[col])
            
            return df

        except Exception as e:
            logger.error(f'Error during preprocessing step {e}')
            raise CustomException('Error in data preprocessing step',e)
        
    def balance_data(self,df):
        try:
            logger.info('Handling Imbalance in the data')
            X = df.drop('booking_status',axis = 1)
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            balance_df = pd.DataFrame(X_resampled,columns=X.columns)
            balance_df['booking_status'] = y_resampled    

            logger.info('Data Balancing complete')

            return balance_df
        
        except Exception as e:
            logger.error(f'Error during balancing step {e}')
            raise CustomException('Error in data balancing step',e)
        
    
    def select_features(self,df):
        try:
            logger.info('Starting our Feature Selection in the data')
            X = df.drop('booking_status',axis = 1)
            y = df['booking_status']
            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)
            feature_importances = model.feature_importances_

            feature_importances_df = pd.DataFrame(
            {
                'features' : X.columns,
                'importance':feature_importances
            }
            )
            feature_importances_df = feature_importances_df.sort_values('importance',ascending=False)

            top_n_features = self.config['data_processing']['top_n_features']
            top_n_features = feature_importances_df[:top_n_features].features.values
            top_n_df = df[top_n_features.tolist() + ['booking_status']]

            logger.info('Feature Selection Completed')

            return top_n_df
        
        except Exception as e:
            logger.error(f'Error during feature selection step {e}')
            raise CustomException('Error in feature selection step',e)
        
    def save_data(self,df,file_path):
        try:
            logger.info('Saving our Data in processed folder')
            df.to_csv(file_path,index = False)
            logger.info(f'Data saved successfully to {file_path}')

        except Exception as e:
            logger.error(f'Error during saving data step {e}')
            raise CustomException('Error while saving data',e)
        
    def process(self):
        try:
            logger.info('Loading Data from Raw Directory')
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            logger.info('Data Loaded Sucessfully from Raw Directory')

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]


            train_df = self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            test_df = self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info('All Data Processing Steps completed successfully on both train and test data')
        except Exception as e:
            logger.error(f'Error during final data process pipeline {e}')
            raise CustomException('Error while data processing pipeline',e)
        

if __name__ == '__main__':
    data_processor = DataProcessor(
        train_path = TRAIN_FILE_PATH,
        test_path = TEST_FILE_PATH,
        processed_dir = PROCESSED_DIR,
        config_path = CONFIG_PATH
        )
    data_processor.process()


                
            

