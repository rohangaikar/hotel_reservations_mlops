import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml,load_data
from config.model_params import *
from config.paths_config import *
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


    def load_and_split(self):
        try:
            logger.info(f'Loading data from {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f'Loading data from {self.train_path}')
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns='booking_status')
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns='booking_status')
            y_test = test_df['booking_status']

            logger.info('Data Splitted successfully for training')

            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error(f'Error occured during loading and splitting processed data : {e}')
            raise CustomException('Error during loading and splitting data in model training step',e)
        
    def train_model(self,X_train,y_train):
        try:
            logger.info('Initializing the model')
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info('Setting Parameters for hyper-parameter tuning')
            random_search = RandomizedSearchCV(
                estimator = lgbm_model,
                param_distributions = self.params_dist,
                n_iter = self.random_search_params['n_iter'],
                cv = self.random_search_params['cv'],
                verbose = self.random_search_params['verbose'],
                n_jobs = self.random_search_params['n_jobs'],
                random_state = self.random_search_params['random_state'],
                scoring = self.random_search_params['scoring'],
            )
            logger.info('Started training of the model')
            random_search.fit(X_train,y_train)
            logger.info('Hyper Parameter tuning done')

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f'The Best Params are : {best_params}')

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f'Error occured during training the model : {e}')
            raise CustomException('Error during training the model in model training step',e)
        
    def evaluate_model(self,best_lgbm_model,X_test,y_test):
        try:
            logger.info('Started model evlaution againts test data')
            y_pred = best_lgbm_model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            roc_auc = roc_auc_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)

            logger.info(f'Model is evaluated. The following are the metric values.')
            logger.info(f'Accuracy : {accuracy}')
            logger.info(f'ROC AUC : {roc_auc}')
            logger.info(f'F1 : {f1}')
            logger.info(f'Recall : {recall}')
            logger.info(f'Precision : {precision}')

            return {'accuracy' : accuracy,
                    'roc_auc' : roc_auc,
                    'f1' : f1,
                    'recall' : recall,
                    'precision' : precision}

        except Exception as e:
            logger.error(f'Error occured during evaluating the model : {e}')
            raise CustomException('Error during evaulation of the model in model training step',e)
        
    
    def save_model(self,model):
        try:
            logger.info('Saving model to model output path')
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            joblib.dump(model,self.model_output_path)
            logger.info(f'Model saved to {self.model_output_path}')

        except Exception as e:
            logger.error(f'Error occured during saving the model : {e}')
            raise CustomException('Error during saving the model pickle file in model training step',e)
    
    def train(self):
        try:
            with mlflow.start_run():
                logger.info('Starting Model training pipeline')

                logger.info('Starting MLFlow Experimentation')

                logger.info('Logging the training and testing dataset to MLFlow')
                mlflow.log_artifact(self.train_path,artifact_path='datasets')
                mlflow.log_artifact(self.test_path,artifact_path='datasets')
                X_train,y_train,X_test,y_test = self.load_and_split()

                logger.info('Logging best model, params and metrics in MLFlow')
                best_lgbm_model = self.train_model(X_train,y_train)
                mlflow.log_artifact(self.model_output_path)
                mlflow.log_params(params = best_lgbm_model.get_params())

                metrics =  self.evaluate_model(best_lgbm_model,X_test,y_test)
                mlflow.log_metrics(metrics = metrics)

                self.save_model(best_lgbm_model)

        except Exception as e:
            logger.error(f'Error occured during running model training pipeline : {e}')
            raise CustomException('Error during Model Training pipeline',e)
        
if __name__ == '__main__':
    trainer = ModelTrainer(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.train()
