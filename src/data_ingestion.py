import os
import pandas as pd 
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/rohangaikar/Downloads/compact-arc-464706-r2-c4a45874f3ae.json"

class DataIngestion:
    def __init__(self,config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_ratio = self.config['train_ratio']
        os.makedirs(RAW_DIR,exist_ok = True)
        logger.info(f"Data Ingestion started with bucket - {self.bucket_name} and file - {self.file_name}..")

    def download_csv_from_gcp(self):
        try:
            logger.info('Connecting to GCP client')
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f'Downloaded CSV {self.file_name} to the path {RAW_FILE_PATH}...')

        except Exception as e:
            logger.error('Error while downloading the file..')
            raise CustomException('Failed to download csv file',e)
    
    def split_data(self):
        try:
            logger.info('Started Splitting the data')
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data,test_size = 1 - self.train_ratio,random_state = 42)
            train_data.to_csv(TRAIN_FILE_PATH,index = False)
            test_data.to_csv(TEST_FILE_PATH,index = False)
            logger.info('Train Data Saved to Train file path..')
            logger.info('Test Data Saved to Test file path..')

        except Exception as e:
            logger.error('Error while spliting the data..')
            raise CustomException('Failed to split the data',e)

    def run(self):
        try:
            logger.info('Started Data Ingestion process')
            self.download_csv_from_gcp()
            self.split_data()
            logger.info('Data Ingestion process ran successfully')
        except CustomException as ce:
            logger.error(f'Custom Exception :  {str(ce)}')

        finally:
            logger.info('Closing Data Ingestion process')

if __name__ == '__main__':
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()




