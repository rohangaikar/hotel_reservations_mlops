from src.logger import get_logger
from src.custom_exception import CustomException
import sys


logger = get_logger(__name__)

def divide_number(a,b):
    try:
        logging.info('Dividing two numbers..')
        return result
    except Exception as e:
        logger.error('Error has occured')
        raise CustomException('Custom Error: You are dividing by ZERO...',sys)

if __name__ == '__main__':
    try:
        logger.info('Starting the main program')
        divide_number(10,2)
    except CustomException as ce:
        logger.error(str(ce))


