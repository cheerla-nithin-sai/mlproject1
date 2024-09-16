import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
## data injestion for getting and dealing and splitting the data

@dataclass
class DataInjestionConfig:
    #  here we are saving data into differet csv file in folder name aftifacts
    raw_data_path:str = os.path.join("artifacts","data.csv")
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")

class DataInjestion:
    def __init__(self):
        ## calling data injestion config class
        self.injestion_config=DataInjestionConfig()
    
    def initiate_data_injestion(self):
        logging.info("Entered the Data Injestion method or component")
        try:
            ## we can import data from anywhere here 
            df= pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset as dataframe")

            # to initiate different directories for the data
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)

            # saving df to csv file at raw data path
            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)

            logging.info("train test splits is initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            ## saving trainset dataframe to csv file at train data path
            train_set.to_csv(self.injestion_config.train_data_path,index=False,header=True)

            # saving testset to csv at test data path
            test_set.to_csv(self.injestion_config.test_data_path,index=False,header=True)

            logging.info("data injestion is completed")

            return (
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataInjestion()
    obj.initiate_data_injestion()



