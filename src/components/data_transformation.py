import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.logger import logging
from src.exception import CustomException
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        logging.info("entered into tranformer method")
        try:
            ## getting the features for transformation
            num_features  = ["reading_score","writing_score"]
            cat_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]


            # creating pipeline for num features
            num_pipeline = Pipeline(
                steps=[
                    ("simpleimputer",SimpleImputer(strategy="median")),
                    ("standard scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ohe",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            transformer = ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            logging.info("numerical transformation is completed")
            logging.info("categorical transformation is completed")
            return transformer
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading train and test data completed")

            logging.info("obtaining preprocessor objects")

            preprocessor_obj = self.get_data_transformer_obj()
            target_column = "math_score"

            input_feature_train_df= train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df= test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying transformer on train and test data for imputing and scaling data")

            input_features_train_transformed = preprocessor_obj.fit_transform(input_feature_train_df)
            input_features_test_transformed = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_features_train_transformed,np.array(input_feature_train_df)]
            test_arr = np.c_[input_features_test_transformed,np.array(input_feature_test_df)]

            logging.info("saving preprocessing file")
            # here save object function is written in utils save object is used to save pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)



