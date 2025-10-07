import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from src.components.data_ingestion import DataIngestion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
import os

class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_colums=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethinicity",
                "parental_level_of_education",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median"))
                    ("scaler",StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder",OneHotEncoder()),
                        ("scaler",StandardScaler())
                    ]
                )
            logging.info(f"categorical coumns encoding completed : {categorical_columns}")
            logging.info(f"numerical columns encoding completed : {numerical_colums}")



            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_colums),
                    ("cat_pipeline",categorical_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self,train_path,test_path);
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"applying preprocessor obj on train and test"
            )

            in_feat_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            in_feat_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                in_feat_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                in_feat_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformer(train_data,test_data)

