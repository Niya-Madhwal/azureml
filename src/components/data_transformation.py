import sys
import os
from dataclasses import dataclass
from src.logger import logging
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_fetaures= ["writing_score", "reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ])
            cat_pippeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Catergorcal column encdoing complelted ")
            logging.info("Standard scaling  complelted")

            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_fetaures),
                    ("cat_pipeline", cat_pippeline, categorical_features)
                ]

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data reading")

            preprocessor_obj= self.get_data_transformation_object()
            target_column_names="math_score"
            numerical_fetaures= ['writing_score', 'reading_ascore']

            input_feature_train_df =train_df.drop(columns=[target_column_names], axis=1)
            target_train_df= train_df[target_column_names]

            input_feature_test_df =test_df.drop(columns=[target_column_names], axis=1)
            target_test_df= test_df[target_column_names]

            logging.info(
                f"Applying preprocessing ontraining and testing dataframe"
            )

            input_feature_train_arr= preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_test_df)
                ]
            save_object(
                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path,
            )
                
        except Exception as e:
            raise CustomException(e, sys)