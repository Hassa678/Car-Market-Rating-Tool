import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)
from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation
from components.model_training import ModelTrainer, ModelTrainingConfig
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train_data.csv')
    test_data_path: str = os.path.join('artifact', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
        # Read the dataset from CSV
            df = pd.read_csv('artifact/data.csv')
        
        # Drop unnecessary column and duplicate rows
            df = df.drop('Unnamed: 0', axis=1)
            df = df.drop_duplicates()
            df.reset_index(drop=True, inplace=True)
        

        
        # Impute missing values in categorical column ('dealRating') with most frequent category
            imp_most_frequent = SimpleImputer(strategy='most_frequent')
            df[['dealRating']] = imp_most_frequent.fit_transform(df[['dealRating']])
        

        # Split data into train and test sets
            train_set, test_set = train_test_split(df, random_state=42, test_size=0.2)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data split completed")
        
            return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    #obj.initiate_data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)