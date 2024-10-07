import sys
import os
from dataclasses import dataclass
import requests
import numpy as np
import pandas as pd
from typing import List, Dict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from exception import CustomException
from logger import logging
from utils import save_objects
from constants import cookies, headers, params

@dataclass
class CarData:
    data: pd.DataFrame

    @classmethod
    def from_api(cls, cookies: Dict, headers: Dict, params: Dict) -> 'CarData':
        all_products = []
        for page_number in range(1, 89):
            params['pageNumber'] = str(page_number)
            response = requests.get('https://www.cargurus.com/Cars/searchPage.action', params=params, cookies=cookies, headers=headers)
            for tile in response.json()['tiles']:
                try:
                    tile['data']['makeName']
                    all_products.append(tile['data'])
                except KeyError:
                    pass
        return cls(pd.json_normalize(all_products))

    def drop_columns_with_one_unique_value(self) -> None:
        columns_to_drop = [
            column for column in self.data.columns
            if len(self.data[column].explode().unique()) == 1
        ]
        self.data.drop(columns=columns_to_drop, axis=1, inplace=True)

    def remove_columns_with_significant_missing_values(self, threshold: int = 1700) -> None:
        columns_to_drop = [
            column for column in self.data.columns
            if self.data[column].explode().count() < threshold
        ]
        self.data.drop(columns=columns_to_drop, axis=1, inplace=True)

    def remove_redundant_columns_and_duplicates(self) -> None:
        redundant_columns = [
            'id', 'unitMileage.unit', 'mileageString', 'unitMileage.value', 'offset',
            'priceString', 'expectedPriceString', 'priceDifferentialString',
            'phoneNumberString', 'phoneNumber', 'originalPictureData.url',
            'originalPictureData.height', 'originalPictureData.width', 'vin',
            'sellerRegion', 'sellerCity', 'listingTitle', 'options', 'dealScore',
            'price', 'expectedPrice', 'stockNumber', 'modelId'
        ]
        self.data.drop(columns=redundant_columns, axis=1, inplace=True)
        self.data.drop_duplicates(inplace=True)

    def save_data(self, filename: str = 'data.csv') -> None:
        self.data.to_csv(os.path.join('artifact', filename))

class DataProcessor:
    def __init__(self, cookies: Dict, headers: Dict, params: Dict):
        self.cookies = cookies
        self.headers = headers
        self.params = params

    def process(self) -> None:
        try:
            car_data = CarData.from_api(self.cookies, self.headers, self.params)
            
            logging.info("Dropping columns with one unique value")
            car_data.drop_columns_with_one_unique_value()

            logging.info("Removing columns with significant missing values")
            car_data.remove_columns_with_significant_missing_values()

            logging.info("Removing redundant columns and duplicates")
            car_data.remove_redundant_columns_and_duplicates()

            logging.info("Saving processed data")
            car_data.save_data()

            logging.info("Data processing completed successfully")
        except Exception as e:
            logging.error(f"An error occurred during data processing: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    processor = DataProcessor(cookies, headers, params)
    processor.process()