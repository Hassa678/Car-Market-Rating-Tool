import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            label_encoding_path=os.path.join('artifact','label_encoder.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            label_encoder = load_object(file_path=label_encoding_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=label_encoder.inverse_transform(model.predict(data_scaled))
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 makeName: str,
                 modelName: str,
                 makeId: str,
                 carYear: int,
                 trimName: str,
                 localizedTransmission: str,
                 bodyTypeGroupId: str,
                 bodyTypeName: str,
                 mileage: float,
                 exteriorColorName: str,
                 normalizedExteriorColor: str,
                 priceDifferential: float,
                 daysOnMarket: int,
                 dealRating: str,
                 sellerId: int,
                 listingPartnerId: int,
                 sellerPostalCode: int,
                 distance: float,
                 serviceProviderId: int,
                 serviceProviderName: str,
                 localizedDriveTrain: str,
                 localizedExteriorColor: str,
                 localizedInteriorColor: str,
                 sellerRating: float,
                 reviewCount: float,
                 howToShop: str,
                 localizedFuelType: str,
                 localizedDoors: str,
                 driveTrain: str,
                 localizedEngineDisplayName: str,
                 ncapOverallSafetyRating: str,
                 interiorColor: str):

        self.makeName = makeName
        self.modelName = modelName
        self.makeId = makeId
        self.carYear = carYear
        self.trimName = trimName
        self.localizedTransmission = localizedTransmission
        self.bodyTypeGroupId = bodyTypeGroupId
        self.bodyTypeName = bodyTypeName
        self.mileage = mileage
        self.exteriorColorName = exteriorColorName
        self.normalizedExteriorColor = normalizedExteriorColor
        self.priceDifferential = priceDifferential
        self.daysOnMarket = daysOnMarket
        self.dealRating = dealRating
        self.sellerId = sellerId
        self.listingPartnerId = listingPartnerId
        self.sellerPostalCode = sellerPostalCode
        self.distance = distance
        self.serviceProviderId = serviceProviderId
        self.serviceProviderName = serviceProviderName
        self.localizedDriveTrain = localizedDriveTrain
        self.localizedExteriorColor = localizedExteriorColor
        self.localizedInteriorColor = localizedInteriorColor
        self.sellerRating = sellerRating
        self.reviewCount = reviewCount
        self.howToShop = howToShop
        self.localizedFuelType = localizedFuelType
        self.localizedDoors = localizedDoors
        self.driveTrain = driveTrain
        self.localizedEngineDisplayName = localizedEngineDisplayName
        self.ncapOverallSafetyRating = ncapOverallSafetyRating
        self.interiorColor = interiorColor

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "makeName": [self.makeName],
                "modelName": [self.modelName],
                "makeId": [self.makeId],
                "carYear": [self.carYear],
                "trimName": [self.trimName],
                "localizedTransmission": [self.localizedTransmission],
                "bodyTypeGroupId": [self.bodyTypeGroupId],
                "bodyTypeName": [self.bodyTypeName],
                "mileage": [self.mileage],
                "exteriorColorName": [self.exteriorColorName],
                "normalizedExteriorColor": [self.normalizedExteriorColor],
                "priceDifferential": [self.priceDifferential],
                "daysOnMarket": [self.daysOnMarket],
                "dealRating": [self.dealRating],
                "sellerId": [self.sellerId],
                "listingPartnerId": [self.listingPartnerId],
                "sellerPostalCode": [self.sellerPostalCode],
                "distance": [self.distance],
                "serviceProviderId": [self.serviceProviderId],
                "serviceProviderName": [self.serviceProviderName],
                "localizedDriveTrain": [self.localizedDriveTrain],
                "localizedExteriorColor": [self.localizedExteriorColor],
                "localizedInteriorColor": [self.localizedInteriorColor],
                "sellerRating": [self.sellerRating],
                "reviewCount": [self.reviewCount],
                "howToShop": [self.howToShop],
                "localizedFuelType": [self.localizedFuelType],
                "localizedDoors": [self.localizedDoors],
                "driveTrain": [self.driveTrain],
                "localizedEngineDisplayName": [self.localizedEngineDisplayName],
                "ncapOverallSafetyRating": [self.ncapOverallSafetyRating],
                "interiorColor": [self.interiorColor]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
