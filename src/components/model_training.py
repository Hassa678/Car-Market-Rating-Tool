import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)
from utils import save_objects

from exception import CustomException
from logger import logging
from utils import evaluate_models


@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = { 
            "Logistic Regression": LogisticRegression(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(), 
            "CatBoost": CatBoostClassifier(verbose=False),
            "AdaBoost": AdaBoostClassifier()
            }
            
        
            model_report:dict=evaluate_models(X_train=X_train,y_train=label_encoder.fit_transform(y_train),X_test=X_test,y_test=label_encoder.transform(y_test),
                                             models=models)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_objects(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            save_objects(
                file_path=os.path.join('artifact','label_encoder.pkl'),
                obj=label_encoder
            )

            predicted= label_encoder.inverse_transform(best_model.predict(X_test))

            accuracy_score_1 = accuracy_score(y_test, predicted)
            print (accuracy_score_1)
            return accuracy_score_1

        except Exception as e:
           raise CustomException(e,sys) 
        
    
    