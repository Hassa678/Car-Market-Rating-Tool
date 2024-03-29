import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
            params = {
            "Decision Tree": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 30, 50],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            },
        "Random Forest": {
        'n_estimators': [100, 300, 500],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    "Gradient Boosting": {
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 500],
        'max_depth': [3, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    "XGBoost": {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 500],
        'max_depth': [3, 7],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 0.2],
        'reg_alpha': [0, 0.01],
        'reg_lambda': [0.1, 1.0]
    },
    "AdaBoost": {
        'n_estimators': [50, 250],
        'learning_rate': [0.01, 0.1],
    },
    "CatBoost": {
        'iterations': [100, 500],
        'learning_rate': [0.01, 0.1],
        'depth': [3, 7],
        'l2_leaf_reg': [1, 5]
    },
    "Logistic Regression": {
    }
}


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
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

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            print (r2_square)
            return r2_square

        except Exception as e:
           raise CustomException(e,sys) 
        
    
    