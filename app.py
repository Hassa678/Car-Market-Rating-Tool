from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.piplines.predict_pipline import CustomData,PredictPipeline


import pandas as pd

# Assuming df is your DataFrame
df = pd.read_csv('Notebook\cargurus.csv')
# Select only categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(exclude=['object']).columns
# Create an empty dictionary to store unique values
unique_values_dict = {}

# Iterate over each categorical column
for column in categorical_columns:
    # Get unique values for the column
    unique_values = df[column].unique()
    # Store the unique values in the dictionary
    unique_values_dict[column] = unique_values




application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            makeName=request.form.get('makeName'),
            modelName=request.form.get('modelName'),
            makeId=request.form.get('makeId'),
            carYear=int(request.form.get('carYear')),
            trimName=request.form.get('trimName'),
            localizedTransmission=request.form.get('localizedTransmission'),
             bodyTypeGroupId=request.form.get('bodyTypeGroupId'),
            bodyTypeName=request.form.get('bodyTypeName'),
            mileage=float(request.form.get('mileage')),
            exteriorColorName=request.form.get('exteriorColorName'),
            normalizedExteriorColor=request.form.get('normalizedExteriorColor'),
            priceDifferential=float(request.form.get('priceDifferential')),
            daysOnMarket=int(request.form.get('daysOnMarket')),
            dealRating=request.form.get('dealRating'),
            sellerId=int(request.form.get('sellerId')),
            listingPartnerId=int(request.form.get('listingPartnerId')),
            sellerPostalCode=int(request.form.get('sellerPostalCode')),
            distance=float(request.form.get('distance')),
            serviceProviderId=int(request.form.get('serviceProviderId')),
            serviceProviderName=request.form.get('serviceProviderName'),
            localizedDriveTrain=request.form.get('localizedDriveTrain'),
            localizedExteriorColor=request.form.get('localizedExteriorColor'),
            localizedInteriorColor=request.form.get('localizedInteriorColor'),
            sellerRating=float(request.form.get('sellerRating')),
            reviewCount=float(request.form.get('reviewCount')),
            howToShop=request.form.get('howToShop'),
            localizedFuelType=request.form.get('localizedFuelType'),
            localizedDoors=request.form.get('localizedDoors'),
            driveTrain=request.form.get('driveTrain'),
            localizedEngineDisplayName=request.form.get('localizedEngineDisplayName'),
            ncapOverallSafetyRating=request.form.get('ncapOverallSafetyRating'),
            interiorColor=request.form.get('interiorColor')
            )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0],cate_value=unique_values_dict,num_value=numerical_columns)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        