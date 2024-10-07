from flask import Flask, request, render_template
from src.piplines.predict_pipline import CustomData, PredictPipeline
import pandas as pd

application = Flask(__name__)

app = application

# Load the data
df = pd.read_csv('Notebook/cargurus.csv')
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('dealRating')
numerical_columns = ['carYear', 'mileage', 'priceDifferential', 'daysOnMarket',
                     'distance'
                     ]

        # Calculate unique values dynamically based on the form data
unique_values_dict = {}
for column in categorical_columns:
    unique_values = df[column].unique()
    unique_values_dict[column] = unique_values

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', cate_value=unique_values_dict, num_value=numerical_columns)
    else:
        # Create CustomData object from form data
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

        # Perform prediction
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0],cate_value=unique_values_dict, num_value=numerical_columns)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
