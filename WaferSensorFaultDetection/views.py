import pandas as pd
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from DataIngestion import DataIngestion
from DataPreProcessing import PreProcessing
from MLAlgo import modeltuner
from Utils import Utils


@csrf_exempt
def train(request):
    if request.method == "POST":
        try:
            path = request.body.decode('ascii')
            # Creating instances for user defined modules
            dataingestion = DataIngestion.DataIngestion()
            preProcessing = PreProcessing.PreProcessing()
            tuner = modeltuner.modelTuner()

            # Copying raw data from Local path to project folder
            dataingestion.rawDataLocal(path)

            # Transferring data to database
            dataingestion.LoadToDB('train')

            # Load data from database
            dataframe = dataingestion.LoadFromDB('train')

            # Splitting Features and labels
            features = dataframe.drop('Good/Bad', axis=1)
            labels = dataframe['Good/Bad']

            # Null values Imputation
            if preProcessing.nullValueCheck(features):
                features = preProcessing.KNNImputer(features, 'train')

            # Removing features with zero variance
            preProcessedData = preProcessing.zerovarcol(features, 'train')

            # Creating best performing model
            tuner.get_best_model(preProcessedData, labels)

            return HttpResponse('done')
        except Exception as e:
            print('Exception occurred during training phase:', e)


@csrf_exempt
def predict(request):
    if request.method == "POST":
        path = request.body.decode('ascii')

        # Creating instances for user defined modules
        dataingestion = DataIngestion.DataIngestion()
        preProcessing = PreProcessing.PreProcessing()
        tuner = modeltuner.modelTuner()
        utils = Utils.utils()

        # Copying raw data from Local path to project folder
        dataingestion.rawDataLocal(path)

        # Transferring data to database
        dataingestion.LoadToDB('predict')

        # Retrieving data from database
        features = dataingestion.LoadFromDB('predict')

        # Null value imputation
        if preProcessing.nullValueCheck(features):
            features = preProcessing.KNNImputer(features, 'predict')

        # Removing features with null values
        preProcessedData = pd.DataFrame(preProcessing.zerovarcol(features, 'predict'))

        # Input data prediction
        predicted_data = tuner.predictData(preProcessedData)

        # Directory creation for saving output
        utils.dircheck('output')

        # Saving prediction results
        predicted_data.to_csv('output/predicted_data.csv', index=False)
        return HttpResponse('done')
