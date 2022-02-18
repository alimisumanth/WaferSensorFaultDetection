import pandas as pd
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from DataIngestion import DataIngestion
from DataPreProcessing import PreProcessing
from MLAlgo import modeltuner
from Utils import Utils
from WaferLogging import WaferLogging


@csrf_exempt
def train(request):
    if request.method == "POST":
        try:
            path = request.body.decode('ascii')
            # Creating instances for user defined modules
            dataingestion = DataIngestion.DataIngestion()
            preProcessing = PreProcessing.PreProcessing()
            tuner = modeltuner.modelTuner()
            WaferLogger = WaferLogging.WaferLogging()
            logger = WaferLogger.getLogger('trainingPhase')

            logger.info('Training Phase Started')
            logger.info('Raw data copying to local path started')
            # Copying raw data from Local path to project folder
            dataingestion.rawDataLocal(path)
            logger.info('Raw data copying to local path ended')

            logger.info('Data Transmission phase from Local to Database started')
            # Transferring data to database
            dataingestion.LoadToDB('train')
            logger.info('Data Transmission phase from Local to Database ended')

            logger.info('Data retrieval phase from database to Local started')
            # Load data from database
            dataframe = dataingestion.LoadFromDB('training')
            logger.info('Data retrieval phase from database to Local ended')

            # Splitting Features and labels
            features = dataframe.drop('Good/Bad', axis=1)
            labels = dataframe['Good/Bad']

            logger.info('Data preprocessing phase started')
            # Null values Imputation
            if preProcessing.nullValueCheck(features):
                features = preProcessing.KNNImputer(features, 'training')

            # Removing features with zero variance
            preProcessedData = preProcessing.zerovarcol(features, 'training')
            logger.info('Data preprocessing phase ended')

            logger.info('Model tuner phase started')
            # Creating best performing model
            tuner.get_best_model(preProcessedData, labels)
            logger.info('Model tuner phase ended')

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
        features = dataingestion.LoadFromDB('prediction')

        # Null value imputation
        if preProcessing.nullValueCheck(features):
            features = preProcessing.KNNImputer(features, 'prediction')

        # Removing features with null values
        preProcessedData = pd.DataFrame(preProcessing.zerovarcol(features, 'prediction'))

        # Input data prediction
        predicted_data = tuner.predictData(preProcessedData)

        # Directory creation for saving output
        utils.dirCheck('output')

        # Saving prediction results
        predicted_data.to_csv('output/predicted_data.csv', index=False)
        return HttpResponse('done')
