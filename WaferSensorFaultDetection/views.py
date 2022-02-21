import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from DataIngestion import DataIngestion
from DataPreProcessing import PreProcessing
from MLAlgo import modeltuner
from Utils import Utils
from WaferLogging import WaferLogging




def home(request):
    return render(request, 'home.html')


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
            dataingestion.rawDataLocal(path, 'training')
            logger = WaferLogger.getLogger('trainingPhase')
            logger.info('Raw data copying to local path ended')

            logger.info('Data Transmission phase from Local to Database started')
            # Transferring data to database
            dataingestion.LoadToDB('training')
            logger = WaferLogger.getLogger('trainingPhase')
            logger.info('Data Transmission phase from Local to Database ended')

            logger.info('Data retrieval phase from database to Local started')
            # Load data from database
            dataframe = dataingestion.LoadFromDB('training')
            logger = WaferLogger.getLogger('trainingPhase')
            logger.info('Data retrieval phase from database to Local ended')

            # Splitting Features and labels
            features = dataframe.drop('Good/Bad', axis=1)
            labels = dataframe['Good/Bad']

            logger.info('Data preprocessing phase started')
            # Null values Imputation

            if preProcessing.nullValueCheck(features, 'training'):
                logger = WaferLogger.getLogger('trainingPhase')
                logger.info('Imputing missing values using KNN imputer')
                features = preProcessing.KNNImputer(features, 'training')
                logger = WaferLogger.getLogger('trainingPhase')
                logger.info('Missing value imputation completed')

            # Removing features with zero variance
            logger.info('Removing features with zero variance')
            preProcessedData = preProcessing.zerovarcol(features, 'training')
            logger = WaferLogger.getLogger('trainingPhase')
            logger.info('Data preprocessing phase ended')

            logger.info('Model tuner phase started')
            # Creating best performing model
            tuner.get_best_model(preProcessedData, labels)
            logger = WaferLogger.getLogger('trainingPhase')
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
        WaferLogger = WaferLogging.WaferLogging()
        logger = WaferLogger.getLogger('predictionPhase')

        # Copying raw data from Local path to project folder
        logger.info('Raw data copying to local path started')
        dataingestion.rawDataLocal(path, 'prediction')
        logger = WaferLogger.getLogger('predictionPhase')
        logger.info('Raw data copying to local path ended')

        # Transferring data to database
        logger.info('Data Transmission phase from Local to Database started')
        dataingestion.LoadToDB('prediction')
        logger = WaferLogger.getLogger('trainingPhase')
        logger.info('Data Transmission phase from Local to Database ended')

        logger.info('Data retrieval phase from database to Local started')
        # Retrieving data from database
        features = dataingestion.LoadFromDB('prediction')

        logger.info('Data preprocessing phase started')
        # Null value imputation
        if preProcessing.nullValueCheck(features, 'prediction'):
            logger = WaferLogger.getLogger('predictionPhase')
            logger.info('Imputing missing values using KNN imputer')
            features = preProcessing.KNNImputer(features, 'prediction')
            logger = WaferLogger.getLogger('predictionPhase')
            logger.info('Missing value imputation completed')

        # Removing features with null values
        logger.info('Removing features with zero variance')
        preProcessedData = pd.DataFrame(preProcessing.zerovarcol(features, 'prediction'))
        logger = WaferLogger.getLogger('predictionPhase')
        logger.info('Data preprocessing phase ended')

        logger.info('Data Prediction started')
        # Input data prediction
        predicted_data = tuner.predictData(preProcessedData)
        logger = WaferLogger.getLogger('predictionPhase')
        logger.info('Data Prediction completed')
        # Directory creation for saving output
        logger.info('Creating output directory')
        utils.dirCheck('output')

        # Saving prediction results
        logger.info('saving prediction results to output directory')
        predicted_data.to_csv('output/predicted_data.csv', index=False)
        return HttpResponse('done')
