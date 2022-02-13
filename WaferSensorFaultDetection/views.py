from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
from DataIngestion import dataIngestion
from DataPreProcessing import PreProcessing
from django.views.decorators.csrf import csrf_exempt
from MLAlgo import modeltuner
from Utils import Utils


@csrf_exempt
def train(request):
    if request.method == "POST":
        path = request.body.decode('ascii')
        dataingestion = dataIngestion.DataIngestion()
        dataingestion.rawDataLocal(path)
        dataingestion.LoadToDB('train')
        utils = Utils.utils()
        utils.removedir('Data/rawData')
        utils.removedir('Data/GoodData')
        dataframe = dataingestion.LoadFromDB()
        preProcessing = PreProcessing.PreProcessing()
        features = dataframe.drop('Good/Bad', axis=1)
        labels = dataframe['Good/Bad']
        if preProcessing.nullValueCheck(features):
            features = preProcessing.KNNImputer(features, 'train')
        else:
            features = dataframe

        preProcessedData = preProcessing.zerovarcol(features,'train')

        tuner = modeltuner.modelTuner()
        tuner.get_best_model(preProcessedData, labels)

        return HttpResponse('done')

def predict(request):
    if request.method == "POST":
        path = request.body.decode('ascii')
        dataingestion = dataIngestion.DataIngestion()
        dataingestion.rawDataLocal(path)
        dataingestion.LoadToDB()
        utils = Utils.utils()
        utils.removedir('Data/rawData')
        utils.removedir('Data/GoodData')
        dataframe = dataingestion.LoadFromDB()
        preProcessing = PreProcessing.PreProcessing()
        features = dataframe.drop('Good/Bad', axis=1)
        labels = dataframe['Good/Bad']
        if preProcessing.nullValueCheck(features):
            features = preProcessing.KNNImputer(features,'predict')
        else:
            features = dataframe
        tuner = modeltuner.modelTuner()
        tuner.findModels(features)
