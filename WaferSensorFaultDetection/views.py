from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
from DataIngestion import dataIngestion
from DataPreProcessing import PreProcessing
from django.views.decorators.csrf import csrf_exempt
from MLAlgo import modeltuner


@csrf_exempt
def train(request):
    if request.method == "POST":
        path = request.body.decode('ascii')
        dataingestion = dataIngestion.DataIngestion()
        dataingestion.rawDataLocal(path)
        dataingestion.LoadToDB()
        dataframe = dataingestion.LoadFromDB()
        preProcessing = PreProcessing.PreProcessing()
        if preProcessing.nullValueCheck(dataframe):
            imputed_data = preProcessing.KNNImputer(dataframe)
        else:
            imputed_data = dataframe


        features = imputed_data.drop('Good/Bad', axis=1)
        labels = imputed_data['Good/Bad']
        preProcessedData = preProcessing.zerovarcol(features)

        tuner=modeltuner.modelTuner()
        tuner.get_best_model(preProcessedData,labels)



        return HttpResponse('done')



