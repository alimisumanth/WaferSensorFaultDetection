from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
from DataIngestion import dataIngestion
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def test(request):
    if request.method == "POST":
        path = request.body.decode('ascii')
        dataingestion=dataIngestion.DataIngestion()
        dataingestion.rawDataLocal(path)
        dataingestion.LoadToDB()
        return HttpResponse('done')



