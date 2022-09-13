from io import DEFAULT_BUFFER_SIZE
from django.shortcuts import render
from django.http import JsonResponse
import json
import pandas as pd
import joblib

model = joblib.load('pipelines/best_model.pkl')


def scoreJson(request):
    data = json.loads(request.body)
    for k in data.keys():
        data[k] = float(data[k])
    
    if data['FGA'] == 0:
        data['FG%'] = 0
    else: data['FG%'] = data['FGM']/data['FGA']*100,


    if data['3PA'] == 0:
        data['3P%'] = 0
    else: data['3P%'] = data['3PMade']/data['3PA']*100,


    if data['FTA'] == 0:
        data['FT%'] = 0
    else: data['FT%'] = data['FTM']/data['FTA']*100,


    dfready = {
        'GP' : data['GP'],
        'MIN' :data['MIN'],
        'PTS' :data['PTS'],
        'FGM':data['FGM'],
        'FGA':data['FGA'],
        'FG%':data['FG%'],
        '3P Made':data['3PMade'],
        '3PA':data['3PA'],
        '3P%':data['3P%'],
        'FTM':data['FTM'],
        'FTA':data['FTA'],
        'FT%':data['FT%'],
        'OREB':data['OREB'],
        'DREB':data['DREB'],
        'REB':data['REB'],
        'AST':data['AST'],
        'STL':data['STL'],
        'BLK':data['BLK'],
        'TOV':data['TOV']
    }
    
    print(dfready)
    df = pd.DataFrame(dfready, index = [0])

    print(df)
    
    proba = model.predict_proba(df.values)[0][1]
    pred = model.predict(df.values)[0]

    print(proba)
    print(pred)

    return JsonResponse({"score":proba,'prediction':pred})