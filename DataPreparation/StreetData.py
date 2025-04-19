import numpy as np 
import pandas as pd

data = pd.read_csv("Data/DataCollection/StopStreets.csv")

frame = pd.DataFrame()

frame[['Stop ID', 'Abreviation', 'Stop Name', 'Lat', 'Long', 'Routes']] = data[['Stop ID', 'FX Abbreviation', 'Stop Name', 'Latitude', 'Longitude', 'Routes']]


def splitter(row):
    row = row.strip()
    row = row.split(',')
    
    return row[:-1]
frame['Routes'] = frame['Routes'].apply(splitter)

frame = frame.to_json('StopInfo.json', orient='records', indent=2)

