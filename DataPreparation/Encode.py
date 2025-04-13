import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pickle import dump

data = pd.read_csv("Data/OldFiles/102_with_weather.csv")

weather = data['Weather'].fillna(method='ffill')
encoder = LabelEncoder()

encode_weather = encoder.fit(weather)

np.save('weather.npy', encode_weather.classes_)

dump(encoder, open('weather.pkl', 'wb'))
print(weather.unique())


