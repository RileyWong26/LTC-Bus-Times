import pandas as pd


tv = pd.read_csv('Data/Traffic_Volumes.csv')

tv = tv.sort_values(by=['StreetName'])
print(tv)
uniqueStreets = tv['StreetName'].unique()
print(uniqueStreets)
newFrame = pd.DataFrame()
# print(newFrame)

temp = tv.loc[tv['StreetName'] == 'Adelaide St N']
print(temp)
print(len(temp))
print(len(list(temp['VolumeCount'])))

volumeCount = []

for street in uniqueStreets:
    number = tv.loc[tv['StreetName'] == street]
    
    listVolumceCount = list(number['VolumeCount'])
    averageVolumeCount = round(sum(listVolumceCount)/len(number),0)
    # print(averageVolumeCount)
    volumeCount.append(averageVolumeCount)

newFrame['StreetName'] = uniqueStreets
newFrame['VolumeCount'] = volumeCount
newFrame.to_csv('Data/CondensedTV.csv', index=False)