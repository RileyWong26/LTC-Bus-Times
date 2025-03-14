import pandas as pd

# CSV with stop ids--------------
stops = pd.read_csv("Data/combined4_102.csv")

# CSV with stop ids and the associated street/location--------------
streets = pd.read_csv("Data/StopStreets.csv")

# CSV with Traffic flow--------------
tf = pd.read_csv("Data/CondensedTV.csv")


# Add a new column where the stop id matches the street fx -------------------------
# street_name = []
# for stop in stops["stop_id"]:
#     stop_street = streets[streets['FX Abbreviation'] == stop]['Stop Name']
#     first_name = stop_street.to_string(index=False).split()[0]
#     street_name.append(first_name)

# stops['Street'] = street_name
# stops.to_csv("Test.csv", index=False)

# CSV with the associated street name -------------------
stops = pd.read_csv("Data/StopsWithStreet.csv")

# Unique first name of street -------------
# unique =[]
# for uni in tf['StreetName'].unique():
#     unique.append(uni.split()[0])

# Multiple streets with same first name, so take the one with most traffic
condensedFrame = pd.DataFrame()
condensedFrame['Street'] = ''
condensedFrame['Traffic'] = ''

for row in tf.itertuples(index=False):
    streetname = row.StreetName.split()[0]
    if streetname not in condensedFrame["Street"].unique():
        temp = [streetname, row.VolumeCount]
        condensedFrame.loc[len(condensedFrame)] = temp
    else:
        if float(condensedFrame[condensedFrame['Street'] == streetname]['Traffic'].to_string(index=False)) < row.VolumeCount:
            # condensedFrame[condensedFrame['Street'] == streetname]['Traffic'] = row.VolumeCount
            condensedFrame.loc[streetname, 'Traffic'] = row.VolumeCount


traffic = []
for row in stops.itertuples(index=False):
    if row.Street in condensedFrame['Street'].unique():
        traffic.append(condensedFrame[condensedFrame['Street'] == row.Street]['Traffic'].to_string(index=False))
    else:
        traffic.append(0)

stops['Traffic'] = traffic
stops.to_csv("StopsTraffic.csv", index=False)



