import pandas as pd
from collections import defaultdict

# ---------- IMPORTS -----------
traffic = pd.read_csv("Data/DataCollection/StreetTraffic.csv")
# dd = defaultdict(list)
# dd = {}
# dd = traffic.to_dict('index', into=dd, index=False)
# print(dd)
# print(dd['Street'] == 'Adelaide')
# print(traffic['Street']=='Adelaide')

street = pd.read_csv("Data/DataCollection/StopStreets.csv")

newFrame = pd.DataFrame()

newFrame['Stop ID'] = street['Stop ID']
newFrame['Stop Code'] = street['FX Abbreviation']

tv = []
for entry in street['Stop Name'].values:
    f = traffic.loc[traffic['Street'] == entry.split()[0]]
    try:
        print(f['Traffic'].item())
        tv.append(f['Traffic'].item())
    except Exception as e:
        tv.append(0)
    
    
newFrame['Traffic'] = tv
print(newFrame)

newFrame.to_csv("StopTraffic.csv", index=False)