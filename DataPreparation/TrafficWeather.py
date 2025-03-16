import pandas as pd

# --------- Import the csv with the weather, the csv with the traffic, -----

weather_data = pd.read_csv('Data/CondensedDataFiles/102_updated.csv')
traffic_data = pd.read_csv('Data/CondensedDataFiles/StopsTraffic.csv')

# ----------- Compare the vehicle id, delay, scheduled time, day, and day of year to see if they are the same stop ------------

# def sameStop (id1, delay1, time1, day1, year1, id2, delay2, time2, day2, year2):
#     return (id1 == id2) and (delay1 ==delay2) and (time1==time2) and (day1==day2) and (year1==year2)
    

# ----------- ARRAY TO HOLD THE TRAFFIC AT THE STOP -------------------
weather_data['Traffic'] = None
# weather_data.loc[0, "Traffic"] = 1
# print(weather_data)


# ----------- LOOP THROUGH THE DATA -----------------

for entry in weather_data.itertuples():
    weather_data.loc[entry.Index, "Traffic"] = traffic_data.get(\
        (traffic_data['delay'] == entry.delay) &\
        (traffic_data['vehicle_id'] == entry.vehicle_id) &\
        (traffic_data['scheduled_time'] == entry.scheduled_time) &\
        (traffic_data['day'] == entry.day) &\
        (traffic_data['day_of_year'] == entry.day_of_year)
        )['Traffic'].to_string(index=False)

    # weather_data.loc[entry.Index, "Traffic"] = traffic_data.get(\
    #     (traffic_data['delay'] == entry.delay) &\
    #     (traffic_data['vehicle_id'] == entry.vehicle_id) &\
    #     (traffic_data['scheduled_time'] == entry.scheduled_time) &\
    #     (traffic_data['day'] == entry.day) &\
    #     (traffic_data['day_of_year'] == entry.day_of_year)).iloc[0]['Traffic']

# ------------ ADD TRAFFIC TO WEATHER DATA ------------------ 
# weather_data['Traffic'] = traffic

# ------------------- TO CSV ---------------------
weather_data.to_csv("test.csv", index=False)