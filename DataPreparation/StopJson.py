import json

with open("Data/CondensedDataFiles/StopTraffic.csv", "r") as file :
    line = file.readline()
    while(line != None):
        currentline = line.split(",")
        with open("Traffic_mapping.json", "a") as newFile:
            newFile.write(f'"{currentline[1]}":{currentline[2].strip()},\n')

        line = file.readline()