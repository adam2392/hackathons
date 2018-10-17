import pandas as pd
import numpy as np

# read in data
data1 = pd.read_csv("xaa.csv")
data2 = pd.read_csv("xab.csv")
data3 = pd.read_csv("xac.csv")

# rename columns accordingly
data1.columns = ["row_id", "player_id", "player_name", "time", "elapsed_time", "lat", "lon", "speed", "heart_rate"]
data2.columns = ["row_id", "player_id", "player_name", "time", "elapsed_time", "lat", "lon", "speed", "heart_rate"]
data3.columns = ["row_id", "player_id", "player_name", "time", "elapsed_time", "lat", "lon", "speed", "heart_rate"]

newdata1 = data1.set_index('row_id')
newdata2 = data2.set_index('row_id')
newdata3 = data3.set_index('row_id')

# merge DataFrame object
frames = [newdata1, newdata2, newdata3]
data = pd.concat(frames)

# show distribution of speed of players
# fig, axs = plt.subplots(len(playerIDs),1)
# for i in range(0,len(playerIDs)):
for i in range(0,1):
    # get the player's data rows
    playerid = playerIDs[i]
    temp = data.loc[data['player_id'] == playerid]
    
    # determine the start and end rows of first half
    firststart = temp.loc[(temp['time'] > '19:31.3')][0:1]
    firstend = temp.loc[(temp['time'] < '20:17:3')]
    firstend = firstend.reindex(index=firstend.index[::-1])[0:1]
    
    # determine the start and end rows of second half
    secondstart = temp.loc[(temp['time'] > '20:30:3')][0:1]
    secondend = temp.loc[(temp['time'] < '21:20:26')]
    secondend = secondend.reindex(index=secondend.index[::-1])[0:1]
    
    
    firststarttime = firststart['time']
    firstendtime = firstend['time']
    secondstarttime = secondstart['time']
    secondendtime = secondend['time']
    
    firststart = firststart['row_id']
    firstend = firstend['row_id']
    secondstart = secondstart['row_id']
    secondend = secondend['row_id']
    
    firsthalf = temp.iloc[int(firststart):int(firstend)]
    secondhalf = temp.iloc[int(secondstart):int(secondend)]
    
#     # produce json for firsthalf
    firsthalfresponses = []
    for j in range(0, len(firsthalf)):
        # set name
        if j is 0:
            name = "Gimme"
        elif j is len(firsthalf) - 1:
            name = "Proletariat"
        else:
            name = "Along route"

        lat = firsthalf.iloc[j]['lat']
        lon = firsthalf.iloc[j]['lon']
        time = (firsthalf.iloc[j]['time']) #- initialtime


        response = {"type": "Feature",
                    "properties": {
                                "latitude": lat,
                                "longitude": lon,
                                "time": time,
                                "id": "route1",
                                "name": name,
                                },
                    "geometry": {
                                "type": "Point",
                                "coordinates": [lon, lat]
                    }
        }
        firsthalfresponses.append(response)

    firsthalfjson = {"userID": playerID,
                    "type": "FeatureCollection",
                    "crs": { 
                            "type": "name", 
                            "properties": { 
                                            "name": "urn:ogc:def:crs:OGC:1.3:CRS84" 
                                        } 
                            },
                    "features": firsthalfresponses
                    }
    
    # produce json for secondhalf
    secondhalfresponses = []
    for j in range(0, len(secondhalf)):
        # set name
        if j is 0:
            name = "Gimme"
        elif j is len(secondhalf) - 1:
            name = "Proletariat"
        else:
            name = "Along route"

        lat = secondhalf.iloc[j]['lat']
        lon = secondhalf.iloc[j]['lon']
        time = (secondhalf.iloc[j]['time']) #- initialtime


        response = {"type": "Feature",
                    "properties": {
                                "latitude": lat,
                                "longitude": lon,
                                "time": time,
                                "id": "route1",
                                "name": name,
                                },
                    "geometry": {
                                "type": "Point",
                                "coordinates": [lon, lat]
                    }
        }
        secondhalfresponses.append(response)

    secondhalfjson = {"userID": playerID,
                    "type": "FeatureCollection",
                    "crs": { 
                            "type": "name", 
                            "properties": { 
                                            "name": "urn:ogc:def:crs:OGC:1.3:CRS84" 
                                        } 
                            },
                    "features": secondhalfresponses
                    }