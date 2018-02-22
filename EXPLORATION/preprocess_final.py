#
# Needs a Python 3 session !!!
#
#!pip install matplotlib, sklearn
'''
Basic Data Summarizer 
By: Adam Li
v1.0 - 02/21/18

'''
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

class DataPreprocessor(object):
  def __init__(self, dataframe, VERBOSE=0):
    self.df = dataframe
    self.intervals = {}
    
    # convert all colnames to lowercase
    self.df.columns = map(str.lower, self.df.columns)
    self.colnames = self.df.columns.values
    self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
    
    
    # drop the unused index column
    try:
      self.df = self.df.drop("unnamed: 0", axis=1)
    except ValueError:
      print('cant drop index col')

    self.VERBOSE = VERBOSE
    
    # merge avg speed
    data_shared = "/home/cdsw/data_shared/"
    avgSpeed = pd.read_csv(os.path.join(data_shared, 'avgSpeed.csv'))
    self.df = pd.merge(self.df, avgSpeed, on=['tourorigin', 'tourdestination'])
    
#    print(self.df)
    # mapping locations -> num, creating timeofday, timestamp, distances
    self.maplocationstonum()
    self.add_date_time()
    self.set_interval([0,6,12,18,24])
    self.add_distances()
    self.mod_headings()
    self.add_timetraveled()
#    display(preprocess.df)

    # create lambda function for preprocessing a distribution
    self._distributions = lambda X: {
      'unscaled': X,
      'max': np.divide(X, np.max(X)),
      'standard':
          StandardScaler().fit_transform(X),
      'minmax':
          MinMaxScaler().fit_transform(X),
      'maxabs':
          MaxAbsScaler().fit_transform(X),
      'robust':
          RobustScaler(quantile_range=(25, 75)).fit_transform(X),
      'uniquant':
          QuantileTransformer(output_distribution='uniform').fit_transform(X),
      'gaussquant':
          QuantileTransformer(output_distribution='normal').fit_transform(X),
      'l2':
          Normalizer().fit_transform(X)
    }
  def normalizecols(self, colname, scaler='unscaled'):
    
    ''' minmax scale across all datapoints '''
    buff_df = self.df
    buff_df[colname] = self._distributions(buff_df[colname].values.reshape(-1,1))[scaler]
    self.df[colname] = buff_df[colname]
    
  def maplocationstonum(self):
    '''
    Map all locations to numerical number instead
    '''
    locs = ['tourorigin', 'tourdestination']

    unique_origins = {}
    unique_destinations = {}
    
    self.update_dictionary(unique_origins, self.get_unique_location(locs[0]), range(len(self.get_unique_location(locs[0]))))
    self.update_dictionary(unique_destinations, self.get_unique_location(locs[1]), range(len(self.get_unique_location(locs[1]))))
    
    self.update_dataframe_location(unique_origins, unique_destinations)
  
  def get_unique_location(self, column):
    '''
    Gets the unique locations from the input column
    '''
    return self.df[column].unique()
    
  def update_dictionary(self, d, keys, values):
    '''
    Updates the dictionary with the key value pairs
    '''
    return d.update(zip(keys, values))
  
  def update_dataframe_location(self, origin, destination):
    '''
    Updates the dataframe from location to numbers
    '''
    self.df["tourorigin"] = self.df["tourorigin"].map(origin)
    self.df["tourdestination"] = self.df["tourdestination"].map(destination)
    
  def split_date(self):
    '''
    Returns the year, month and the day
    '''
    return self.df["timestamp"].dt.year, self.df["timestamp"].dt.month, self.df["timestamp"].dt.day
  
  def split_time(self):
    '''
    Returns the time in hh:mm:ss
    '''
    return self.df["timestamp"].dt.hour, self.df["timestamp"].dt.minute, self.df["timestamp"].dt.second
  
  def get_day_of_week(self):
    """
    Returns the day of the week
    """
    return self.df['timestamp'].dt.dayofweek
    
  def set_interval(self, times):
      '''
      Generates the intervals
      '''
      for index in range(len(times)-1):
        self.intervals[str(times[index])+'-'+str(times[index+1])] = index
        
      df_interval = []
      idx = 0
      for hour in self.time[0]:  # loop through hours
        NOT_SET = True
        for index in range(len(times)-1): # loop through all intervals we created
          if hour >= times[index] and hour < times[index+1]:
            NOT_SET=False
            df_interval.append(self.intervals[str(times[index])+'-'+str(times[index+1])])
            break
        '''
        SUPER SHITTY HACK, BUT THIS WILL ADD NANS 
        TO MAKE SURE THE LIST CREATED IS THE SAME
        
        -> CREATES NANS AT TIME INTERVALS FOR MISSING TIMESTAMP ROWS
        '''
        if NOT_SET:
          df_interval.append(np.nan)

      self.df['interval'] = df_interval
      
  def add_date_time(self):
    self.time = self.split_time()
    date = self.split_date()
    day_of_week = self.df['timestamp'].dt.dayofweek
    
    final_df = pd.DataFrame({
      'year': date[0],
      'month': date[1],
      'day' : date[2],
      'day_of_week' : day_of_week,
      'hours' : self.time[0],
      'minutes' : self.time[1],
      'seconds' : self.time[2]
    })
    final_df = final_df[['hours', 'minutes', 'seconds','day_of_week', 'day', 'month', 'year']]
    self.df = pd.concat([self.df, final_df], axis=1)
    
  def add_distances(self):
    self.df["max_remdistance"]=np.nan
    for ride_id in self.df["ride_id"].unique():
      self.df.loc[self.df["ride_id"] == ride_id, "max_remdistance"] = self.df.loc[self.df["ride_id"] == ride_id, "remdistance"].max()
    self.df["drivendistance"]=self.df["max_remdistance"]-self.df["remdistance"]

  def mod_headings(self):
    heading = {'N':0,'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE': 135, 'SSE':157.5, 'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
    self.df = self.df.replace({"heading":heading})
    
  def getuniquerides(self):
    return self.df.ride_id.unique()

  def add_timetraveled(self):
    '''
    Adds travbeld time in seconds
    '''
    uniquerides = self.getuniquerides()
    
    dflen = len(self.df)
    self.df['traveledtime'] = pd.Series(np.nan*np.ones(dflen), index=self.df.index)
    
    # go through each ride and create a new list of dict dataframes
    for idx,ride in enumerate(uniquerides):
      ride_df = self.df.loc[self.df.ride_id == ride]
      initialtime_secs = ride_df['timestamp'].iloc[0].timestamp()
      
      # convert all times to seconds
      ride_secs = np.array([x.timestamp() if x is not pd.NaT else np.nan for x in ride_df['timestamp']])
  
      # compute traveledtime
      traveledtime = list(np.ceil(np.subtract(ride_secs,initialtime_secs)))
      self.df.loc[self.df.ride_id == ride,'traveledtime'] = traveledtime

class MungeDataset(object):
  def __init__(self, dataframe):
    self.df = dataframe
    self.munged_df = []
    
    self.mungedX_df = []
    self.mungedY_df = []
    
#    self.getuniquerides_df()
#    self.computelineartrend()
  def getuniquerides(self):
    return self.df.ride_id.unique()
  
  def getuniqueorigins(self):
    return self.df.tourorigin.unique()
  
  def getuniquedest(self):
    return self.df.tourdestination.unique()
  
  def getuniquerides_df(self):
    '''
    Possibly create a data struct for all unqiue rides
    '''
    uniquerides = self.getuniquerides()
    
    # go through each ride and create a new list of dict dataframes
    for ride in uniquerides:
      ride_df = self.df[self.df['ride_id'] == ride]
      otherrides = self.df[self.df['ride_id'] != ride]
      
      self.munged_df.append({'ride': ride_df, 'other': otherrides})
#      self.mungedX_df.append({
#                            'ride':ride_df.drop('actual_eta', axis=1),
#                            'other': otherrides.drop('actual_eta', axis=1)
#                            })
#      self.mungedY_df.append({
#                            'ride':ride_df['actual_eta'],
#                            'other': otherrides['actual_eta']
#                            })
  def computelineartrend(self):
    from sklearn import linear_model
    '''
    Given a ride dataframe, compute the linear trend
    of timepoints vs actual_eta
    '''
    uniqueorigins = self.getuniqueorigins()
    
    self.linmods = dict()
    # create vector of traveled time and eta
    X = dict()
    y = dict()
    
    allrides = dict()
    # loop through all unique origins
    for ride in uniqueorigins:
      thisride_df = self.df[self.df['tourorigin'] == ride]
      thisride_dests = thisride_df['tourdestination'].unique()
      
      # loop through possible destinations
      for dest in thisride_dests:
        # the keyvalue
        keyval = str(ride) + '-' + str(dest)
        
        uniqueride_df = thisride_df[thisride_df['tourdestination'] == dest]
        if keyval not in allrides.keys():
          allrides[keyval] = thisride_df[thisride_df['tourdestination'] == dest]
          X[keyval] = [uniqueride_df['traveledtime'].astype(float).ravel()]
          y[keyval] = [uniqueride_df['actual_eta'].astype(float).ravel()]
        else:
          print('adding more data')
          X[keyval].append(uniqueride_df['traveledtime'].astype(float).ravel())
          y[keyval].append(uniqueride_df['actual_eta'].astype(float).ravel())
          
    self.X = X
    self.y = y
    self.allrides = allrides
    
    '''
        NEED TO DELETE WHEN THERE IS EXTRA DATA
    '''
    lr = linear_model.LinearRegression()
    
    for uniqueride in X.keys():
      X = self.X[uniqueride][0]
      y = self.y[uniqueride][0]
      lr.fit(X[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
      self.linmods[uniqueride] = lr
      
  def filltraveledtime(self):
    ''' Call for the testdf'''
    uniquerides = self.getuniquerides()
    
    # go through each ride and create a new list of dict dataframes
    for ride in uniquerides:
      ride_df = self.df[self.df['ride_id'] == ride]
      
      origin = ride_df['tourorigin'].sample(1)
      dest = ride_df['tourdestination'].sample(1)
      
      # get the last timestamp we have available
      lastind = ride_df['traveledtime'].last_valid_index()
      last_traveledtime = ride_df['traveledtime'].ix[lastind]
      
      # get the average time stamp differences
      timediff1 = np.diff(ride_df['traveledtime'])
      timediff1 = np.nanmean(timediff1)
      
      # the keyvalue
      keyval = str(origin) + '-' + str(dest)
      
      # fit with linear curve
      num_nans = len(ride_df.loc[ride_df['traveledtime'].isnull(), 'traveledtime'])
      fill_traveltime =  last_traveledtime + np.multiply(timediff1,np.arange(num_nans).astype(int)+1)
      
      # modify the existing dataframe
      self.df.loc[(self.df.ride_id == ride) & 
                  (self.df['traveledtime'].isnull()),'traveledtime'] = fill_traveltime
#      self.df.loc[(self.df.ride_id == ride), 'traveledtime'] = ride_df['traveledtime'].interpolate()
#      print(self.df[self.df.ride_id==ride])
  def fill_dayofweek(self):
    from sklearn.preprocessing import Imputer
    imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)

    ''' Call for the testdf'''
    uniquerides = self.getuniquerides()
    colname = 'day_of_week'
    
    # go through each ride and create a new list of dict dataframes
    for ride in uniquerides:
      ride_df = self.df[self.df['ride_id'] == ride]
      # get the last timestamp we have available
      lastind = ride_df[colname].last_valid_index()
      last_val = ride_df[colname].ix[lastind]
      
      colnan_df = self.df.loc[(self.df['ride_id'] == ride) & 
                          (self.df[colname].isnull()), colname]
      self.df.loc[(self.df['ride_id'] == ride) &
             (self.df[colname].isnull()), colname] = imputer.fit_transform(colnan_df.values.reshape(-1,1))
      
  def fill_interval(self):
    from sklearn.preprocessing import Imputer
    imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)

    ''' Call for the testdf'''
    uniquerides = self.getuniquerides()
    colname = 'interval'
    
    # go through each ride and create a new list of dict dataframes
    for ride in uniquerides:
      ride_df = self.df[self.df['ride_id'] == ride]
      # get the last timestamp we have available
      lastind = ride_df[colname].last_valid_index()
      last_val = ride_df[colname].ix[lastind]
      
#      ride_df = ride_df.fillna(ride_df[colname].value_counts().index[0])
      ride_df = ride_df.fillna(method='ffill')
      self.df[self.df['ride_id'] == ride] = ride_df
      
if __name__ == '__main__':
  data_shared = "/home/cdsw/data_shared/"
  TEST_FILE = os.path.join(data_shared,'df_test_given_data.csv')
  TRAIN_FILE = os.path.join(data_shared, 'df_train.csv')
  
  # LOAD IN THE TRAINING DATA
  train_df = pd.read_csv(TRAIN_FILE)
  test_df = pd.read_csv(TEST_FILE)
  
  # get the preprocessed dataframe
  preprocess = DataPreprocessor(train_df)
  dataproctest = DataPreprocessor(test_df)
  test_df = dataproctest.df
  train_df = preprocess.df
#  print(train_df.head())
#  print(test_df.head(20))
  
  munger = MungeDataset(test_df)
  munger.getuniquerides_df()
#  print(munger.mungedX_df[0]['ride'].head())
#  print(munger.mungedY_df[0]['ride'].head())
#  munger.computelineartrend()
#  print(munger.linmods)
  
  munger.filltraveledtime()
#  munger.fill_dayofweek()
  munger.fill_interval()
  print(munger.df.head(20))