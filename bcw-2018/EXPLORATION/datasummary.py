#
# Needs a Python 3 session !!!
#
#!pip install matplotlib, sklearn
'''
Basic Data Summarizer 
By: Adam Li
v1.0 - 02/21/18

'''

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

import seaborn as sns
import matplotlib.pyplot as plt

# import modules related

class DataSummary(object):
  def __init__(self, dataframe, VERBOSE=0):
    self.df = dataframe
    # convert all colnames to lowercase
    self.df.columns = map(str.lower, self.df.columns)
    
    self.colnames = self.df.columns.values
    
    # create lambda function for preprocessing a distribution
    self._distributions = lambda X: {
      'unscaled': X,
      'standard':
          StandardScaler().fit_transform(X),
      'minmax':
          MinMaxScaler().fit_transform(X),
      'maxabs':
          MaxAbsScaler().fit_transform(X),
      'robust':
          RobustScaler(quantile_range=(25, 75)).fit_transform(X),
      'uniquant':
          QuantileTransformer(output_distribution='uniform')
          .fit_transform(X),
      'gaussquant':
          QuantileTransformer(output_distribution='normal')
          .fit_transform(X),
      'l2':
          Normalizer().fit_transform(X)
    }

    
    self.VERBOSE = VERBOSE 
  def getuniquerides(self):
    if self.VERBOSE:
      print("All unique rides are: ", datasumm.getuniquerides())
      print("Total number of unique rides are: ", len(datasumm.getuniquerides()))
    return self.df.ride_id.unique()
  
  def getuniqueorigins(self):
    if self.VERBOSE:
      print("All unique rides are: ", datasumm.getuniqueorigins())
      print("Total number of unique rides are: ", len(datasumm.getuniqueorigins()))
    return self.df.tourorigin.unique()
  
  def getuniquedest(self):
    if self.VERBOSE:
      print("All unique rides are: ", datasumm.getuniquedest())
      print("Total number of unique rides are: ", len(datasumm.getuniquedest()))
    return self.df.tourdestination.unique()
  
  def getuniquerides(self):
    uniqueorigins = self.getuniqueorigins()
    uniquedest = self.getuniquedest()
    
    allrides = dict()
    # loop through all unique origins
    for ride in uniqueorigins:
      thisride_df = self.df[self.df['tourorigin'] == ride]
      thisride_dests = thisride_df['tourdestination'].unique()
      
      # loop through possible destinations
      for dest in thisride_dests:
        keyval = ride + '-' + dest
        
        if keyval not in allrides.keys():
          allrides[keyval] = len(thisride_df[thisride_df['tourdestination'] == dest])
    self.allrides = allrides
    
  def getridefrom_atob(self, a, b):
    thisride_df = self.df[self.df['tourorigin'] == a]
    thisride_df = thisride_df[thisride_df['tourdestination']==b]
    
    return thisride_df
  
  def normalizecols(self, colname, scaler='unscaled'):
    scaler = MinMaxScaler()
    
    ''' minmax scale across all datapoints '''
    buff_df = self.df
    buff_df[colname] = self._distributions(buff_df[colname].values)[scaler]
    
    ''' another idea is to minmax scale across all starting origins '''
#    vartogroup = 'tourorigin'
#    uniqueorigins = self.getuniqueorigins()
#    for origin in uniqueorigins:
#      buff_df = self.df[self.df[vartogroup]==origin]
#      buff_df[colname] = self._distributions(buff_df[colname].values)[scaler]
    
    ''' another idea is to minmax scale across all end points '''
#    uniquedest = self.getuniquedest()
#    vartogroup = 'tourdestination'
#    for dest in uniquedest:
#      buff_df = self.df[self.df[vartogroup]==dest]
#      buff_df[colname] = self._distributions(buff_df[colname].values)[scaler]
    
    ''' another idea is to minmax scale across all the unique rides '''
#    uniquerides = self.allrides
#    vartogroup1 = 'tourorigin'
#    vartogroup2 = 'tourdestination'
#    for ride in uniquerides:
#      origin, dest = ride.split('-')
#      buff_df = self.df[self.df[vartogroup1]==origin]
#      buff_df = buff_df[buff_df[vartogroup2]==dest]
#      buff_df[colname] = self._distributions(buff_df[colname].values)[scaler]
    return buff_df
    
  def summarize_plots(self, grouper='ride_id', suptitle=None):
    plots = ['tourorigin', 'tourdestination', 'remdistance',
            'alltours', 'elevation', 'heading', 'lat', 'lng']
    
    fig, axes = plt.subplots(figsize=(8,20), nrows=5, ncols=2)
    sns.set(font_scale=1.)
    fig.subplots_adjust(top=4.85)
    
    '''           TOUR ORIGIN                '''
    colname='tourorigin'
    data_dist = self.df.groupby(grouper).apply(lambda df: df.sample(1))
    ax = sns.countplot(x=colname, data=data_dist, ax=axes[0,0])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_xlabel(colname)
    ax.set_ylabel('Number of rides')
    
    if suptitle:
      ax.set_title(suptitle)
    
    '''           TOUR DESTINATION                '''
    colname='tourdestination'
    data_dist = self.df.groupby(grouper).apply(lambda df: df.sample(1))
    ax = sns.countplot(x=colname, data=data_dist, ax=axes[0,1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_xlabel(colname)
    ax.set_ylabel('Number of rides')

    '''           REMAINING DISTANCE                '''
    colname='remdistance'
    data_dist = self.df.groupby(grouper).apply(lambda df: df.sample(1))
    ax = sns.distplot(data_dist[colname], ax=axes[1,0], kde=False)
    ax.set_xlabel(colname)
    ax.set_ylabel('Number of datapoints')

    '''           ALL TOURS                '''
#    self.getuniquerides()
    data_dist = pd.Series(self.allrides).to_frame('counts')
    ax = sns.barplot(x=data_dist.index, y='counts', data=data_dist, ax=axes[1,1])
    ax.set_ylabel('Number of datapoints')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')

    
    '''           ELEVATION                '''
    colname='elevation'
    data_dist = self.df
    ax = sns.distplot(data_dist[colname], ax=axes[2,0], kde=False)
    ax.set_xlabel(colname)
    ax.set_ylabel('Number of datapoints')
    
    '''           HEADING                '''
    colname='heading'
    data_dist = self.df
    ax = sns.countplot(data_dist[colname], ax=axes[2,1])
    ax.set_xlabel(colname)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_ylabel('Number of datapoints')
    
    '''           LAT                '''
    colname='lat'
    data_dist = self.df
    ax = sns.distplot(data_dist[colname], ax=axes[3,0], kde=False)
    ax.set_xlabel(colname)
    ax.set_ylabel('Number of datapoints')
    
    '''           LNG                '''
    colname='lng'
    data_dist = self.df
    ax = sns.distplot(data_dist[colname], ax=axes[3,1], kde=False)
    ax.set_xlabel(colname)
    ax.set_ylabel('Number of datapoints')
    
    if suptitle:
      plt.suptitle(suptitle)
    fig.tight_layout()

if __name__ == '__main__':    
  TEST_FILE = os.path.join('./data_shared/df_test_given_data.csv')
  TRAIN_FILE = os.path.join('./data_shared/df_train.csv')

  train_df = pd.read_csv(TRAIN_FILE)
  test_df = pd.read_csv(TEST_FILE)

  print(train_df.columns.values)
  print(train_df.head())
  # SUMMARIZE TRAINING DATA #
  datasumm = DataSummary(train_df)
  uniquerides = datasumm.getuniquerides()
  uniqueorigins = datasumm.getuniqueorigins()
  uniquedest = datasumm.getuniquedest()
  datasumm.getuniquerides()
  datasumm.summarize_plots(suptitle='TRAIN DATASET')
  
  '''
  To get the unique rides
  '''
  uniqrides_df = datasumm.df['uni']
  
  print('\n\n')
  
  # SUMMARIZE TESTING DATA #
  datasumm = DataSummary(test_df)
  uniquerides = datasumm.getuniquerides()
  uniqueorigins = datasumm.getuniqueorigins()
  uniquedest = datasumm.getuniquedest()
  datasumm.getuniquerides()
  
  
  datasumm.summarize_plots(suptitle='TEST DATASET')
#  print(datasumm.df['heading'].unique())
  #print(datasumm.df[datasumm.df['tourorigin']=='Zagreb'])
  #print(train_df.shape)
  #print(test_df.shape)%%!