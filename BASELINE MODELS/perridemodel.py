#
# Needs a Python 3 session !!!
#
#!pip install matplotlib, sklearn
'''
Basic Data Summarizer 
By: Adam Li
v1.0 - 02/21/18

'''
#!pip install sklearn
# !pip3 install xmlrpc
import xmlrpc.client

import os
import numpy as np
import pandas as pd

# pretty charting
import seaborn as sns
sns.set_palette('muted')
sns.set_style('darkgrid')
import warnings

import sys
sys.path.insert(0,'/home/cdsw/EXPLORATION')
sys.path.insert(0,'/home/cdsw/')
from preprocess_final import DataPreprocessor, MungeDataset

# Regression
from sklearn import cross_validation
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

class RideModel:
    def __init__(self, target_column):
        """
        The user provides a train and test filename which are loaded as X_train and X_test as pandas dataframe.
        X_train is used to train the model and optimize the model parameters with a grid search on hyper parameters.
        The trained model is used to create the y_pred values using the X_test data frame.
        The y_pred values are returned back as numpy array.
        :param filename_train_feature: str, filename for train features.
        :param filename_test_feature: str, filename for test features.
        :param target_column: str, th        :param target_column: str, the column storing the target values.
e column storing the target values.
        """
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None
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
        self.columns_test = ['tourorigin',
                             'tourdestination',
                              'lat',
                             'lng', 
#                              'accuracy',
                              'drivendistance', 
                              'traveledtime',
                              'remdistance', 
                              'elevation',
                              'speed',
#                              'heading',
#                      'day_of_week','interval',
#                      'max_remdistance', 
#                  'timestamp',
#                  'actual_eta',
#                      'hours', 'minutes', 'seconds', 
#                       'day', 'month', 'year'
                     ]

    def normalize(self):
      ''' NORMALIZATION '''
      colnames = ['lat', 
                  'lng', 
                  'elevation',
                  'remdistance', 
                  'drivendistance',
                  'speed',
#                  'heading'
                 ]
#      colnames = []
      for colname in colnames:
      #        'unscaled': X,
      #      'standard':
      #      'minmax':
      #      'maxabs':
      #      'robust':
      #      'uniquant':
      #      'gaussquant':
      #      'l2':
        df = self.X_train
        preprocesstype = 'max'
        df = self.normalizecols(df, colname, scaler=preprocesstype)
        self.X_train = self.X_train
    def normalizetest(self):
      ''' NORMALIZATION '''
      colnames = ['lat', 
                  'lng', 
                  'elevation',
                  'remdistance', 
                  'drivendistance',
                  'speed',
#                  'heading'
                 ]
      for colname in colnames:
      #        'unscaled': X,
      #      'standard':
      #      'minmax':
      #      'maxabs':
      #      'robust':
      #      'uniquant':
      #      'gaussquant':
      #      'l2':
        df = self.X_test
        preprocesstype = 'max'
        df = self.normalizecols(df, colname, scaler=preprocesstype)
        self.X_test = self.X_test
      
    def normalizecols(self, df, colname, scaler='unscaled'):
      ''' minmax scale across all datapoints '''
      buff_df = df
      buff_df[colname] = self._distributions(buff_df[colname].values.reshape(-1,1))[scaler]
#      self.df[colname] = buff_df[colname]
      return buff_df
    def loaddbdata(self, TRAIN_FILE):
      # LOAD IN THE TRAINING DATA
      train_df = pd.read_csv(TRAIN_FILE)

      # get the preprocessed dataframe
      dataproctrain = DataPreprocessor(train_df)
      dataproctrain.df = dataproctrain.df.dropna(axis=0, how='any')
#      dataproctrain = self.normalize(dataproctrain)
#      dataproctrain.df = dataproctrain.df.dropna(axis=0, how='any')
      df_train = dataproctrain.df
      
      ''' MUNGING TO FILL IN RELEVANT NEW DATA '''
      munger_train = MungeDataset(df_train)
      munger_train.computelineartrend()

      ''' SET COLS TO LOOK AT '''
      # set dependent variable
      # Store the independent features (and the target column) as list
      columns_train = self.columns_test + ['actual_eta']

      # Subset the relevant columns
      df_train = munger_train.df[columns_train]

#      rows = np.random.randint(0, size=len(df_train)).astype('bool')
      percentile = 0.005
      rows = np.random.binomial(1, percentile*100, size=len(df_train)).astype('bool')
      if self.X_train is not None:
        X_train = df_train.drop(self.target_column, axis=1)[rows]
        y_train = df_train[self.target_column][rows]
        
        data = np.append(X_train.as_matrix(), self.X_train.as_matrix(),axis=0)
        self.X_train = pd.DataFrame(data=data, columns=self.X_train.columns)
        data = np.append(y_train.as_matrix(), self.y_train.as_matrix(), axis=0)
        self.y_train = pd.DataFrame(data=data)#, columns=self.y_train.columns)

      print(self.X_train.shape)
      print(self.y_train.shape)
      
    def getuniqueorigins(self):
      return self.X_train.tourorigin.unique()
  
    def getuniquedest(self):
      return self.X_train.tourdestination.unique()
      
    def fit_model(self, dfX, dfY, model_name='EXTRA_TREE'):
      """
        Here you can add additional regression models from the sklearn family.
        You can also play around with the hyper parameters - the code will find the optimal set of these
        hyper parameters through cross validation.
        """
#      model_name = 'RANDOMFOREST'
      lambda_model = lambda model, model_params: model(**model_params)
      if model_name == "EXTRA_TREE":
          model = ExtraTreesRegressor()
          # model_params = {"max_depth": [5, 10, 15, 20, 25], "n_estimators": [40, 50, 70, 100]}
          model_params = {"max_depth":25, "n_estimators": 1000}

          reg = ExtraTreesRegressor(**model_params)
      elif model_name == 'LINEAR':
          model_params = {}
          reg = LinearRegression()
      elif model_name == "EXTRA_TREE2":
          model = ExtraTreesRegressor()
          # model_params = {"max_depth": [5, 10, 15, 20, 25], "n_estimators": [40, 50, 70, 100]}
          model_params = {"max_depth":30, "n_estimators": 1000}
          reg = ExtraTreesRegressor(**model_params)
      elif model_name == "SVR":
          model = SVR()
          # model_params = {"C": [1, 10, 100, 1000, 10000], "kernel" : ["rbf"], "gamma": [0.1, 0.01, 0.001, 0.0001, 1.0e-05]}
          model_params = {"C": 1, "kernel": "rbf", "gamma": 0.1}
          
          reg = SVR(**model_params)
      elif model_name == 'NN':
          model = KNeighborsRegressor()
          model_params = {'n_neighbors': [5,10,15], 'weights': ["distance"]}
          model_params = {'n_neighbors': 5, 'weights': "distance"}
          
          reg = KNeighborsRegressor(**model_params)
      elif model_name == 'RANDOMFOREST':
          model = RandomForestRegressor()
#            model_params = {'max_depth': [5,10,15], 'n_estimators':[10,50,100], 'max_features': [1,2,5]}
          model_params = {'max_depth': 30, 'n_estimators':1000, 'max_features': 2}
          
          reg =RandomForestRegressor(**model_params)
      elif model_name == "BOOST":
          model = GradientBoostingRegressor()
          model_params = {"n_estimators": [500], "learning_rate": [1.0], "max_depth": [2]}
          model_params = {"n_estimators": 500, "learning_rate": 1.0, "max_depth": 2}
          
          reg = GradientBoostingRegressor(**model_params)
      elif model_name == 'BAYESRIDGE':
          # Fit the Bayesian Ridge Regression and an OLS for comparison
          reg = BayesianRidge(compute_score=False)
      elif model_name == "DTR":
          model = DecisionTreeRegressor()
          model_params = {"max_depth": 10}
          
          reg = DecisionTreeRegressor(**model_params)
          
#        reg = GridSearchCV(estimator=model, param_grid=model_params, cv=5)
#      reg = RandomForestRegressor(**model_params)
      reg.fit(dfX, dfY)
      return reg
  
    def trainperride(self):
      # get all unique origins
      uniqueorigins = self.getuniqueorigins()
      
      models = ['EXTRA_TREE', 'RANDOMFOREST', 'LINEAR', 'BAYESRIDGE']
#      models = ['EXTRA_TREE']
      
      originmodels = []
      # train model per unique origin
      for origin in uniqueorigins:
        # get the dftrain data per unique origin
        origin_df = self.X_train.copy()
        ilocs = origin_df[origin_df['tourorigin'] == origin].index
        
        origin_df = origin_df.iloc[ilocs]
        yorigin_df = self.y_train.copy()
        yorigin_df = yorigin_df.iloc[ilocs]
        
        for model_name in models:
          # train model for this origin
          predmodel = self.fit_model(origin_df, yorigin_df, model_name='EXTRA_TREE')
          originmodels.append(predmodel)
        
      self.originmodels = originmodels
    def loaddata(self, TRAIN_FILE):
      # LOAD IN THE TRAINING DATA
      train_df = pd.read_csv(TRAIN_FILE)

      # get the preprocessed dataframe
      dataproctrain = DataPreprocessor(train_df)
#      dataproctrain = self.normalize(dataproctrain)
      df_train = dataproctrain.df
      
      ''' MUNGING TO FILL IN RELEVANT NEW DATA '''
      munger_train = MungeDataset(df_train)
      munger_train.computelineartrend()

      ''' SET COLS TO LOOK AT '''
      # set dependent variable
      # Store the independent features (and the target column) as list
      columns_train = self.columns_test + ['actual_eta']

      # Subset the relevant columns
      df_train = munger_train.df[columns_train]
      
      self.X_train = df_train.drop(self.target_column, axis=1)
      self.y_train = df_train[self.target_column]
      
      # normalize
      self.normalize()
      
    def trainwholemodel(self):
      # normalize
#      self.X_train = 
      
      # loop through train dataset and make predictions using all regressors
      for idx,reg in enumerate(self.originmodels):
        y_pred = reg.predict(self.X_train)
        
        if idx == 0:
          y_predarray = np.array(y_pred)[:, np.newaxis]
        else:
          y_predarray = np.concatenate((y_predarray, np.array(y_pred)[:, np.newaxis]), axis=1)

      # train model for this origin
      y_train = self.y_train.copy()
      print(y_predarray.shape)
      print(y_train.shape)
      predmodel = self.fit_model(y_predarray, y_train, model_name='EXTRA_TREE2')
          
      self.predmodel = predmodel
    def testwholemodel(self, TEST_FILE):
      # LOAD IN THE TRAINING DATA
      test_df = pd.read_csv(TEST_FILE)
      
      # get the preprocessed dataframe
      dataproctest = DataPreprocessor(test_df)
#      print(test_df.head())
#      print(dataproctest.df.head())
#      dataproctest.df['speed'].fillna(dataproctest.df['speed'].mean())
#      dataproctest = self.normalize(dataproctest)

      df_test = dataproctest.df
      
      ''' MUNGING TO FILL IN RELEVANT NEW DATA '''
      munger_test = MungeDataset(df_test)
      munger_test.filltraveledtime()
#      munger_test.computelineartrend()
#      munger_test.fill_dayofweek()
#      munger_test.fill_interval()

#      print(munger_test.df)
      ''' SET COLS TO LOOK AT '''
      # set dependent variable
      columns_train = self.columns_test + ['actual_eta']

      # Subset the relevant columns
      df_test = munger_test.df[columns_train]
      self.X_test = df_test
      
      # normalize
      self.normalizetest()
#      print(df_test)
      # Remove lines where actual_eta is NA as this is the part we want to predict
      print("Before Removing", self.X_test.shape)
      self.X_test = self.X_test[self.X_test.actual_eta.isnull()]
      self.X_test = self.X_test.drop('actual_eta', axis=1)
      print("After Rem oving", self.X_test.shape)
#      print(self.X_test)
      # loop through train dataset and make predictions using all regressors
      for idx,reg in enumerate(self.originmodels):
        y_pred = reg.predict(self.X_test)
        
        if idx == 0:
          y_predarray = np.array(y_pred)[:, np.newaxis]
        else:
          y_predarray = np.concatenate((y_predarray, np.array(y_pred)[:, np.newaxis]), axis=1)

      final_prediction = self.predmodel.predict(y_predarray)
      self.test_model(final_prediction)
      
      ''' BOOSTING SECTION  '''
      # ATTEMPT BOOSTING USING BEST REGRESSOR
#      boosted_model = AdaBoostRegressor(n_estimators=50, 
#                                        base_estimator=self.predmodel,
#                                        learning_rate=0.4)
#      # train model for this origin
#      y_train = self.y_train.copy()
#      # loop through train dataset and make predictions using all regressors
#      for idx,reg in enumerate(self.originmodels):
#        y_pred = reg.predict(self.X_train)
#        if idx == 0:
#          y_predarray_boost = np.array(y_pred)[:, np.newaxis]
#        else:
#          y_predarray_boost = np.concatenate((y_predarray_boost, np.array(y_pred)[:, np.newaxis]), axis=1)
#      boosted_model.fit(y_predarray_boost, y_train)
#      final_prediction = boosted_model.predict(y_predarray)
#      self.test_model(final_prediction)
      
    def test_model(self, y_pred, teamname='ETA_Masters'):
      # Store the predicted values to the disk -
      # this file you need to upload to the Cloudera workbench. this file needs to have 1023 entries.
      pd.Series(y_pred).to_csv("y_pred.csv", index=False)

      teamname = "ETA_Masters" 
      s = xmlrpc.client.ServerProxy('http://104.155.33.37:8085/RPC2') 
      print("Result Server Answer: "+ s.scoring(teamname, y_pred.tolist()))

def main():
  TRAIN = True
  if TRAIN:
    # Train the models and store their score and the best cross-validation optimized model
    model_score_list = list()
    print ("Model Scoring:",model_score_list)
    for model_name in regressors:
        reg = predmodel._get_model(model_name)
        print(reg)
        reg.fit(X_train, y_train)
        model_score_list.append((reg.best_estimator_, reg.best_score_))

    # Get the predictions of the best model
    model_scores = np.array([score[1] for score in model_score_list])

    # get bestmdoel and make prediction
    best_model_index = model_scores.argmax()
    best_model = model_score_list[best_model_index][0]

    # make final prediction
    final_prediction = best_model.predict(X_test)
    # Get the predicted ETA for the test data.
    y_pred = predmodel.test_model(final_prediction)
   
    # ATTEMPT BOOSTING USING BEST REGRESSOR
    boosted_model = AdaBoostRegressor(n_estimators=1000, 
                                      base_estimator=best_model,
                                      learning_rate=1)
    boosted_model.fit(X_train, y_train)
    final_prediction = boosted_model.predict(X_test)

    ''' 
    OUTPUT YPRED
    '''
#    print(len(final_prediction))
    # Get the predicted ETA for the test data.
    y_pred = predmodel.test_model(final_prediction)
    
  return predmodel
if __name__ == '__main__':
  data_shared = "/home/cdsw/data_shared/"
  TRAIN_FILE = os.path.join(data_shared,'df_train.csv')
  TEST_FILE = os.path.join(data_shared,"df_test_given_data.csv")
  TRAIN_DB_FILE = os.path.join(data_shared, 'df_train_db.csv')
    
  np.random.seed(12345678)  # for reproducibility, set random seed

  target_column = 'actual_eta'
  predmodel = RideModel(target_column=target_column)
  predmodel.loaddata(TRAIN_FILE)
#  predmodel.loaddbdata(TRAIN_DB_FILE)
  
#  X_train = predmodel.X_train
#  y_train = predmodel.y_train
#  X_test = predmodel.X_test
  
  predmodel.trainperride()
  predmodel.trainwholemodel()
  predmodel.testwholemodel(TEST_FILE)
# add db data to the training set
#  predmodel.loaddbdata()
#  X_traindb = predmodel.Xdb_train
##  y_traindb = predmodel.ydb_train
#  X_train = X_train.append(X_traindb)
#  y_train = y_train.append(y_traindb)

  
#  predmodel = main()
  
#  print(predmodel)
#  print(predmodel.X_test)
  print('hi')
