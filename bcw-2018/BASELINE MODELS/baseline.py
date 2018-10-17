#!pip install sklearn
!pip3 install xmlrpc
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

class PredictiveModel:

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

    def test_model(self, y_pred, teamname='ETA_Masters'):
      # Store the predicted values to the disk -
      # this file you need to upload to the Cloudera workbench. this file needs to have 1023 entries.
      pd.Series(y_pred).to_csv("y_pred.csv", index=False)

      teamname = "ETA_Masters" 
      s = xmlrpc.client.ServerProxy('http://104.155.33.37:8085/RPC2') 
      print("Result Server Answer: "+ s.scoring(teamname, y_pred.tolist()))

    def _get_model(self, model_name):
        """
        Here you can add additional regression models from the sklearn family.
        You can also play around with the hyper parameters - the code will find the optimal set of these
        hyper parameters through cross validation.
        """

        if model_name == "EXTRA_TREE":
            model = ExtraTreesRegressor()
            # model_params = {"max_depth": [5, 10, 15, 20, 25], "n_estimators": [40, 50, 70, 100]}
            model_params = {"max_depth": [7], "n_estimators": [750]}

        elif model_name == "SVR":
            model = SVR()
            # model_params = {"C": [1, 10, 100, 1000, 10000], "kernel" : ["rbf"], "gamma": [0.1, 0.01, 0.001, 0.0001, 1.0e-05]}
            model_params = {"C": [1], "kernel": ["rbf"], "gamma": [0.1]}
          
        elif model_name == 'NN':
            model = KNeighborsRegressor()
            model_params = {'n_neighbors': [5,10,15], 'weights': ["distance"]}

        elif model_name == 'RANDOMFOREST':
            model = RandomForestRegressor()
#            model_params = {'max_depth': [5,10,15], 'n_estimators':[10,50,100], 'max_features': [1,2,5]}
            model_params = {'max_depth': [7], 'n_estimators':[50], 'max_features': [2]}
            
        elif model_name == "BOOST":
            model = GradientBoostingRegressor()
            model_params = {"n_estimators": [500], "learning_rate": [1.0], "max_depth": [2]}

        elif model_name == "DTR":
            model = regr_1 = DecisionTreeRegressor()
            model_params = {"max_depth": [7]}

        reg = GridSearchCV(estimator=model, param_grid=model_params, cv=5)
#        reg.fit(self.X_train, self.y_train)
        return reg

    def ploterrors(self):
    #      # leave one out analysis or cross-validation
    ##         loo = LeaveOneOut(len(X))
    #      loo = cross_validation.KFold(n=len(X), n_folds=10, shuffle=False, random_state=None)
    #
    #      # compute scores for running this regressor
    #      scores = cross_validation.cross_val_score(reg, X, y, scoring='mean_squared_error', cv=loo)
    #
    #    errors[idx1, idx2,] = [scores.mean(), scores.std()]
    #    print("MSE of %s: %f (+/- %0.5f)" % (names[idx2], scores.mean(), scores.std() * 2))
    #
    #  fig = plt.figure()
    #  plt.errorbar(errors[0,0], yerr = errors[0,1], hold=True, label=names[0])
    #  plt.errorbar(errors[1,0], yerr = errors[1,1], color='green', hold=True, label=names[1])
    #  plt.errorbar(errors[2,0], yerr = errors[2,1], color='red', hold=True, label=names[2])
    #  plt.errorbar(errors[3,0], yerr = errors[3,1], color='black', hold=True, label=names[3])
    #  plt.xscale('log')
    #  plt.xlabel('number of samples')
    #  plt.ylabel('MSE')
    #  plt.title('MSE of Regressions under simulated data')
    #  plt.axhline(1, color='red', linestyle='--')
    #  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #  plt.show()

      pass
    def loaddbdata(self):
      # LOAD IN THE TRAINING DATA
      data_shared = "/home/cdsw/data_shared/"
      TRAIN_FILE = os.path.join(data_shared,'df_train_db.csv')
      train_df = pd.read_csv(TRAIN_FILE)

      # get the preprocessed dataframe
      dataproctrain = DataPreprocessor(train_df)
      print("Before drop: ", dataproctrain.df.shape)
      dataproctrain.df = dataproctrain.df.dropna(axis=0, how='any')
      print("After drop: ", dataproctrain.df.shape)
      
      ''' SET COLS TO LOOK AT '''
      # set dependent variable
      # Store the independent features (and the target column) as list
      columns_test = ['tourorigin', 'tourdestination',
                      'lat', 'lng', 'accuracy',
                      'drivendistance', 'traveledtime',
                      'remdistance', 'elevation',
                      'heading',
#                      'day_of_week','interval',
#                      'max_remdistance', 
    #                  'timestamp',
    #                  'actual_eta',
#                      'hours', 'minutes', 'seconds', 
#                       'day', 'month', 'year'
                     ]
      columns_train = columns_test + ['actual_eta']

      
      # Subset the relevant columns
      df_train = dataproctrain.df
      
      ''' MUNGING TO FILL IN RELEVANT NEW DATA '''
      munger_train = MungeDataset(df_train)
      munger_train.computelineartrend()
#      munger_test.fill_interval()

      # Subset the relevant columns
      df_train = munger_train.df[columns_train]

      self.Xdb_train = df_train.drop(self.target_column, axis=1)
      self.ydb_train = df_train[self.target_column]

    def loaddata(self):
      # LOAD IN THE TRAINING DATA
      data_shared = "/home/cdsw/data_shared/"
      TEST_FILE = os.path.join("/home/cdsw/","processed_test_data.csv")
      TRAIN_FILE = os.path.join(data_shared,'df_train.csv')
      train_df = pd.read_csv(TRAIN_FILE)
      test_df = pd.read_csv(TEST_FILE)

      # get the preprocessed dataframe
      dataproctrain = DataPreprocessor(train_df)
      dataproctest = DataPreprocessor(test_df)

      ''' NORMALIZATION '''
      colnames = ['lat', 'lng', 
                  'elevation',
                  'remdistance', 
                  'drivendistance',
                  'heading'
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
        preprocesstype = 'unscaled'
        dataproctrain.normalizecols(colname, scaler=preprocesstype)
        dataproctest.normalizecols(colname, scaler=preprocesstype)
      
      ''' SET COLS TO LOOK AT '''
      # set dependent variable
      # Store the independent features (and the target column) as list
      columns_test = ['tourorigin', 'tourdestination',
                      'lat', 'lng', 'accuracy',
                      'drivendistance', 'traveledtime',
                      'remdistance', 'elevation',
                      'heading',
#                      'day_of_week','interval',
#                      'max_remdistance', 
    #                  'timestamp',
    #                  'actual_eta',
#                      'hours', 'minutes', 'seconds', 
#                       'day', 'month', 'year'
                     ]
      columns_train = columns_test + ['actual_eta']

      
      # Subset the relevant columns
      df_train = dataproctrain.df
      df_test = dataproctest.df
      
      ''' MUNGING TO FILL IN RELEVANT NEW DATA '''
      munger_train = MungeDataset(df_train)
      munger_test = MungeDataset(df_test)
      munger_train.computelineartrend()
      munger_test.filltraveledtime()
#      munger_test.fill_dayofweek()
#      munger_test.fill_interval()

      # Subset the relevant columns
      df_train = munger_train.df[columns_train]
      df_test = dataproctest.df[columns_train]

      self.X_train = df_train.drop(self.target_column, axis=1)
      self.y_train = df_train[self.target_column]
      self.X_test = df_test

#      return df_train, df_test

def main():
  np.random.seed(12345678)  # for reproducibility, set random seed
  ''' STILL NEED TO ADD GRIDSEFARCH CV '''
  regressors = [
#      'EXTRA_TREE',
#      'NN',
      'RANDOMFOREST',
#      'BOOST',
#      'DTR'
    ] 
  errors = np.zeros((len(regressors), 2), dtype=np.dtype('float64'))
  
  target_column = 'actual_eta'
  predmodel = PredictiveModel(target_column=target_column)
  predmodel.loaddata()
  predmodel.loaddbdata()
  X_train = predmodel.X_train
  y_train = predmodel.y_train
  X_test = predmodel.X_test
  
  # add db data to the training set
  X_traindb = predmodel.Xdb_train
  y_traindb = predmodel.ydb_train
  X_train = X_train.append(X_traindb)
  y_train = y_train.append(y_traindb)
  '''
  DO PREDICTION AND TRAINING
  '''
  print(X_train.columns)
  print(X_test.shape)
  print(y_train.shape)
#  print(X_train.head())
#  print(y_train.head())
#  print(X_test.head())

  # Remove lines where actual_eta is NA as this is the part we want to predict
  print("Before Removing", X_test.shape)
  X_test = X_test[X_test.actual_eta.isnull()]
  X_test = X_test.drop('actual_eta', axis=1)
  print("After Removing", X_test.shape)
#  print(X_test)
  
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
    boosted_model = AdaBoostRegressor(n_estimators=100, 
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
  predmodel = main()
  
  print(predmodel)
#  print(predmodel.X_test)
  print('hi')
