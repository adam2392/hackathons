from scipy import stats
import numpy as np
import pandas as pd
import xmlrpc.client
from IPython.display import display, HTML

import sys
sys.path.insert(0,'/home/cdsw/EXPLORATION')
sys.path.insert(0,'/home/cdsw/')

from preprocess_final import DataPreprocessor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

#!pip3 install xmlrpc

no_of_methods = 6
data_shared = "/home/cdsw/data_shared/"
TEST_FILE = os.path.join(data_shared,'df_test_given_data.csv')
TRAIN_FILE = os.path.join(data_shared,'df_train.csv')

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

data = DataPreprocessor(train_df)
data.df['ride_id'] = list(map(lambda ride: ride.split('_')[1], data.df['ride_id'].values))

test_data = DataPreprocessor(test_df)
test_data.df['ride_id'] = list(map(lambda ride: ride.split('_')[1], test_data.df['ride_id'].values))

#features = ['ride_id', 'tourorigin', 'tourdestination', 'lat', 'lng', 'accuracy',\
#            'remdistance', 'elevation', 'heading', 'timestamp', 'actual_eta', \
#            'hours', 'minutes', 'seconds', 'day_of_week', 'day', 'month', 'year',\
#            'interval', 'max_remdistance', 'drivendistance', 'traveledtime']

features = ['ride_id','elevation', 'tourorigin', 'tourdestination', 'lat', 'lng', 'remdistance',\
            'max_remdistance', 'drivendistance', 'actual_eta']

#Y = data.df[['actual_eta', 'traveledtime']]
Y = data.df[['actual_eta']]

#X = data.df[features].drop('traveledtime', axis=1)
#X = X.drop('actual_eta', axis=1)

X = data.df[features].drop('actual_eta', axis=1)

test_data.df = test_data.df[features]

null_indices = np.array(np.where(test_data.df['actual_eta'].notnull() == False))[0]
not_null_indices = np.array(np.where(test_data.df['actual_eta'].notnull() == True))[0]

not_null_test_data = test_data.df['actual_eta'][not_null_indices]

prediction_values = np.zeros([len(test_data.df['actual_eta']), no_of_methods])
predictions_null_indices = np.zeros([len(null_indices), no_of_methods])
predictions_not_null_indices = np.zeros([len(not_null_indices), no_of_methods])

test_data.df = test_data.df.drop('actual_eta', axis=1)
#test_data.df = test_data.df.drop('traveledtime', axis=1)

####### K NEIGHBORS #####
print("####### K NEIGHBORS #####")
neigh = KNeighborsRegressor(n_neighbors=30, weights='distance')
neigh.fit(X, Y)

prediction_values[:,0] = neigh.predict(test_data.df)[:,0]

predictions_null_indices[:,0] = prediction_values[:,0][null_indices]
predictions_not_null_indices[:,0] = prediction_values[:,0][not_null_indices]


####### K NEIGHBORS #####


#######MULTI OUTPUT REGRESSOR #####
print("#######MULTI OUTPUT REGRESSOR #####")
knn = KNeighborsRegressor()
regr = MultiOutputRegressor(knn)

regr.fit(X,Y)
prediction_values[:,1] = regr.predict(test_data.df)[:,0]

predictions_null_indices[:,1] = prediction_values[:,1][null_indices]
predictions_not_null_indices[:,1] = prediction_values[:,1][not_null_indices]


#######MULTI OUTPUT REGRESSOR #####


######## SVR ###########
print("######## SVR ###########")
classifier = SVR(C=1.0,epsilon=0.6, gamma=0.05)
classifier.fit(X, Y)

prediction_values[:,2] = classifier.predict(test_data.df)

predictions_null_indices[:,2] = prediction_values[:,2][null_indices]
predictions_not_null_indices[:,2] = prediction_values[:,2][not_null_indices]

######## SVR ###########

######## EXTRA TREES ###########
print("######## EXTRA TREES ###########")
ext_clf = ExtraTreesRegressor(n_estimators = 600, max_depth = 10)
ext_clf.fit(X,Y)

prediction_values[:,3] = ext_clf.predict(test_data.df)

predictions_null_indices[:,3] = prediction_values[:,3][null_indices]
predictions_not_null_indices[:,3] = prediction_values[:,3][not_null_indices]

######## EXTRA TREES ###########

######## GRADIENT BOOSTING ###########
print("######## GRADIENT BOOSTING ###########")
params = {'n_estimators': 700, 'max_depth': 10, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'huber', 'max_features':'auto'}
boost_clf = GradientBoostingRegressor(**params)

boost_clf.fit(X, Y)

prediction_values[:,4] = boost_clf.predict(test_data.df)

predictions_null_indices[:,4] = prediction_values[:,4][null_indices]
predictions_not_null_indices[:,4] = prediction_values[:,4][not_null_indices]

######## GRADIENT BOOSTING ###########

########## MLP ###########

mlp_clf = MLPRegressor(hidden_layer_sizes=(100, 80), activation='relu',solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001)
mlp_clf.fit(X,Y)

prediction_values[:,5] = mlp_clf.predict(test_data.df)

predictions_null_indices[:,5] = prediction_values[:,5][null_indices]
predictions_not_null_indices[:,5] = prediction_values[:,5][not_null_indices]

########## MLP ###########

print(predictions_null_indices)
print(predictions_not_null_indices)
print(not_null_test_data)

m = []
c = []
for i in range(no_of_methods):
  em, cee = np.linalg.lstsq(np.vstack([predictions_not_null_indices[:,i], np.ones(len(predictions_not_null_indices[:,0]))]).T, not_null_test_data)[0]
  m.append(em)
  c.append(cee)

std_e = []
slopes = []
const = []
for i in range(no_of_methods):
  slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_not_null_indices[:,i],not_null_test_data)
  std_e.append(std_err)
  slopes.append(slope)
  const.append(intercept)

#print(m)
#print(c)
#print(prediction_values)
#for i in range(no_of_methods):
#  prediction_values[:,i] = slopes[i] * prediction_values[:,i] + const[i]
  #prediction_values[:,i] = m[i] * prediction_values[:,i] + c[i]
  
#pd.Series(values).to_csv("output_values.csv", index=False)
#y_pred = pd.read_csv("output_values.csv", header=None).iloc[:, 0].values

for i in range(no_of_methods):
  s = xmlrpc.client.ServerProxy('http://104.155.33.37:8085/RPC2')
  print("Result Server Answer: "+ s.scoring("ETA_Masters",prediction_values[:,i][null_indices].tolist()))