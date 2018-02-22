import numpy as np
import pandas as pd
from IPython.display import display, HTML
import sys
sys.path.insert(0,'/home/cdsw/EXPLORATION')
sys.path.insert(0,'/home/cdsw/')
from preprocess_final import DataPreprocessor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Imputer


data_shared = "/home/cdsw/data_shared/"
TEST_FILE = os.path.join(data_shared,'df_test_given_data.csv')
TRAIN_FILE = os.path.join(data_shared,'df_train.csv')

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

data = DataPreprocessor(train_df)
test_data = DataPreprocessor(test_df)

#heading = {'N':0,'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE': 135, 'SSE':157.5, 'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
#data.df = data.df.replace({"heading":heading})

data.df['ride_id'] = list(map(lambda ride: ride.split('_')[1], data.df['ride_id'].values))

Y = data.df['actual_eta']
data.df = data.df.drop('timestamp', axis=1)
data.df = data.df.drop('actual_eta', axis=1)
X = data.df

#test_data.df = test_data.df.replace({"heading":heading})
test_data.df['ride_id'] = list(map(lambda ride: ride.split('_')[1], test_data.df['ride_id'].values))
test_data.df = test_data.df.drop('timestamp', axis=1)
test_data.df = test_data.df.drop('actual_eta', axis=1)

test_data.df = test_data.df.interpolate(method='index', axis=0)
test_data.df[['hours','minutes', 'seconds', 'day_of_week', 'day', 'month', 'year']] = test_data.df[['hours','minutes', 'seconds', 'day_of_week', 'day', 'month', 'year']].apply(pd.to_numeric)
#test_data.df = test_data.df[test_data.df.actual_eta.isnull() == False]


display(test_data)


print(X.shape)
print(Y.shape)
print(test_data.df.shape)


#clf = Ridge(alpha=1.0)
#clf.predict(X, Y)


classifier = SVR(C=0.4,epsilon=0.8, gamma=0.1)
classifier.fit(X,Y)

hello = classifier.predict(test_data.df)
