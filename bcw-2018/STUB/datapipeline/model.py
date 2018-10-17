
!pip3 install sklearn

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


class PredictiveModel:

    def __init__(self, filename_train_feature, filename_test_feature, target_column):
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

        self.filename_train_feature = filename_train_feature
        self.filename_test_feature = filename_test_feature
        self.target_column = target_column
        self.X_train = None
        self.X_test = None
        self.y_train = None

    def __get_model(self, model_name):
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
        
        elif model_name == "BOOST":
            model = GradientBoostingRegressor()
            model_params = {"n_estimators": [500], "learning_rate": [1.0], "max_depth": [2]}
            
        elif model_name == "DTR":
            model = regr_1 = DecisionTreeRegressor()
            model_params = {"max_depth": [7]}

        reg = GridSearchCV(estimator=model, param_grid=model_params, cv=5)
        print(reg)
        reg.fit(self.X_train, self.y_train)
        
        return reg

    def get_y_pred(self):
        """
        Estimates the model and returns the predictions
        :return: y_pred, np.array.
        """

        # Load the data
        df_train = pd.read_csv(self.filename_train_feature)
        X_train = df_train.drop(self.target_column, axis=1)
        self.y_train = df_train[self.target_column]
        X_test = pd.read_csv(self.filename_test_feature)

        # Remove non-numeric columns
        self.X_train = X_train._get_numeric_data()
        self.X_test = X_test._get_numeric_data()

        # Train the models and store their score and the best cross-validation optimized model
        model_score_list = list()
        for model_name in ["EXTRA_TREE", "SVR", "BOOST", "DTR"]:
            reg = self.__get_model(model_name)
            model_score_list.append((reg.best_estimator_, reg.best_score_))

        # Get the predictions of the best model
        #model_scores = np.array([score[1] for score in model_score_list])
        #best_model_index = model_scores.argmax()
        #best_model = model_score_list[best_model_index][0]
        
       # print("Best Model:", best_model)
        
        best_model = ExtraTreesRegressor(max_depth=7,n_estimators=750)
        best_model.fit(self.X_train, self.y_train)
        final_prediction = best_model.predict(self.X_test)
        
        #Need to try out other Boosting Algorithms
        #boosted_model = AdaBoostRegressor(n_estimators=100, base_estimator=best_model,learning_rate=1)
        #boosted_model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1)
        
        #boosted_model.fit(self.X_train, self.y_train)
        
        
       
        #final_prediction = boosted_model.predict(self.X_test)
        
        
        
        return final_prediction


def main():
    # Fix the seed for reproducibility
    np.random.seed(42)

    # Settings
    train_models = True
    prepare_data = True
    
    #data_shared = "/home/cdsw/STUB/data_shared/"
    data_shared = "/home/cdsw/"

    # Prepare data set for modelling
    if prepare_data:
        # Store file names of the original data sets
        # = data_shared + "df_train.csv"
        #filename_test_feature = data_shared + "df_test_given_data.csv"
        
        filename_train_feature = data_shared + "processed_train_data.csv"
        filename_test_feature = data_shared + "processed_test_data.csv"

        # Load the original files
        
        df_train = pd.read_csv(filename_train_feature, index_col=0)
        df_test = pd.read_csv(filename_test_feature, index_col=0)
        
       # print(df_test.head())
        
       # dp_train = DataPreprocessor(df_train)
       # dp_test = DataPreprocessor(df_test)
        
        
        
        

        # HERE YOU CAN ADD NEW FEATURES

        # Remove lines where actual_eta is NA as this is the part we want to predict
        print("Before Removing",df_test)
        df_test = df_test[df_test.actual_eta.isnull()]
      
        print("After Removing",df_test)

        # Store the independent features (and the target column) as list
        #columns_test = ['ride_id', 'tourOrigin', 'tourDestination', 'lat', 'lng', 'accuracy', 'remDistance',
                        #'elevation', 'heading']
        #columns_train = columns_test + ['actual_eta']

        # Subset the relevant columns
        
        #df_train = df_train[columns_train]
        #df_test = df_test[columns_test]
        
       # print(df_train)
        
        heading = {'N':0,'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE': 135, 'SSE':157.5, 'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}

        df_train = df_train.replace({"heading":heading})
       # df_train['ride_id'] = list(map(lambda ride: ride.split('_')[1], df_train['ride_id'].values))

        #Y = df_train['actual_eta']
        #df_train = df_train.drop('timestamp', axis=1)
        #df_train = df_train.drop('actual_eta', axis=1)
        #X = df_train

        df_test = df_test.replace({"heading":heading})
       # df_test['ride_id'] = list(map(lambda ride: ride.split('_')[1], df_test['ride_id'].values))
        df_test = df_test.drop('timestamp', axis=1)
       # df_test = df_test.drop('actual_eta', axis=1)

        df_test = df_test.interpolate(method='values')
        
        print("Interpolation",df_test)



        # Store the new train and test data to be used for modeling
        df_train.to_csv("train_data.csv", index=False)
        df_test.to_csv("test_data.csv", index=False)
        
        #data.to_csv("train_data.csv", index=False)
        #test_data.to_csv("test_data.csv", index=False)

    # Train models
    if train_models:
        # Initialize the model with path for the saved feature data
        model = PredictiveModel(filename_train_feature="train_data.csv",
                                filename_test_feature="test_data.csv",
                                target_column="actual_eta")

        # Get the predicted ETA for the test data.
        y_pred = model.get_y_pred()
        print("y_pred type:",type(y_pred))

        # Store the predicted values to the disk -
        # this file you need to upload to the Cloudera workbench. this file needs to have 1023 entries.
        pd.Series(y_pred).to_csv("y_pred.csv", index=False)
        #!pip3 install xmlrpc
        import xmlrpc.client
        teamname = "ETA_Masters" 
        s = xmlrpc.client.ServerProxy('http://104.155.33.37:8085/RPC2') 
        print("Result Server Answer: "+ s.scoring(teamname, y_pred.tolist()))

if __name__ == "__main__":
    main()