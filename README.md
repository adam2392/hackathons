# Getting Started with CDSW and Python

CDSW allows easy coding in a Big-Data environment.

## Where are the docs?

[Logistics challenge documentation (hit the download button on the upper right on the next page)](http://cdsw.datahack2018.de/kamir/truckdata/files/Documentation___DOWNLOAD_ME_PLEASE.pdf)

[CDSW documentation](https://www.cloudera.com/documentation/data-science-workbench/latest/topics/cdsw_user_guide.html)

[PySpark Cheat-Sheet](https://www.datacamp.com/community/blog/pyspark-sql-cheat-sheet)

This baseline project shows how to get the most out of [Python](http://ipython.org) 
on Cloudera Data Science Workbench. To begin, open the project workbench.

## DEMO_01
* `analysis.py` -- An example Python analysis script.

## DEMO_02
* `step_01.py` -- Python example: Access Hive table via Impala
* `step_02.py` -- Python example: Access Hive table via SparkSQL
* `step_03.py` -- Python example: Create RDD with LabeledVectors from Hive table


## formats for datetime
http://strftime.org/

## below code for timeStamp comversion
train['timeStamp_date'] = pd.to_datetime(train['timeStamp'], format = '%Y-%m-%dT%H:%M:%S.%fZ')

## CentOS installation
https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-centos-7

## Datetime to unix time
https://stackoverflow.com/questions/15203623/convert-pandas-datetimeindex-to-unix-time

## datetime to int64
a = pd.DatetimeIndex(time)
a.astype(np.int64)//10**9

## Test code
times= pd.to_datetime(train_df['timestamp'], format = '%Y-%m-%dT%H:%M:%S.%fZ')
  print(times)
  a = pd.DatetimeIndex(times)
#  a.astype(np.int64)//10**9
  print(a)
  
##Bus ETA Prediction which aggregates Bus Ride Data and Weather Data
https://github.com/chrisdaly/Bus-ETA-Prediction/blob/master/Notebooks/7%20-%20Models.ipynb

# merge avg speed
df_train = pd.merge(df_train, avgSpeed, on=['tourOrigin', 'tourDestination'])
df_test = pd.merge(df_test, avgSpeed, on=['tourOrigin', 'tourDestination'])

# BOSCH CONNECTED WORLD 2018 (2/21 - 2/22) WORK
Here, I briefly describe the work done for the BCW hackathon that I took part in. We worked with the data up above and wanted to predict truck ETA given data streams. We were given a training dataset with some features and a test dataset, and we had to perform regression given time samples to make a prediction on the missing ETA values.

The code is dirty and reflects the massive iteration we went through as a team of 6 while working on a Cloudera sponsored jupyter-lab style work bench. 

EXPLORATION/
    This directory examined the data in terms of visualization and worked on syncing the impala database with certain features down using SQL. 

    - datasummary: creates a data summary visualization of the training and testing datasets using matplotlib and seaborn and some pandas
    - exploredbdata: also explores the db dataset and creates a csv file after doing some JOINs and MERGEs to create the dataset similar to our given train/test csv files.
    - preprocess_final: this was a preprocessing module that was fed in a dataframe and essentially performed all necessary preprocessing and feature engineering that we thought of to come up with our superset-dataframe that was used in model building.

BASELINE MODELS/
    This directory was looking at different models built to analyze the data.

    - baseline: This was a script to setup some baseline single models to analyze the data and come up with some baseline accuracy. 
    - perridemodel: This was the work that I did on iterating through the models and coming up with a hierarchical training pipeline that trained a separate model for each beginning ride origin, and then training a master model on top of that to make the final predictions of the eta.

## Results
From the initial SVR model given to us by Bosch data scientists, we improved on the model by ~25% mainly attributed to feature engineering and performing the hierarchical per-ride training of models.
