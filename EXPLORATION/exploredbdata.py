#
# Needs a Python 3 session !!!
#
#!pip install matplotlib, sklearn
'''
Basic Data Summarizer 
By: Adam Li
v1.0 - 02/21/18

'''
# !pip3 install impyla, sklearn
#
# Needs a Python 3 session !!!
#

import os
# Specify IMPALA_HOST as an environment variable in your project settings
IMPALA_HOST = os.getenv('IMPALA_HOST', '<FQDN_Impala_daemon_host>')
import pandas
import seaborn as sns
from impala.dbapi import connect
from impala.util import as_pandas

import sys
sys.path.insert(0,'/home/cdsw/EXPLORATION')
from datasummary import DataSummary

def convertactualeta():
  pass

def main():
  # Connect to Impala using Impyla
  #
  # * If you have not already established your Kerberos credentials in CDSW do so before running this script.
  # * Remove auth_mechanism and use_ssl parameters on non-secure clusters.
  conn = connect(host='10.132.0.19',port=21050,use_ssl=False)

#  cursor = conn.cursor()
#  cursor.execute("""CREATE DATABASE if not exists ADAM2392""")  
#  # Get the destinations as exact as possible
#  cursor = conn.cursor()
#  cursor.execute("""SELECT *
#                 FROM truckdata
#                 ORDER BY
#                   truckid,
#                   tourdestination,
#                   tourorigin,
#                   tourno""")
  
  '''
  # all column names
  tourorigin
  tourdestination
  lat
  lng
  accuracy
  time
  elevation
  heading
  remdistance
  cureta
  
  # not in csv files
  truckid
  name
  tourno
  isonbreak
  curiteration
  '''
  
  ###############################
  # Get each ori/dest pair and their distance as exact as possible
  cursor = conn.cursor()
  cursor.execute("""
                 DROP TABLE if EXISTS ADAM2392.rides;
  
                 """)
  
  cursor.execute("""
                 CREATE TABLE ADAM2392.rides AS SELECT 
                 ride_id,
                 name,
                 tourorigin,
                 tourdestination,
                 lat,
                 lng,
                 accuracy,
                 time,
                 elevation,
                 heading,
                 cureta,
                 remdistance
                 
                 FROM (
                      SELECT
                        row_number() over (partition by
                          truckid,tourorigin,tourdestination,tourno
                          order by remdistance desc) 'ride_id',
                         name,
                         tourorigin,
                         tourdestination,
                         lat,
                         lng,
                         accuracy,
                         time,
                         elevation,
                         heading,
                         cureta,
                         remdistance
                       FROM
                        truckdata
                      ) t;
                 """)
  
  cursor.execute("""           
                 SELECT * FROM ADAM2392.rides;
                 """)
  
  # convert SQL table to pandas
  tables = as_pandas(cursor)
#  tables.fillna(-1)
  tables['ride_id'] = 'Ride_' + tables['ride_id'].astype(str)
  tables['ride_id'] = tables['ride_id'].astype(str)
  
  # convert column names
  tables=tables.rename(columns = {'time':'timestamp'})
  tables=tables.rename(columns = {'cureta':'actual_eta'})
  
  # convert actual eta to a timedelta
  tables.actual_eta = (tables.actual_eta - tables.timestamp).dt.days*24 + (tables.actual_eta - tables.timestamp).dt.seconds/3600
  
  datasumm = DataSummary(tables)
  uniquerides = datasumm.getuniquerides()
  uniqueorigins = datasumm.getuniqueorigins()
  uniquedest = datasumm.getuniquedest()
  datasumm.getuniquerides()
#  datasumm.summarize_plots(suptitle='TRAIN DATASET')

  return datasumm
if __name__ == '__main__':
  datasumm = main()
  datasumm.df
  print(datasumm.df.columns)
  print(datasumm.df.head(5))
  
#  print((datasumm.df.actual_eta - datasumm.df.timestamp).dt.days*24 + (datasumm.df.actual_eta - datasumm.df.timestamp).dt.seconds/3600)
#  (df.fr-df.to).dt.days*24 + (df.fr-df.to).dt.seconds/3600
  data_shared = "/home/cdsw/data_shared/"
  datasumm.df.to_csv(os.path.join(data_shared, 'df_train_db.csv'), index=False)