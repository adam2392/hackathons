#impyla_example.py

!pip3 install impyla

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

# Connect to Impala using Impyla
#
# * If you have not already established your Kerberos credentials in CDSW do so before running this script.
# * Remove auth_mechanism and use_ssl parameters on non-secure clusters.
conn = connect(host='10.132.0.19',port=21050,use_ssl=False)

cursor = conn.cursor()
cursor.execute("""CREATE DATABASE if not exists ADAM2392""")  


# Get the destinations as exact as possible
cursor = conn.cursor()
cursor.execute("""SELECT 
               name,
               tourorigin,
               tourdestination,
               remdistance,
               time,
               cureta,
               tourno
               FROM truckdata
               where remdistance < 5
               ORDER BY
               truckid,
               tourdestination,
               tourno,
               remdistance 
               LIMIT 15;""")

tables = as_pandas(cursor)
tables


###############################
# Get each ori/dest pair and their distance as exact as possible
cursor = conn.cursor()
cursor.execute("""
               DROP TABLE if EXISTS ADAM2392.oridest;

               """)

cursor.execute("""
               CREATE TABLE ADAM2392.oridest AS SELECT 
               rnk,
               name,
               tourorigin,
               tourdestination,
               remdistance as initial_distance,
               time,
               cureta,
               tourno
               FROM (
                      SELECT
                      row_number() over (partition by
                        tourorigin,tourdestination
                        order by remdistance desc) 'rnk',
                      name,
                      tourorigin,
                      tourdestination,
                      remdistance,
                      time,
                      cureta,
                      tourno
                      FROM
                      truckdata
                    ) t            
               where rnk < 2;
               """)

cursor.execute("""           
               SELECT * FROM ADAM2392.oridest LIMIT 50;
               """)

###############################

# get each tour with its start-time and end-time

cursor.execute("""
               DROP TABLE if EXISTS ADAM2392.tours;

               """)

cursor.execute("""
               CREATE TABLE ADAM2392.tours AS SELECT 
               tourorigin,
               tourdestination,
               (concat(tourorigin, "-", tourdestination)) as orig_dest,
               name,
               tourno,
               starttime,
               endtime,
               truckid,
               initial_distance,
               last_recorded_distance_at_end
               FROM (
                      SELECT
                      starts.tourorigin,
                      starts.tourdestination,
                      starts.name,
                      starts.truckid,
                      starts.tourno,
                      starttime,
                      endtime,
                      starts.remdistance as initial_distance,
                      ends.remdistance as last_recorded_distance_at_end
                      FROM (
                        SELECT
                        row_number() over (partition by
                          tourorigin,tourdestination,tourno
                          order by remdistance desc) 'rnk1',
                        tourorigin,
                        tourdestination,
                        tourno,
                        name,
                        truckid,
                        remdistance,
                        time as starttime
                        FROM
                        truckdata
                      ) starts
                      JOIN (
                        SELECT
                        row_number() over (partition by
                          tourorigin,tourdestination,tourno
                          order by remdistance asc) 'rnk2',
                        tourorigin,
                        tourdestination,
                        tourno,
                        truckid,
                        remdistance,
                        time as endtime
                        FROM
                        truckdata
                      ) ends
                      ON (
                        starts.tourorigin = ends.tourorigin
                        AND
                        starts.tourdestination = ends.tourdestination
                        AND
                        starts.tourno = ends.tourno
                        AND
                        starts.truckid = ends.truckid
                      )
                      WHERE rnk1 < 2 AND rnk2 < 2 
                    ) t            
               """)

cursor.execute("""           
               SELECT * FROM ADAM2392.tours LIMIT 200;
               """)


tables = as_pandas(cursor)
tables



###############################

# final relation Madrid/Kornwestheim

cursor.execute("""SELECT 
               t.name,
               t.tourorigin,
               t.tourdestination,
               o.orig_dest,
               t.remdistance,
               o.initial_distance,
               (o.initial_distance - t.remdistance) as
                  distance_travelled,
               t.time,
               (unix_timestamp(t.time)-unix_timestamp(o.starttime)) as
                  seconds_travelled,
               t.cureta,
               t.tourno
               FROM truckdata t
               JOIN tours o ON
               (
                t.tourorigin = o.tourorigin
                AND
                t.tourdestination = o.tourdestination
                AND
                t.tourno = o.tourno
                AND
                t.truckid = o.truckid
               )               
               ;
               """)


table = as_pandas(cursor)
table

#sns.jointplot("distance_travelled", "seconds_travelled", table, kind='reg').fig.suptitle("Arrival time Regression Madrid / Kornwestheim", y=1.01)

sns.lmplot("distance_travelled", "seconds_travelled", table, markers=["x"], col="orig_dest", col_wrap=2,scatter_kws={"s": 30},sharex=False).fig.suptitle("Arrival time Regression by origin/destination", y=1.05)

