#
# Use a Hive-Context to access the data and
# convert records into labeled points
#

!pip install numpy



import os,sys
from scipy import stats
import numpy as np



from pyspark import SparkContext, HiveContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS



table_name = "tours"
target_col = "last_recorded_distance_at_end"



sc = SparkContext()
hc = HiveContext(sc)

# get the table from the hive context
df = hc.table(table_name) 



columnNames = df.columns
columnNames = ["last_recorded_distance_at_end","tourno", "truckid","initial_distance" ]



# reorder columns so that we know the index of the target column
df = df.select(target_col, *[col for col in columnNames if col != target_col])

df.printSchema()

rdd_of_labeled_points = df.rdd.map(lambda row: LabeledPoint(row[0], row[1:]))

rdd_of_labeled_points.take(1)
 





