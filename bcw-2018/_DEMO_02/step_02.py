#
#  Simply access the dataset via Spark
#

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Route data via PySpark and HiveQL") \
.enableHiveSupport() \
.getOrCreate()

resultsHiveDF = spark.sql("SELECT * FROM tours")

resultsHiveDF.show(5)

