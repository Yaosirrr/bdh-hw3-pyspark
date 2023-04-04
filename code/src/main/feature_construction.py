# # Dependencies
from pyspark.sql.functions import datediff, to_date, max as max_, lit, col, collect_list, row_number, concat_ws, format_number, concat, monotonically_increasing_id
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
import shutil
import os
from pyspark.sql.window import Window
import operator
import pyspark.sql.functions as F



def svmlight_convert(normalized_data, identifier_map):

  features = normalized_data.join(identifier_map, ["eventid"], 'left_outer')
  features = features.withColumn("V_tuple",concat_ws(":",features.event_index.cast(StringType()),format_number(features.normalized_feature_value,3)))

  win = Window.partitionBy("patientid").orderBy("event_index")

  features = features.withColumn("sparse_feature", collect_list("V_tuple").over(win))

  grouped_features = features.groupBy("patientid").agg(max_("sparse_feature").alias("sparse_feature"))

  return grouped_features



def svmlight_samples(grouped_features, diag_data):
    
    samples = grouped_features.join(diag_data["patientid", "label"], ["patientid"], 'left_outer')
    samples = samples.na.fill(2,["label"]) # 0 for neg, 1 for pos, 2 for unknown

    samples = samples.withColumn("sparse_feature_string", concat_ws(" ", samples.sparse_feature))

    new_col_name =  "save_feature"
    samples = samples.withColumn(new_col_name, concat_ws(" ", samples.label, samples.sparse_feature))
	
    return samples