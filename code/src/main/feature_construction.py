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
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark import RDD
from models import Diagnostic, Medication, LabResult
from typing import Tuple

spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()

sc = spark.sparkContext

FeatureTuple = Tuple[Tuple[str, str], float]

def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]) -> RDD[FeatureTuple]:

  diag = diagnostic.map(lambda x:((x.patientID, x.code), 1.0))
  diag = diag.reduceByKey(lambda a, b: a + b)
  return diag

def constructMedicationFeatureTuple(medication):

  med = medication.map(lambda x:((x.patientID, x.medicine), 1.0))
  med = med.reduceByKey(lambda a, b: a + b)
  return med

def constructLabFeatureTuple(labResult):

  lab_sum = labResult.map(lambda x: ((x.patientID, x.reslultName), x.value)).reduceByKey(lambda a, b: a + b)
  lab_count = labResult.map(lambda x: ((x.patientID, x.reslultName), 1.0)).reduceByKey(lambda a, b: a + b)
  lab = lab_sum.join(lab_count).map(lambda x: (x[0], x[1][0] / x[1][1]))
  return lab

def constructDiagnosticFeatureTuple(diagnostic, candidateCode):

  diag = diagnostic.map(lambda x:((x.patientID, x.code), 1.0))
  diag = diag.reduceByKey(lambda a, b: a + b)
  diag_feature = diag.filter(lambda x: x[0][1] in candidateCode)
  return diag_feature

def constructMedicationFeatureTuple(medication, candidateMedication):

  med = medication.map(lambda x:((x.patientID, x.medicine), 1.0))
  med = med.reduceByKey(lambda a, b: a + b)
  med_feature = med.filter(lambda x: x[0][1] in candidateMedication)
  return med_feature

def constructLabFeatureTuple(labResult, candidateLab):
  
  lab_sum = labResult.map(lambda x: ((x.patientID, x.reslultName), x.value)).reduceByKey(lambda a, b: a + b)
  lab_count = labResult.map(lambda x: ((x.patientID, x.reslultName), 1.0)).reduceByKey(lambda a, b: a + b)
  lab = lab_sum.join(lab_count).map(lambda x: (x[0], x[1][0] / x[1][1]))
  lab_feature = lab.filter(lambda x: x[0][1] in candidateLab)
  return lab_feature

def construct(feature):
  print(feature)
  print(feature.collect())

  feature_names = feature.map(lambda x: x[0][1]).distinct()
  print(feature_names.collect())
  feature_num = feature_names.count()
  print(feature_num)
  feat_idx_map = feature_names.zipWithIndex()
  print(feat_idx_map.collect())
  feat_table = feature.map(lambda x:(x[0][1], (x[0][0], x[1]))).join(feat_idx_map)
  print(feat_table.collect())

  # [('DIAG4', (('19992', 1.0), 0)), ('DRUG7', (('19992', 0.9), 1))]                
  idxed_features = feat_table.map(lambda x: (x[1][0][0],(x[1][1], x[1][0][1])))
  print(idxed_features.collect())
  grouped_features = idxed_features.groupByKey().mapValues(list)
  print(grouped_features.collect())
  # vectorized_grouped_features = grouped_features.map((lambda x: (x[0],[y[0] for y in x[1]], [y[1] for y in x[1]])))
  # print(vectorized_grouped_features.collect())

  result = grouped_features.map(lambda x: (x[0], Vectors.sparse(feature_num, x[1])))
  print(result.collect())
  return result
  
featureTuple = ((str,str), float)

from pyspark import RDD

#Create RDD from parallelize    
data = [(('19992', 'DIAG4'), 1.000),(('19992', 'DRUG7'), 0.900)]
rdd=sc.parallelize(data)
a = construct(rdd)
