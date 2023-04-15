# # Dependencies
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark import RDD

from typing import Tuple
from datetime import date
import sys
sys.path.append('./')

from src.main.models import Diagnostic, Medication, LabResult
from src.main.loadRddRawData import *

FeatureTuple = Tuple[Tuple[str, str], float]

def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]) -> RDD[FeatureTuple]:

    diag = diagnostic.map(lambda x:((x.patientID, x.code), 1.0))
    diag = diag.reduceByKey(lambda a, b: a + b)
    return diag

def constructMedicationFeatureTuple(medication: RDD[Medication]) -> RDD[FeatureTuple]:

    med = medication.map(lambda x:((x.patientID, x.medicine), 1.0))
    med = med.reduceByKey(lambda a, b: a + b)
    return med

def constructLabFeatureTuple(labResult: RDD[LabResult]) -> RDD[FeatureTuple] :

    lab_sum = labResult.map(lambda x: ((x.patientID, x.reslultName), x.value)).reduceByKey(lambda a, b: a + b)
    lab_count = labResult.map(lambda x: ((x.patientID, x.reslultName), 1.0)).reduceByKey(lambda a, b: a + b)
    lab = lab_sum.join(lab_count).map(lambda x: (x[0], x[1][0] / x[1][1]))
    return lab

'''
Functions below need to be renamed. Otherwise, they will override
same-named functions above since Python is fully override.

If you want to use the same function name, set 'candidateCode' as optional input
or simply modify function logic as:

def constructDiagnosticFeatureTuple(diagnostic, candidateCode=None):
    diag = diagnostic.map(lambda x:((x.patientID, x.code), 1.0))
    diag = diag.reduceByKey(lambda a, b: a + b)
    if candidateCode is not None:
      diag_feature = diag.filter(lambda x: x[0][1] in candidateCode)
    return diag_feature

then, you can only use three functions, instead of six
'''
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
    feature_names = feature.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x)
    feature_num = feature_names.count()
    feat_idx_map = feature_names.zipWithIndex()
    feat_table = feature.map(lambda x:(x[0][1], (x[0][0], x[1]))).join(feat_idx_map)
    idxed_features = feat_table.map(lambda x: (x[1][0][0],(x[1][1], x[1][0][1])))
    grouped_features = idxed_features.groupByKey().mapValues(list)
    result = grouped_features.map(lambda x: (x[0], Vectors.sparse(feature_num, x[1])))
    return result
'''
Functions above need to be renamed. Otherwise, they will override
same-named functions above since Python is fully override.
'''


if __name__ == '__main__':
    spark = SparkSession.builder.appName('Construct Features').getOrCreate()

    sc = spark.sparkContext
    
    medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)
    # medication_rdd = spark.sparkContext.parallelize(medication_rdd.collect())
    # print(medication_rdd.count())
    # print(lab_result_rdd.count())
    # print(diagnostic_rdd.count())
    # print(type(medication_rdd))
    
    diagnostic_feature_tuples = constructDiagnosticFeatureTuple(lab_result_rdd)
    medicine_feature_tuples = constructMedicationFeatureTuple(medication_rdd)
    lab_result_feature_tuples  = constructLabFeatureTuple(diagnostic_rdd)
    
    
    data = [Medication("patient1", date.today(), "code1"),
        Medication("patient1", date.today(), "code2")]

    meds = spark.sparkContext.parallelize(data)
    print(meds.collect())
    feature_rdd = constructMedicationFeatureTuple(meds)
    print(feature_rdd.collect())
    feature_sparse_vector = construct(feature_rdd)

    spark.stop()
