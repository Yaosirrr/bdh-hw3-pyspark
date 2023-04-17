# # Dependencies
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark import RDD

from typing import Tuple
from typing import Set
from datetime import date
import sys
sys.path.append('./')

from src.main.models import Diagnostic, Medication, LabResult
from src.main.loadRddRawData import *

FeatureTuple = Tuple[Tuple[str, str], float]

def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candidateCode: Set = None) -> RDD[FeatureTuple]:
    if candidateCode is None:
        diag = diagnostic.map(lambda x:((x.patientID, x.code), 1.0))
        diag = diag.reduceByKey(lambda a, b: a + b)

    else:
        diag = diagnostic.map(lambda x:((x.patientID, x.code), 1.0))
        diag = diag.reduceByKey(lambda a, b: a + b)
        diag = diag.filter(lambda x: x[0][1] in candidateCode)

    return diag

def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set = None) -> RDD[FeatureTuple]:
    if candidateMedication is None:
        med = medication.map(lambda x:((x.patientID, x.medicine), 1.0))
        med = med.reduceByKey(lambda a, b: a + b)
    
    else:
        med = medication.map(lambda x:((x.patientID, x.medicine), 1.0))
        med = med.reduceByKey(lambda a, b: a + b)
        med = med.filter(lambda x: x[0][1] in candidateMedication)

    return med

def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set = None) -> RDD[FeatureTuple] :
    if candidateLab is None:
        lab_sum = labResult.map(lambda x: ((x.patientID, x.resultName), x.value)).reduceByKey(lambda a, b: a + b)
        lab_count = labResult.map(lambda x: ((x.patientID, x.resultName), 1.0)).reduceByKey(lambda a, b: a + b)
        lab = lab_sum.join(lab_count).map(lambda x: (x[0], x[1][0] / x[1][1]))
    
    else:
        lab_sum = labResult.map(lambda x: ((x.patientID, x.resultName), x.value)).reduceByKey(lambda a, b: a + b)
        lab_count = labResult.map(lambda x: ((x.patientID, x.resultName), 1.0)).reduceByKey(lambda a, b: a + b)
        lab = lab_sum.join(lab_count).map(lambda x: (x[0], x[1][0] / x[1][1]))
        lab = lab.filter(lambda x: x[0][1] in candidateLab)

    return lab


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
    print(lab_result_rdd.count())
    # print(diagnostic_rdd.count())
    # print(type(medication_rdd))
    
    diagnostic_feature_tuples = constructDiagnosticFeatureTuple(diagnostic_rdd)
    print(diagnostic_feature_tuples.count())
    
    medicine_feature_tuples = constructMedicationFeatureTuple(medication_rdd)
    print(medicine_feature_tuples.count())

    lab_result_feature_tuples  = constructLabFeatureTuple(lab_result_rdd)
    print(lab_result_feature_tuples.count())

    
    data = [Medication("patient1", date.today(), "code1"),
        Medication("patient1", date.today(), "code2")]

    meds = spark.sparkContext.parallelize(data)
    print(meds.collect())
    feature_rdd = constructMedicationFeatureTuple(meds)
    print(feature_rdd.collect())
    feature_sparse_vector = construct(feature_rdd)

    spark.stop()
