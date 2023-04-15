# # Dependencies
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark import RDD

from typing import Tuple
from datetime import date
import sys
sys.path.append('./')

for pat in sys.path:
  print(pat)
from src.main.models import Diagnostic, Medication, LabResult
from src.main.feature_construction import *


spark = SparkSession.builder.appName('Feature Construction Test').getOrCreate()

####################################################################
def test_aggregate_one_event_diagnostic(spark):
    data = [Diagnostic("patient1", "code1", date.today())]
    diags = spark.sparkContext.parallelize(data)

    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 1.0)]
    assert actual == expected

def test_aggregate_two_different_events_diagnostic(spark):
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags = spark.sparkContext.parallelize(data)

    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 1.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

def test_aggregate_two_same_events_diagnostic(spark):
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today())]
    diags = spark.sparkContext.parallelize(data)

    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

def test_aggregate_three_events_with_duplication_diagnostic(spark):
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags = spark.sparkContext.parallelize(data)

    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 2.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

def test_filter_diagnostic(spark):
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags = spark.sparkContext.parallelize(data)

    actual = constructDiagnosticFeatureTuple(diags, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 1.0)]
    assert actual == expected

    actual = constructDiagnosticFeatureTuple(diags, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

####################################################################
def test_aggregate_one_event_medication(spark):
    data = [Medication("patient1", date.today(), "code1")]
    meds = spark.sparkContext.parallelize(data)

    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 1.0)]
    assert actual == expected

def test_aggregate_two_different_events_medication(spark):
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds = spark.sparkContext.parallelize(data)

    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 1.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

def test_aggregate_two_same_events_medication(spark):
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1")]
    meds = spark.sparkContext.parallelize(data)

    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

def test_aggregate_three_events_with_duplication_medication(spark):
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds = spark.sparkContext.parallelize(data)

    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 2.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

def test_filter_medication(spark):
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds = spark.sparkContext.parallelize(data)

    actual = constructMedicationFeatureTuple(meds, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 1.0)]
    assert actual == expected

    actual = constructMedicationFeatureTuple(meds, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

####################################################################
def test_aggregate_one_event_lab(spark):
    data = [LabResult("patient1", date.today(), "code1", 42.0)]
    labs = spark.sparkContext.parallelize(data)

    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 42.0)]
    assert actual == expected

def test_aggregate_two_different_events_lab(spark):
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code2", 24.0)]
    labs = spark.sparkContext.parallelize(data)

    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 42.0),
                (('patient1', 'code2'), 24.0)]
    assert actual == expected

def test_aggregate_two_same_events_lab(spark):
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0)]
    labs = spark.sparkContext.parallelize(data)

    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2)]
    assert actual == expected

def test_aggregate_three_events_with_duplication_lab(spark):
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0),
            LabResult("patient1", date.today(), "code2", 7475.0)]
    labs = spark.sparkContext.parallelize(data)

    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2),
                (('patient1', 'code2'), 7475.0)]
    assert actual == expected

def test_filter_lab(spark):
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0),
            LabResult("patient1", date.today(), "code2", 7475.0)]
    labs = spark.sparkContext.parallelize(data)

    actual = constructLabFeatureTuple(labs, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 7475.0)]
    assert actual == expected

    actual = constructLabFeatureTuple(labs, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2)]
    assert actual == expected


def test_construct_unique_ids(spark):
    patient_features = spark.sparkContext.parallelize([
        (("patient1", "code2"), 49.0),
        (("patient1", "code7"), 19.0),
        (("patient1", "code1"), 24.0)
    ])
    actual = construct(patient_features).collect()
    expected = [('patient1', Vectors.sparse(3, [(0, 24.0), (1, 49.0), (2, 19.0)]))]
    print(expected)
    assert actual == expected

def test_construct_sparse_vectors(spark):
    patient_features = spark.sparkContext.parallelize([
        (("patient1", "code0"), 42.0),
        (("patient1", "code2"), 24.0),
        (("patient2", "code1"), 12.0)
    ])
    actual = construct(patient_features).sortBy(lambda x: x[0]).collectAsMap()
    print(actual)
    # expected_dense = {
    #     "patient1": Vectors.dense([42.0, 0.0, 24.0]),
    #     "patient2": Vectors.dense([0.0, 12.0, 0.0])
    # }
    # print(expected_dense)

    expected = {
        "patient1": Vectors.sparse(3, [(0, 42.0), (2, 24.0)]),
        "patient2": Vectors.sparse(3, [(1, 12.0)])
    }
    print(expected)
    assert actual == expected
test_construct_unique_ids(spark)
test_construct_sparse_vectors(spark)