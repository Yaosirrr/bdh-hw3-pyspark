# # Dependencies
from nose.tools import with_setup, eq_, ok_,nottest
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
from src.main.loadRddRawData import *


spark = SparkSession.builder.appName('Feature Construction Test').getOrCreate()

medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)

####################################################################
def setup_module ():
    global deliverables_path 
    deliverables_path = './'
    
@nottest
def setup_unique_ids(spark):
    global patient_features 
    patient_features = spark.sparkContext.parallelize([
        (("patient1", "code2"), 49.0),
        (("patient1", "code7"), 19.0),
        (("patient1", "code1"), 24.0)
    ])

@with_setup(setup_unique_ids(spark))
def test_unique_ids():    
    temp = construct(patient_features).collect()

    res = True
    
    expected = [('patient1', Vectors.sparse(3, [(0, 24.0), (1, 49.0), (2, 19.0)]))]

    if temp != expected:
        res = False
    # if len(expected) != len(temp):
    #     res = False

    # for eve in temp:
    #     if eve not in expected:
    #         res = False
    #         break
    eq_(res, True, "feature (event) ids are missed/repeated/unsorted")

    
@nottest
def setup_sparse_vectors(spark):
    global patient_features 
    patient_features = spark.sparkContext.parallelize([
        (("patient1", "code0"), 42.0),
        (("patient1", "code2"), 24.0),
        (("patient2", "code1"), 12.0)
    ])

@with_setup(setup_sparse_vectors(spark))
def test_sparse_vectors():    
    temp = construct(patient_features).sortBy(lambda x: x[0]).collectAsMap()

    res = True
    
    expected = {
        "patient1": Vectors.sparse(3, [(0, 42.0), (2, 24.0)]),
        "patient2": Vectors.sparse(3, [(1, 12.0)])
    }
    
    if temp != expected:
        res = False
    # if len(expected) != len(temp):
    #     res = False

    # for eve in temp:
    #     if eve not in expected:
    #         res = False
    #         break
    eq_(res, True, "feature type (Vectors.sparse) or values (vector length or values) are incorrect")


####################################################################
@nottest
def setup_diags_one(spark):
    global diags
    data = [Diagnostic("patient1", "code1", date.today())]
    diags = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_one(spark))
def test_aggregate_one_event_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 1.0)]
    assert actual == expected

@nottest
def setup_diags_two_different(spark):
    global diags
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_two_different(spark))
def test_aggregate_two_different_events_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 1.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

@nottest
def setup_diags_two_same(spark):
    global diags
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today())]
    diags = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_two_same(spark))
def test_aggregate_two_same_events_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

@nottest
def setup_diags_three(spark):
    global diags
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_three(spark))
def test_aggregate_three_events_with_duplication_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags).collect()
    expected = [(('patient1', 'code1'), 2.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

@with_setup(setup_diags_three(spark))
def test_filter_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 1.0)]
    assert actual == expected

    actual = constructDiagnosticFeatureTuple(diags, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

####################################################################
@nottest
def setup_meds_one(spark):
    global meds
    data = [Medication("patient1", date.today(), "code1")]
    meds = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_one(spark))
def test_aggregate_one_event_medication():
    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 1.0)]
    assert actual == expected

@nottest
def setup_meds_two_diff(spark):
    global meds
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_two_diff(spark))
def test_aggregate_two_different_events_medication():
    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 1.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

@nottest
def setup_meds_two_same(spark):
    global meds
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1")]
    meds = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_two_same(spark))
def test_aggregate_two_same_events_medication():
    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

@nottest
def setup_meds_three(spark):
    global meds
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_three(spark))
def test_aggregate_three_events_with_duplication_medication():
    actual = constructMedicationFeatureTuple(meds).collect()
    expected = [(('patient1', 'code1'), 2.0),
                (('patient1', 'code2'), 1.0)]
    assert actual == expected

@with_setup(setup_meds_three(spark))
def test_filter_medication():
    actual = constructMedicationFeatureTuple(meds, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 1.0)]
    assert actual == expected

    actual = constructMedicationFeatureTuple(meds, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    assert actual == expected

####################################################################
@nottest
def setup_labs_one(spark):
    global labs
    data = [LabResult("patient1", date.today(), "code1", 42.0)]
    labs = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_one(spark))
def test_aggregate_one_event_lab():
    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 42.0)]
    assert actual == expected

@nottest
def setup_labs_two_diff(spark):
    global labs
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code2", 24.0)]
    labs = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_two_diff(spark))
def test_aggregate_two_different_events_lab():
    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 42.0),
                (('patient1', 'code2'), 24.0)]
    assert actual == expected

@nottest
def setup_labs_two_same(spark):
    global labs
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0)]
    labs = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_two_same(spark))
def test_aggregate_two_same_events_lab():
    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2)]
    assert actual == expected

@nottest
def setup_labs_three(spark):
    global labs
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0),
            LabResult("patient1", date.today(), "code2", 7475.0)]
    labs = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_three(spark))
def test_aggregate_three_events_with_duplication_lab():
    actual = constructLabFeatureTuple(labs).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2),
                (('patient1', 'code2'), 7475.0)]
    assert actual == expected

@with_setup(setup_labs_three(spark))
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