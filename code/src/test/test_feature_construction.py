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
    global diags1
    data = [Diagnostic("patient1", "code1", date.today())]
    diags1 = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_one(spark))
def test_aggregate_one_event_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags1).collect()
    expected = [(('patient1', 'code1'), 1.0)]
    res= actual == expected
    eq_(res, True, "Diagnostic: test aggregate one event failed")

@nottest
def setup_diags_two_different(spark):
    global diags2
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags2 = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_two_different(spark))
def test_aggregate_two_different_events_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags2).collect()
    expected = [(('patient1', 'code1'), 1.0),
                (('patient1', 'code2'), 1.0)]
    res= actual == expected
    eq_(res, True, "Diagnostic: test aggregate two different events failed")

@nottest
def setup_diags_two_same(spark):
    global diags3
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today())]
    diags3 = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_two_same(spark))
def test_aggregate_two_same_events_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags3).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    res= actual == expected
    eq_(res, True, "Diagnostic: test aggregate two same events failed")

@nottest
def setup_diags_three(spark):
    global diags4
    data = [Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code1", date.today()),
            Diagnostic("patient1", "code2", date.today())]
    diags4 = spark.sparkContext.parallelize(data)

@with_setup(setup_diags_three(spark))
def test_aggregate_three_events_with_duplication_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags4).collect()
    expected = [(('patient1', 'code1'), 2.0),
                (('patient1', 'code2'), 1.0)]
    res= actual == expected
    eq_(res, True, "Diagnostic: test aggregate three events failed")

@with_setup(setup_diags_three(spark))
def test_filter_diagnostic():
    actual = constructDiagnosticFeatureTuple(diags4, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 1.0)]
    res = actual == expected

    actual = constructDiagnosticFeatureTuple(diags4, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    res = res and (actual == expected)
    eq_(res, True, "Diagnostic: test filter events failed")

####################################################################
@nottest
def setup_meds_one(spark):
    global meds1
    data = [Medication("patient1", date.today(), "code1")]
    meds1 = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_one(spark))
def test_aggregate_one_event_medication():
    actual = constructMedicationFeatureTuple(meds1).collect()
    expected = [(('patient1', 'code1'), 1.0)]
    res= actual == expected
    eq_(res, True, "Medication: test aggregate one event failed")

@nottest
def setup_meds_two_diff(spark):
    global meds2
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds2 = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_two_diff(spark))
def test_aggregate_two_different_events_medication():
    actual = constructMedicationFeatureTuple(meds2).collect()
    expected = [(('patient1', 'code1'), 1.0),
                (('patient1', 'code2'), 1.0)]
    res= actual == expected
    eq_(res, True, "Medication: test aggregate two different events failed")

@nottest
def setup_meds_two_same(spark):
    global meds3
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1")]
    meds3 = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_two_same(spark))
def test_aggregate_two_same_events_medication():
    actual = constructMedicationFeatureTuple(meds3).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    res= actual == expected
    eq_(res, True, "Medication: test aggregate two same events failed")

@nottest
def setup_meds_three(spark):
    global meds4
    data = [Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code1"),
            Medication("patient1", date.today(), "code2")]
    meds4 = spark.sparkContext.parallelize(data)

@with_setup(setup_meds_three(spark))
def test_aggregate_three_events_with_duplication_medication():
    actual = constructMedicationFeatureTuple(meds4).collect()
    expected = [(('patient1', 'code1'), 2.0),
                (('patient1', 'code2'), 1.0)]
    res= actual == expected
    eq_(res, True, "Medication: test aggregate three events failed")

@with_setup(setup_meds_three(spark))
def test_filter_medication():
    actual = constructMedicationFeatureTuple(meds4, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 1.0)]
    res = actual == expected

    actual = constructMedicationFeatureTuple(meds4, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 2.0)]
    res = res and (actual == expected)
    eq_(res, True, "Medication: test filter events failed")

####################################################################
@nottest
def setup_labs_one(spark):
    global labs1
    data = [LabResult("patient1", date.today(), "code1", 42.0)]
    labs1 = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_one(spark))
def test_aggregate_one_event_lab():
    actual = constructLabFeatureTuple(labs1).collect()
    expected = [(('patient1', 'code1'), 42.0)]
    res= actual == expected
    eq_(res, True, "Labs: test aggregate one event failed")

@nottest
def setup_labs_two_diff(spark):
    global labs2
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code2", 24.0)]
    labs2 = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_two_diff(spark))
def test_aggregate_two_different_events_lab():
    actual = constructLabFeatureTuple(labs2).collect()
    expected = [(('patient1', 'code1'), 42.0),
                (('patient1', 'code2'), 24.0)]
    res= actual == expected
    eq_(res, True, "Labs: test aggregate two different events failed")

@nottest
def setup_labs_two_same(spark):
    global labs3
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0)]
    labs3 = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_two_same(spark))
def test_aggregate_two_same_events_lab():
    actual = constructLabFeatureTuple(labs3).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2)]
    res= actual == expected
    eq_(res, True, "Labs: test aggregate two same events failed")

@nottest
def setup_labs_three(spark):
    global labs4
    data = [LabResult("patient1", date.today(), "code1", 42.0),
            LabResult("patient1", date.today(), "code1", 24.0),
            LabResult("patient1", date.today(), "code2", 7475.0)]
    labs4 = spark.sparkContext.parallelize(data)

@with_setup(setup_labs_three(spark))
def test_aggregate_three_events_with_duplication_lab():
    actual = constructLabFeatureTuple(labs4).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2),
                (('patient1', 'code2'), 7475.0)]
    res= actual == expected
    eq_(res, True, "Labs: test aggregate three events failed")

@with_setup(setup_labs_three(spark))
def test_filter_lab():
    actual = constructLabFeatureTuple(labs4, {"code2"}).collect()
    expected = [(('patient1', 'code2'), 7475.0)]
    res = actual == expected

    actual = constructLabFeatureTuple(labs4, {"code1"}).collect()
    expected = [(('patient1', 'code1'), 66.0 / 2)]
    res = res and (actual == expected)
    eq_(res, True, "Labs: test filter events failed")