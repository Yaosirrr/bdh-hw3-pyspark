from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from pyspark.sql.types import StringType, BooleanType, IntegerType, DoubleType
from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest,assert_is

import sys
sys.path.append("./")
from src.main.models import Diagnostic, Medication, LabResult
from src.main.phenotype import transform
from src.main.loadRddRawData import load_rdd_raw_data

global spark
spark = SparkSession.builder.appName('Feature Construction Test').getOrCreate()
# spark.SparkContext.setLogLevel("ERROR")

@nottest
def setup_phenotype(spark):
    global patient_features
    patient_features = spark.sparkContext.parallelize([
        (("patient1", "code2"), 49.0),
        (("patient1", "code7"), 19.0),
        (("patient1", "code1"), 24.0)
    ])

# @with_setup()
def test_phenotype(spark):
    numTrueCases = 976
    numTrueControl = 948
    numTrueOthers = 3688 - numTrueCases - numTrueControl

    medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)
    phenotypeLabel = transform(medication_rdd, lab_result_rdd, diagnostic_rdd)
    numCases = phenotypeLabel.filter(lambda x: x[1] == 1).count()
    numControl = phenotypeLabel.filter(lambda x: x[1] == 2).count()
    numOthers = phenotypeLabel.filter(lambda x: x[1] == 3).count()
    
    print(numCases, numControl, numOthers)
    assert_almost_equals( numTrueCases ,numCases, msg="UNEQUAL in Cases_patients, Expected:%s, Actual:%s" %( numTrueCases ,numCases))
    assert_almost_equals( numTrueControl ,numControl, msg="UNEQUAL in Control_patients, Expected:%s, Actual:%s" %( numTrueControl ,numControl))
    assert_almost_equals( numTrueOthers ,numOthers, msg="UNEQUAL in Others_patients, Expected:%s, Actual:%s" %( numTrueOthers ,numOthers))

test_phenotype(spark)

