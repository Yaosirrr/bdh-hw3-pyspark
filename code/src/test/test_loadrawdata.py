from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from pyspark.sql.types import StringType, BooleanType, IntegerType, DoubleType
from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest,assert_is
import sys
sys.path.append("..")
from main.loadRddRawData import load_rdd_raw_data

def test_loadrddrawdata():
    numTrueMedication = 31552
    numTrueLabResult = 106894
    numTrueDiagnostic = 112811
    spark = SparkSession.builder.appName("myApp").getOrCreate()
    
    medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)
    numMedication = medication_rdd.count()
    numLabResult = lab_result_rdd.count()
    numDiagnostic = diagnostic_rdd.count()

    assert_almost_equals( numTrueMedication ,numMedication, msg="UNEQUAL in Medication_rdd, Expected:%s, Actual:%s" %( numTrueMedication ,numMedication))
    assert_almost_equals(numTrueLabResult,numLabResult, msg="UNEQUAL in LabResult_rdd, Expected:%s, Actual:%s" %(numTrueLabResult,numLabResult))
    assert_almost_equals( numTrueDiagnostic,numDiagnostic, msg="UNEQUAL in Diagnostic_rdd, Expected:%s, Actual:%s" %(numTrueDiagnostic,numDiagnostic))

