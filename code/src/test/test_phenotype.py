from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from pyspark.sql.types import StringType, BooleanType, IntegerType, DoubleType
from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest,assert_is
import sys
sys.path.append("..")
from main.phenotype import transform
from main.loadRddRawData import load_rdd_raw_data

def test_phenotype():
    numTrueCases = 976
    numTrueControl = 948
    numTrueOthers = 3688 - numTrueCases - numTrueControl
    spark = SparkSession.builder.appName("myApp").getOrCreate()

    medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)
    phenotypeLabel = transform(medication_rdd, lab_result_rdd, diagnostic_rdd)
    numCases = phenotypeLabel.filter(lambda x: x[1] == 1).count()
    numControl = phenotypeLabel.filter(lambda x: x[1] == 2).count()
    numOthers = phenotypeLabel.filter(lambda x: x[1] == 3).count()

    assert_almost_equals( numTrueCases ,numCases, msg="UNEQUAL in Cases_patients, Expected:%s, Actual:%s" %( numTrueCases ,numCases))
    assert_almost_equals( numTrueControl ,numControl, msg="UNEQUAL in Control_patients, Expected:%s, Actual:%s" %( numTrueControl ,numControl))
    assert_almost_equals( numTrueOthers ,numOthers, msg="UNEQUAL in Others_patients, Expected:%s, Actual:%s" %( numTrueOthers ,numOthers))



