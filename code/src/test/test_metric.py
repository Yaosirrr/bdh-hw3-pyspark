from pyspark import *
from pyspark.sql import SparkSession
from math import isclose
from nose.tools import with_setup, eq_, ok_,nottest, assert_almost_equals

import sys
sys.path.append("./")
from src.main.Metrics import getPurity


spark = SparkSession.builder.appName("TestMetrics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
@nottest
def setup_get_purity(spark):
    global test_input1, test_input2
    # test_input1 = spark.sparkContext.parallelize([((1, 1), 0)]) 

    test_input1 = spark.sparkContext.parallelize([
                                  (([1, 1]), 0), 
                                  (([1, 2]), 1), 
                                  (([1, 3]), 5), 
                                  (([2, 1]), 1), 
                                  (([2, 2]), 4), 
                                  (([2, 3]), 1), 
                                  (([3, 1]), 3), 
                                  (([3, 2]), 0), 
                                  (([3, 3]), 2)])
    
    test_input2 = spark.sparkContext.parallelize([
                                  (([1, 1]), 0), 
                                  (([1, 2]), 53), 
                                  (([1, 3]), 10), 
                                  (([2, 1]), 0), 
                                  (([2, 2]), 1), 
                                  (([2, 3]), 60), 
                                  (([3, 1]), 0), 
                                  (([3, 2]), 16), 
                                  (([3, 3]), 0)])

@with_setup(setup_get_purity(spark))
def test_get_purity():
    right_answer1 = (5 + 4 + 3) / 17.0
    student_purity1 = getPurity(test_input1) # output from getPurity is always 1.0
    
    assert_almost_equals(right_answer1, student_purity1, msg="UNEQUAL in first purity, Expected:%s, Actual:%s" %(right_answer1, student_purity1))

    # self.assertTrue(isclose(student_purity1, right_answer1, rel_tol=0.001))

    student_purity2 = getPurity(test_input2)
    right_answer2 = (53 + 60 + 16) / 140.0
    assert_almost_equals(right_answer2, student_purity2, msg="UNEQUAL in first purity, Expected:%s, Actual:%s" %(right_answer2, student_purity2))

    # self.assertTrue(isclose(student_purity2, right_answer2, rel_tol=0.001))