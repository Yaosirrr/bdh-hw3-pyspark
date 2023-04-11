from pyspark import *
from pyspark.sql import SparkSession
from Metrics import getPurity
from math import isclose


# spark = SparkSession.builder.appName("TestMetrics").getOrCreate()


def test_get_purity(self):

    test_input1 = spark.sparkContext.parallelize([(([1, 1]), 0), 
                                  (([1, 2]), 1), 
                                  (([1, 3]), 5), 
                                  (([2, 1]), 1), 
                                  (([2, 2]), 4), 
                                  (([2, 3]), 1), 
                                  (([3, 1]), 3), 
                                  (([3, 2]), 0), 
                                  (([3, 3]), 2)])
    
    test_input2 = spark.sparkContext.parallelize([(([1, 1]), 0), 
                                  (([1, 2]), 53), 
                                  (([1, 3]), 10), 
                                  (([2, 1]), 0), 
                                  (([2, 2]), 1), 
                                  (([2, 3]), 60), 
                                  (([3, 1]), 0), 
                                  (([3, 2]), 16), 
                                  (([3, 3]), 0)])
    
    student_purity1 = getPurity(test_input1)
    right_answer1 = (5 + 4 + 3) / 17.0
    self.assertTrue(isclose(student_purity1, right_answer1, rel_tol=0.001))

    student_purity2 = getPurity(test_input2)
    right_answer2 = (53 + 60 + 16) / 140.0
    self.assertTrue(isclose(student_purity2, right_answer2, rel_tol=0.001))