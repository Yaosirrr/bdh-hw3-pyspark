# # Dependencies
from nose.tools import with_setup, eq_, ok_, nottest, assert_almost_equals, nottest,assert_is

from pyspark.sql.functions import datediff, to_date, max as max_, lit, col, collect_list, row_number, concat_ws, format_number, concat, monotonically_increasing_id
from src.etl import calculate_index_dates,filter_events, aggregate_events, generate_feature_mapping, normalization, svmlight_convert, svmlight_samples
import pyspark.sql.functions as F

# from src.etl import *
import datetime 
import pandas as pd

import os, errno
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()

sc = spark.sparkContext



@nottest
def setup_svm_convert():
    global normalized_df, mapping_df 

    normalized = [(19992, 'DIAG4', 1.000),
        (19992, 'DIAG8', 1.000),
        (19993, 'LAB3', 0.500),
        (19993, 'DIAG4', 0.667)]
    columns_normalized = ["patientid", "eventid", "normalized_feature_value"]
    normalized_df = spark.createDataFrame(data=normalized, schema=columns_normalized)

    mapping = [('DIAG4', 2),
        ('DIAG8', 9),
        ('LAB3', 12)]
    columns_mapping = ["eventid", "event_index"]
    mapping_df = spark.createDataFrame(data=mapping, schema=columns_mapping)

@with_setup (setup_svm_convert)
def test_svm_convert():
    # INPUT:
    # normalized
    # e.g.
    # +---------+------------+------------------------+
    # |patientid|     eventid|normalized_feature_value|
    # +---------+------------+------------------------+
    # |    20459|  LAB3023103|                   0.062|
    # |    20459|  LAB3027114|                   1.000|
    # |    20459|  LAB3007461|                   0.115|
    # +---------+------------+------------------------+

    # event_map
    # e.g.
    # +----------+-----------+
    # |   eventid|event_index|
    # +----------+-----------+
    # |DIAG132797|          0|
    # |DIAG135214|          1|
    # |DIAG137829|          2|
    # +----------+-----------+

    # +---------+-------------------+
    # |patientid|   sparse_feature  |
    # +---------+-------------------+
    # |    19992|[2:1.000, 9:1.000] |
    # |    19993|[2:0.667, 12:0.500]|
    # +---------+-------------------+

    svmlight = svmlight_convert(normalized_df, mapping_df)
    svmlight.show()

 
    res = True
    temp = svmlight.rdd.map(lambda x: (x['patientid'], x['sparse_feature'])).collect()

    expected = {19992:["2:1.000", "9:1.000"], 19993:["2:0.667", "12:0.500"]}

    res = True
    if len(expected) != len(temp):
        res = False

    for pid, feat in temp:
        if pid not in expected.keys():
            res = False
            break
        elif feat != expected[pid]:
            res = False
            break

    print("Actual: ", end = " ")
    print(temp)
    expected_tuples = []
    for p in zip(expected.keys(), expected.values()):
        expected_tuples.append(p)
    # expected_tuples = [(19992, ["2:1.000", "9:1.000"]), (19993, ["2:0.667", "12:0.500"])]

    print("Expected: ", end = " ")
    print((expected_tuples))

    eq_(res, True, "Svmlight conversion is not correct!")




@nottest
def setup_svm_samples():
    global svmlight_df, diag_df

    svmlight_data = [(19994, ["2:1.000", "97:0.667"]),
                     (19993, ["1:0.500", "97:0.667"]),
                  (19992, ["2:1.000", "2001:1.000"])]
    columns_svmlight = ["patientid", "sparse_feature"]
    svmlight_df = spark.createDataFrame(data=svmlight_data, schema=columns_svmlight)

    diag_df = [(19994, datetime.date(2000, 4, 19), 0), (19993, datetime.date(2000, 9, 19), 1)]
    columns_diag = ["patientid", "mtimestamp", "label"]
    diag_df = spark.createDataFrame(data=diag_df, schema=columns_diag)

@with_setup(setup_svm_samples)
def test_svm_samples():
    # INPUT:
    # svmlight
    # +---------+--------------------+
    # |patientid|      sparse_feature|
    # +---------+--------------------+
    # |     5206|[4:1.000, 5:1.000...|
    # |    13905|[1:1.000, 11:1.00...|
    # |    18676|[0:1.000, 2:1.000...|
    # |    20301|[10:1.000, 12:1.0...|
    # |    20459|[136:0.250, 137:1...|
    # +---------+--------------------+

    # mortality
    # +---------+----------+-----+
    # |patientid|mtimestamp|label|
    # +---------+----------+-----+
    # |    13905|2000-01-30|    1|
    # |    18676|2000-02-03|    1|
    # |    20301|2002-08-08|    1|
    # +---------+----------+-----+

    # OUTPUT
    # samples
    # +---------+--------------------+-------------+--------------------+
    # |patientid|      sparse_feature|other columns|        save_feature|
    # +---------+--------------------+-------------+--------------------+
    # |     5206|[4:1.000, 5:1.000...|     ...     |0 4:1.000 5:1.000...|
    # |    13905|[1:1.000, 11:1.00...|     ...     |1 1:1.000 11:1.00...|
    # |    18676|[0:1.000, 2:1.000...|     ...     |1 0:1.000 2:1.000...|
    # |    20301|[10:1.000, 12:1.0...|     ...     |1 10:1.000 12:1.0...|
    # |    20459|[136:0.250, 137:1...|     ...     |0 136:0.250 137:1...|
    # +---------+--------------------+-------------+--------------------+



    expected = [[19992, "2 2:1.000 2001:1.000"], [19993, "1 1:0.500 97:0.667"], [19994, "0 2:1.000 97:0.667"]]

    samples = svmlight_samples(svmlight_df, diag_df)
    temp = samples.rdd.map(lambda x: [x['patientid'], x['save_feature']]).collect()
    print(temp)
    res = True
    if len(expected) != len(temp):
        res = False

    for feat in temp:
        if feat not in expected:
            res = False
            break

    print("Actual: ", end = " ")
    print(temp)

    print("Expected: ", end = " ")
    print(expected)

    eq_(res, True, "Svmlight feature string is not correct!")
