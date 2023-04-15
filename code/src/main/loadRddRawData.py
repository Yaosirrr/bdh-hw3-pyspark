#### new version
# import sys
# sys.path.append("..")
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from pyspark.sql.types import StringType, BooleanType, IntegerType, DoubleType
#import main.models as models
from src.main.models import Medication , LabResult ,Diagnostic

# Define the case classes
# this is in the model part in original files

# these are from main file
def sql_date_parser(input, pattern="yyyy-MM-dd'T'HH:mm:ssX"):
    date_format = datetime.strptime(input, pattern)
    return date_format.date()

def load_rdd_raw_data(spark):
    # Define the schema for each table
    #lab_result_schema = spark.createDataFrame([], LabResult()).schema
    
#     lab_result_schema = StructType([
#     StructField("member_id", StringType(), True),
#     StructField("date_resulted", DateType(), True),
#     StructField("result_name", StringType(), True),
#     StructField("numeric_result", FloatType(), True)
# ])
#     diagnostic_schema = spark.createDataFrame([], Diagnostic()).schema
#     medication_schema = spark.createDataFrame([], Medication()).schema
    
    # Load the CSV files and create the corresponding tables
    # csv_files = ["../../data/encounter_INPUT.csv", "../../data/encounter_dx_INPUT.csv", "../../data/lab_results_INPUT.csv", "../../data/medication_orders_INPUT.csv"]
    relative_path = "./data/"
    file_name = ["encounter_INPUT.csv",\
                "encounter_dx_INPUT.csv", \
                "lab_results_INPUT.csv", \
                 "medication_orders_INPUT.csv"]
    
    csv_files = [relative_path+x for x in file_name]

    for file in csv_files:
        df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file)
        table_name = file.split("/")[-1].split(".")[0]
        df.createOrReplaceTempView(table_name)
    
    # Define the SQL queries and convert the resulting DataFrames into RDDs
    lab_result_df = spark.sql("SELECT Member_ID, Date_Resulted, Result_Name, Numeric_Result FROM lab_results_INPUT WHERE Numeric_Result IS NOT NULL and Numeric_Result<>''")
    lab_result_rdd = lab_result_df.rdd.map(lambda r: LabResult(r[0], r[1], r[2].strip().lower(), float(r[3].replace(",", ""))))
    
    diagnostic_df = spark.sql("SELECT Member_ID, Encounter_DateTime, Code_ID FROM encounter_dx_INPUT JOIN encounter_INPUT ON encounter_dx_INPUT.Encounter_ID = encounter_INPUT.Encounter_ID")
    diagnostic_rdd = diagnostic_df.rdd.map(lambda r: Diagnostic(r[0], r[2], r[1]))
    
    medication_df = spark.sql("SELECT Member_ID, Order_Date, Drug_Name FROM medication_orders_INPUT")
    medication_rdd = medication_df.rdd.map(lambda r: Medication(r[0], r[1], r[2].strip().lower()))
    
    return medication_rdd, lab_result_rdd, diagnostic_rdd


if __name__ == '__main__':
    spark = SparkSession.builder.appName("myApp").getOrCreate()
    medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)
    medication_rdd = spark.sparkContext.parallelize(medication_rdd.collect())
    print(medication_rdd.count())
    print(lab_result_rdd.count())
    print(diagnostic_rdd.count())
    print(type(medication_rdd))
    spark.stop()