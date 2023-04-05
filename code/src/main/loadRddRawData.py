#### new version
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime

# Define the case classes
# this is in the model part in original files
class Medication:
    def __init__(self, patientID, date, medicine):
        self.patientID = patientID
        self.date = date
        self.medicine = medicine

class LabResult:
    def __init__(self, patientID, date, resultName, value):
        self.patientID = patientID
        self.date = date
        self.resultName = resultName
        self.value = value

class Diagnostic:
    def __init__(self, patientID, code, date):
        self.patientID = patientID
        self.date = date
        self.code = code

# these are from main file
def sql_date_parser(input, pattern="yyyy-MM-dd'T'HH:mm:ssX"):
    date_format = datetime.strptime(input, pattern)
    return date_format.date()

def load_rdd_raw_data(spark):
    # Define the schema for each table
    lab_result_schema = spark.createDataFrame([], LabResult()).schema
    diagnostic_schema = spark.createDataFrame([], Diagnostic()).schema
    medication_schema = spark.createDataFrame([], Medication()).schema
    
    # Load the CSV files and create the corresponding tables
    csv_files = ["data/encounter_INPUT.csv", "data/encounter_dx_INPUT.csv", "data/lab_results_INPUT.csv", "data/medication_orders_INPUT.csv"]
    for file in csv_files:
        df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file)
        table_name = file.split("/")[-1].split(".")[0]
        df.createOrReplaceTempView(table_name)
    
    # Define the SQL queries and convert the resulting DataFrames into RDDs
    lab_result_df = spark.sql("SELECT Member_ID, Date_Resulted, Result_Name, Numeric_Result FROM lab_results_INPUT WHERE Numeric_Result IS NOT NULL and Numeric_Result<>''")
    lab_result_rdd = lab_result_df.rdd.map(lambda r: LabResult(r[0], sql_date_parser(r[1]), r[2].strip().lower(), float(r[3].replace(",", ""))))
    
    diagnostic_df = spark.sql("SELECT Member_ID, Encounter_DateTime, code FROM encounter_dx_INPUT JOIN encounter_INPUT ON encounter_dx_INPUT.Encounter_ID = encounter_INPUT.Encounter_ID")
    diagnostic_rdd = diagnostic_df.rdd.map(lambda r: Diagnostic(r[0], sql_date_parser(r[1]), r[2]))
    
    medication_df = spark.sql("SELECT Member_ID, Order_Date, Drug_Name FROM medication_orders_INPUT")
    medication_rdd = medication_df.rdd.map(lambda r: Medication(r[0], sql_date_parser(r[1]), r[2].strip().lower()))
    
    return medication_rdd, lab_result_rdd, diagnostic_rdd