from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark import SparkContext
from pyspark.sql import SparkSession
from typing import Tuple
from models import Medication , LabResult ,Diagnostic
from loadRddRawData import load_rdd_raw_data
from pyspark.sql.functions import lower 

#class T2dmPhenotype:
T1DM_DX = {"250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
               "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93"}

T2DM_DX = {"250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
               "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02"}

T1DM_MED = {"lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente"}

T2DM_MED = {"chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
                "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
                "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
                "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
                "avandia", "actos", "actos", "glipizide"}

DM_RELATED_DX = {"790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648",
    "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4"}

abnormal_lab_values = {
        "HbA1c": 6.0,
        "Hemoglobin A1c": 6.0,
        "Fasting Glucose": 110.0,
        "Fasting blood glucose": 110.0,
        "fasting plasma glucose": 110.0,
        "Glucose": 110.0,
        "glucose": 110.0,
        "Glucose, Serum": 110.0
    }
    
    
def transform(medication, labResult, diagnostic):
        sc = medication.context

        groupedDiag = diagnostic.map(lambda x: (x.patientID, [x.code])).reduceByKey(lambda x, y: x + y)
        
        # data = groupedDiag.collect()
        # print(data[0])

        type_2_dm_diag = groupedDiag.filter(lambda x: not any(code in x[1] for code in T1DM_DX)) \
            .filter(lambda x: any(code in x[1] for code in T2DM_DX))
        print('first numberis xxxxxxxxxxxxxxxxxxxxxx' + str(type_2_dm_diag.count()))
            
        filtered_case_patients = type_2_dm_diag.map(lambda x: x[0]).collect()


        groupedMed = medication.map(lambda x: (x.patientID, {x.medicine.lower()})).reduceByKey(lambda a, b: a | b)

        type_1_med_order = groupedMed \
            .filter(lambda x: any(med in x[1] for med in T1DM_MED)) \
            .map(lambda x: x[0]) \
            .collect()

        casePatients1 = sc.parallelize(
            [(patient, 1) for patient in filtered_case_patients if patient not in type_1_med_order])

        filtered_type_1_med_order = [patient for patient in filtered_case_patients if patient in type_1_med_order]

        type_2_med_order = groupedMed \
            .filter(lambda x: any(med in x[1] for med in T2DM_MED)) \
            .map(lambda x: x[0]) \
            .collect()

        casePatients2 = sc.parallelize(
            [(patient, 1) for patient in filtered_type_1_med_order if patient not in type_2_med_order])

        filtered_type_2_med_order = [patient for patient in filtered_type_1_med_order if patient in type_2_med_order]

        type_1_med_order_rdd = medication.filter(lambda x: x.medicine.lower() in T1DM_MED) \
            .map(lambda x: (x.patientID, x.date.timestamp()))

        min_order_1_date = type_1_med_order_rdd.reduceByKey(lambda a, b: min(a, b))

        type_2_med_order_rdd = medication.filter(lambda x: x.medicine.lower() in T2DM_MED) \
            .map(lambda x: (x.patientID, x.date.timestamp()))

        min_order_2_date = type_2_med_order_rdd.reduceByKey(lambda a, b: min(a, b))

        join = min_order_1_date.join(min_order_2_date) \
            .filter(lambda x: x[1][1] < x[1][0]) \
            .map(lambda x: x[0]) \
            .collect()

        casePatients3 = sc.parallelize(
            [(patient, 1) for patient in filtered_type_2_med_order if patient in join])

        casePatients = casePatients1.union(casePatients2).union(casePatients3)

        # glucoseLabResults = labResult.filter(lambda x: x.resultName in abnormal_lab_values) \
        #     .filter(lambda x: x.value < abnormal_lab_values.get(x.resultName, 0.0)) \
        #     .map(lambda x: (x.patientID, {x.resultName})) \
        #     .reduceByKey(lambda a, b: a | b) \
        #     .map(lambda x: x[0]) \
        #     .collect()

        # dm_diag = groupedDiag.filter(lambda x: any(code in x[1] for code in T1DM_DX) or any(code in x[1] for code in T2DM_DX)) \
        #     .map(lambda x: x[0]) \
        #     .collect()
        
        labResult = spark.sparkContext.parallelize(labResult.collect())
        print(type(labResult))

        All_type_1 = labResult.filter(lambda x: "glucose" in x.resultName.lower()).map(lambda x: x.patientID).distinct()
        All_type = All_type_1.subtract(casePatients)
        glucose_set = set(All_type.collect())
        patients_with_glucose = labResult.filter(lambda x: x.patientID in glucose_set)
        patients_with_glucose = spark.sparkContext.parallelize(patients_with_glucose.collect())
        print(type(patients_with_glucose))
        ab1 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "hba1c" and x.value >= 6.0).map(lambda x: x.patientID)
        ab2 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "hemoglobin a1c" and x.value >= 6.0).map(lambda x: x.patientID)
        ab3 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "fasting glucose" and x.value >= 110).map(lambda x: x.patientID)
        ab4 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "fasting blood glucose" and x.value >= 110).map(lambda x: x.patientID)
        ab5 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "fasting plasma glucose" and x.value >= 110).map(lambda x: x.patientID)
        ab6 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "glucose" and x.value > 110).map(lambda x: x.patientID)
        ab7 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "glucose" and x.value > 110).map(lambda x: x.patientID)
        ab8 = patients_with_glucose.filter(lambda x: x.resultName.lower() == "glucose, serum" and x.value > 110).map(lambda x: x.patientID)
        ab1 = spark.sparkContext.parallelize(ab1.collect())
        ab2 = spark.sparkContext.parallelize(ab2.collect())
        ab3 = spark.sparkContext.parallelize(ab3.collect())
        ab4 = spark.sparkContext.parallelize(ab4.collect())
        ab5 = spark.sparkContext.parallelize(ab5.collect())
        ab6 = spark.sparkContext.parallelize(ab6.collect())
        ab7 = spark.sparkContext.parallelize(ab7.collect())
        ab8 = spark.sparkContext.parallelize(ab8.collect())
        ab = patients_with_glucose.union(ab1).union(ab2).union(ab3).union(ab4).union(ab5).union(ab6).union(ab7).union(ab8).distinct()
        ab_lab_value = ab.distinct()
        second = All_type.subtract(ab_lab_value).distinct()
        second_set = set(second.collect())

        third1 = diagnostic.filter(lambda x: x.patientID in second_set).filter(lambda x: x.code in DM_RELATED_DX).map(lambda x: x.patientID).distinct()
        third2 = diagnostic.filter(lambda x: x.code.startswith("250.")).map(lambda x: x.patientID).distinct()
        third = second.subtract(third1.union(third2)).distinct()

        control_Patients = third
        control_Patients = spark.sparkContext.parallelize(control_Patients.collect())
        controlPatients = control_Patients.map(lambda x: (x, 2))


        # controlPatients = sc.parallelize(
        #     [(patient, 2) for patient in control_Patients])

        casePatientIds = casePatients.map(lambda x: x[0]).collect()
        controlPatientIds = controlPatients.map(lambda x: x[0]).collect()

        medPatientIds = medication.map(lambda x: x.patientID).collect()
        labResultPatientIds = labResult.map(lambda x: x.patientID).collect()
        diagPatientIds = diagnostic.map(lambda x: x.patientID).collect()

        allPatientIds = set(medPatientIds).union(labResultPatientIds).union(diagPatientIds)

        others = sc.parallelize([(patient, 3) for patient in allPatientIds
                                if patient not in casePatientIds and patient not in controlPatientIds])

        phenotypeLabel = casePatients.union(controlPatients).union(others)

        return phenotypeLabel




# def stat_calc(labResult, phenotypeLabel)-> Tuple[Double, Double, Double]:
#     glucoseLabResults = labResult.filter(lambda x: (x.testName.lower() == "glucose"))

#     casePatientIds = phenotypeLabel.filter(lambda x: (x[1] == 1)).map(lambda x: x[0]).collect()
#     controlPatientIds = phenotypeLabel.filter(lambda x: (x[1] == 2)).map(lambda x: x[0]).collect()
#     otherPatientIds = phenotypeLabel.filter(lambda x: (x[1] == 3)).map(lambda x: x[0]).collect()

#     case_mean = glucoseLabResults.filter(lambda x: x.patientID in casePatientIds) \
#         .map(lambda x: x.value) \
#         .mean()
#     control_mean = glucoseLabResults.filter(lambda x: x.patientID in controlPatientIds) \
#         .map(lambda x: x.value) \
#         .mean()
#     other_mean = glucoseLabResults.filter(lambda x: x.patientID in otherPatientIds) \
#         .map(lambda x: x.value) \
#         .mean()

#     return case_mean, control_mean, other_mean

if __name__ == '__main__':
    spark = SparkSession.builder.appName("myApp").getOrCreate()
    medication_rdd, lab_result_rdd, diagnostic_rdd = load_rdd_raw_data(spark)
    data = lab_result_rdd.collect()
    #for x in data[0]: print(x)
    #print(data[0])
    phenotypeLabel = transform(medication_rdd, lab_result_rdd, diagnostic_rdd)
    #medication_rdd = spark.sparkContext.parallelize(medication_rdd.collect())
    # print(medication_rdd.count())
    # print(lab_result_rdd.count())
    print(phenotypeLabel.count())
    print(phenotypeLabel.filter(lambda x: x[1] == 1).map(lambda x: x[0]).count())
    print(phenotypeLabel.filter(lambda x: x[1] == 2).map(lambda x: x[0]).count())
    print(phenotypeLabel.filter(lambda x: x[1] == 3).map(lambda x: x[0]).count())
    print(type(phenotypeLabel))
    spark.stop()
