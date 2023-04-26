from pyspark.sql import SparkSession

from src.main.phenotype import *
from src.main.loadRddRawData import *
from src.main.feature_construction import *
from src.main.testClustering import *



def loadLocalRawData() -> Tuple[Set[str], Set[str], Set[str]]:
    
    with open("data/med_filter.txt") as f:
        candidateMedication = set(map(str.lower, f.read().splitlines()))
    
    with open("data/lab_filter.txt") as f:
        candidateLab = set(map(str.lower, f.read().splitlines()))
    
    with open("data/icd9_filter.txt") as f:
        candidateDiagnostic = set(map(str.lower, f.read().splitlines()))
    
    return (candidateMedication, candidateLab, candidateDiagnostic)



if __name__ == "__main__":
    
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    # Set log levels
    logger = spark._jvm.org.apache.log4j
    logger.Level.WARN

    # Initialize loading of data
    medication, lab_result, diagnostic = load_rdd_raw_data(spark)
    candidate_medication, candidate_lab, candidate_diagnostic = loadLocalRawData()

    # Conduct phenotyping
    phenotype_label = transform(medication, lab_result, diagnostic)

    # Feature construction with all features
    feature_tuples = sc.union(
        constructDiagnosticFeatureTuple(diagnostic),
        constructLabFeatureTuple(lab_result),
        constructMedicationFeatureTuple(medication),
    )

    rawFeatures = construct(sc, feature_tuples)

    kMeansPurity, gaussianMixturePurity = test_clustering(phenotypeLabel, rawFeatures)
    print(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    print(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    


    filteredFeatureTuples = sc.union(
      constructDiagnosticFeatureTuple(diagnostic, candidate_diagnostic),
      constructLabFeatureTuple(lab_result, candidate_lab),
      constructMedicationFeatureTuple(medication, candidate_medication)
    )

    filteredRawFeatures = construct(sc, filteredFeatureTuples)

    kMeansPurity2, gaussianMixturePurity2 = test_clustering(phenotypeLabel, filteredRawFeatures)
    print(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    print(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
  }