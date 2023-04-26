from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.mllib.linalg import DenseMatrix, SparseVector, Matrices, Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

import sys
sys.path.append('./')


from src.main.feature_construction import construct
from src.main.testClustering import test_clustering
from src.main.Metrics import getPurity

spark = SparkSession.builder.appName('Test Clustering').getOrCreate()

sc = spark.sparkContext


def test_phenotyping_with_your_data_loader():
    
    # Load the data output from the solution code
    phenotypeLabel = sc.textFile("data/phenotypeLabel.txt").map(lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
    featureTuples = sc.textFile("data/featureTuples.txt").filter(lambda x: int(x[:9]) % 17 == 0).map(lambda x: ((x.split("\t")[0], x.split("\t")[1]), float(x.split("\t")[2])))

    # Convert tuples to vector using FeatureConstruction.construct solution
    rawFeatures = construct(featureTuples)

    # Run student solution to check KMeans, GaussianMixture if they can run properly
    kMeansPurity, gaussianMixturePurity = test_clustering(phenotypeLabel, rawFeatures)

    print(f"Purity of kMeans is: {kMeansPurity:.5f}")
    print(f"Purity of GMM is: {gaussianMixturePurity:.5f}")