from pyspark import SparkContext
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.mllib.linalg import DenseMatrix, SparseVector, Matrices, Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

import sys
sys.path.append('./')


from src.main.testClustering import test_clustering
from src.main.Metrics import getPurity

sc = SparkContext


def test_phenotyping_with_your_data_loader():
    
    # Load the data output from the solution code
    phenotypeLabel = sc.textFile("data/phenotypeLabel/part-*").map(lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
    featureTuples = sc.textFile("data/featureTuples/part-*").filter(lambda x: int(x[:9]) % 17 == 0).map(lambda x: ((x.split("\t")[0], x.split("\t")[1]), float(x.split("\t")[2])))

    # Convert tuples to vector using FeatureConstruction.construct solution
    rawFeatures = construct(sc, featureTuples)

    # Run student solution to check KMeans, GaussianMixture if they can run properly
    kMeansPurity, gaussianMixturePurity = test_clustering(phenotypeLabel, rawFeatures)

    print(kMeansPurity)
    print(gaussianMixturePurity)
    
    def construct(sc, feature):
        
        feature_names = feature.map(lambda t: t[0][1]).distinct()
        feature_num = feature_names.count().toInt()
        feat_idx_map = feature_names.zipWithIndex()
        fat_table = feature.map(lambda t: (t[0][1], (t[0][0], t[1]))).join(feat_idx_map)
        idxed_features = fat_table.map(lambda t: (t[1][0][0], (t[1][1].toInt(), t[1][0][1])))
        grouped_features = idxed_features.groupByKey()
        result = grouped_features.map(lambda t: (t[0], SparseVector(feature_num, t[1].map(lambda r: r[0]).toArray(), t[1].map(lambda r: r[1]).toArray())))
       
        return result