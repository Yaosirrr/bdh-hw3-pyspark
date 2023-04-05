from pyspark.sql import SparkSession
from pyspark.mllib.feature import StandardScaler
from pyspark.ml.linalg import Vectors, DenseMatrix, Matrices
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.clustering import KMeans, GaussianMixture
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import code.src.main.Metrics


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


def test_clustering(phenotypeLabel, rawFeatures):
    print('phenotypeLabel: ' + phenotypeLabel.count())
    standardizer = StandardScaler(True, True)
    scaler = standardizer.fit(rawFeatures.map(lambda x: x[1]))
    features = rawFeatures.map(lambda x: (x[0], scaler.transform(Vectors.dense(x[1].toArray()))))
    print('features' + features.count())
    raw_feature_vectors = features.map(lambda x: x[1]).cache()
    print('raw_feature_vectors: ' + raw_feature_vectors)

    # reduce dimension
    mat = RowMatrix(raw_feature_vectors.map(Matrices.dense))
    pc = mat.computePrincipalComponents(10)
    feature_vectors = mat.multiply(pc).rows

    # dense_pc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray()).toArray().tolist()
    # dense_pc = DenseMatrix(pc.numRows, pc.numCols, dense_pc)

    def transform(feature):
        scaled = scaler.transform(Vectors.dense(feature.toArray()))
        transformed_feature = DenseMatrix(1, len(scaled.toArray()), scaled.toArray()).toArray()
        return Vectors.dense(transformed_feature)

    # train a k-means model from mllib
    kmeans = KMeans.train(feature_vectors.cache(), k=3, maxIterations=20, initializationMode="k-means||", seed=6250)

    kmeans_cluster_assignment_and_label = features.join(phenotypeLabel).map(lambda x: (kmeans.predict(transform(x[1])), x[2]))

    kmeans_purity = Metrics.getPurity(kmeans_cluster_assignment_and_label)

    # train a gmm model from mllib
    gmm = GaussianMixture.train(feature_vectors.cache(), k=3, seed=6250)
    rdd_of_vectors = features.join(phenotypeLabel).map(lambda x: transform(x[1][0]))
    labels = features.join(phenotypeLabel).map(lambda x: transform(x[1][1]))
    gmm_cluster_assignment_and_label = gmm.predict(rdd_of_vectors).zip(labels)
    gmm_purity = Metrics.getPurity(gmm_cluster_assignment_and_label)

    return kmeans_purity, gmm_purity
