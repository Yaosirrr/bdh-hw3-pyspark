from pyspark import SparkContext
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import GaussianMixture, KMeans, StreamingKMeans
from pyspark.mllib.linalg import DenseMatrix, SparseVector, Matrices, Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

import sys
sys.path.append('./')


from model import student_clustering
from metric import getPurity

sc = SparkContext

def test_phenotyping_with_your_data_loader(self):
    
    # Load the data output from the solution code
    phenotypeLabel = self.sc.textFile("data/phenotypeLabel/part-*").map(lambda x: (x.split("\t")[0], int(x.split("\t")[1])))
    featureTuples = self.sc.textFile("data/featureTuples/part-*").filter(lambda x: int(x[:9]) % 17 == 0).map(lambda x: ((x.split("\t")[0], x.split("\t")[1]), float(x.split("\t")[2])))

    # Convert tuples to vector using FeatureConstruction.construct solution
    rawFeatures = construct(sc, featureTuples)

    # Run student test clustering code
    stu_kMeansPurity, stu_gaussianMixturePurity = student_clustering(phenotypeLabel, rawFeatures)



    # Run official solution clustering code just to check KMeans, GaussianMixture, and StreamingKMeans
    kMeansPurity, gaussianMixturePurity = test_clustering(phenotypeLabel, rawFeatures)

    # ==========================================================================
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
    
    def construct(sc, feature):
        
        feature_names = feature.map(lambda t: t[0][1]).distinct()
        feature_num = feature_names.count().toInt()
        feat_idx_map = feature_names.zipWithIndex()
        fat_table = feature.map(lambda t: (t[0][1], (t[0][0], t[1]))).join(feat_idx_map)
        idxed_features = fat_table.map(lambda t: (t[1][0][0], (t[1][1].toInt(), t[1][0][1])))
        grouped_features = idxed_features.groupByKey()
        result = grouped_features.map(lambda t: (t[0], SparseVector(feature_num, t[1].map(lambda r: r[0]).toArray(), t[1].map(lambda r: r[1]).toArray())))
       
        return result