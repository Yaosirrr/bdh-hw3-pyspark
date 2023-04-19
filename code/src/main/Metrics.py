from pyspark import rdd


def getPurity(cluster_assignment_and_label):
    """
    Given input RDD with tuples of assigned cluster id by clustering,
    and corresponding real class. Calculate the getPurity of clustering.
    Purity is defined as
                \fract{1}{N}\sum_K max_j |w_k \cap c_j|
    where N is the number of samples, K is number of clusters and j
    is index of class. w_k denotes the set of samples in k-th cluster
    and c_j denotes set of samples of class j.
    @param clusterAssignmentAndLabel RDD in the tuple format
                                     (assigned_cluster_id, class)
    @return
    :param cluster_assignment_and_label: RDD in the tuple format ((assigned_cluster_id, class), number)
    :return: double
    """
    n = cluster_assignment_and_label.count()
    tmp = cluster_assignment_and_label.map(lambda p: ((p[0], p[1]), 1.0))\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda p: (p[0][0], p[1]))\
        .reduceByKey(lambda x, y: max(x, y))\
        .map(lambda x: x[1])\
        .reduce(lambda x, y: x + y)
    return 1.0 * tmp / n
