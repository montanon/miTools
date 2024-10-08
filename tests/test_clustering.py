import unittest
from unittest import TestCase

from numpy import array_equal, unique
from pandas import DataFrame, MultiIndex, Series
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestCentroid

from mitools.clustering import (
    N_ELEMENTS_COL,
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
    agglomerative_clustering,
    agglomerative_ncluster_search,
    display_clusters_size,
    get_clusters_centroids,
    get_clusters_centroids_distances,
    get_cosine_similarities,
    get_distances_to_centroids,
    kmeans_clustering,
    kmeans_ncluster_search,
)


class TestKMeansClustering(TestCase):
    def setUp(self):
        # Mock data setup: Create a simple dataset for clustering
        self.data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])
        # Empty dataframe for testing
        self.empty_data = DataFrame(columns=["x", "y"])

    def test_valid_clustering(self):
        n_clusters = 3
        kmeans_model, labels = kmeans_clustering(self.data, n_clusters)
        # Check that the correct number of labels are generated
        self.assertEqual(len(labels), len(self.data))
        # Verify that the number of unique clusters matches `n_clusters`
        unique_labels = unique(labels)
        self.assertEqual(len(unique_labels), n_clusters)
        # Ensure the model object is of type `KMeans`
        self.assertIsInstance(kmeans_model, KMeans)

    def test_empty_dataframe(self):
        # Ensure the function raises an error for an empty dataframe
        with self.assertRaises(ArgumentStructureError):
            kmeans_clustering(self.empty_data, 3)

    def test_invalid_n_clusters(self):
        # Invalid `n_clusters` less than 2
        with self.assertRaises(ArgumentValueError):
            kmeans_clustering(self.data, 1)

        # Invalid `n_clusters` greater than number of samples
        with self.assertRaises(ArgumentValueError):
            kmeans_clustering(self.data, 101)

    def test_invalid_n_clusters_type(self):
        # Ensure function raises an error with non-integer n_clusters value
        with self.assertRaises(ArgumentTypeError):
            kmeans_clustering(self.data, "three")  # Passing a string instead of an int

    def test_algorithm_literal_values(self):
        # Check if function works with each valid algorithm literal
        for algo in ["auto", "full", "elkan"]:
            kmeans_model, labels = kmeans_clustering(self.data, 3, algorithm=algo)
            self.assertIsInstance(kmeans_model, KMeans)
            self.assertEqual(len(labels), len(self.data))

    def test_invalid_algorithm(self):
        # Ensure the function raises an error with an invalid algorithm value
        with self.assertRaises(ValueError):
            kmeans_clustering(self.data, 3, algorithm="invalid_algo")

    def test_custom_parameters(self):
        # Test custom parameters: n_init, max_iter, tol, verbose
        kmeans_model, labels = kmeans_clustering(
            self.data,
            3,
            random_state=42,
            n_init=20,
            max_iter=500,
            tol=1e-3,
            verbose=True,
        )
        # Validate that the KMeans model has the expected custom parameters
        self.assertEqual(kmeans_model.n_init, 20)
        self.assertEqual(kmeans_model.max_iter, 500)
        self.assertEqual(kmeans_model.tol, 1e-3)
        self.assertTrue(kmeans_model.verbose)

    def test_output_shape(self):
        # Ensure that the output has the expected shape
        kmeans_model, labels = kmeans_clustering(self.data, 3)
        self.assertEqual(labels.shape, (len(self.data),))


class TestAgglomerativeClustering(TestCase):
    def setUp(self):
        # Mock data setup: Create datasets with blobs that can be clustered
        self.data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])

    def test_positive_case(self):
        n_clusters = 3
        result = agglomerative_clustering(self.data, n_clusters).labels_
        # Ensure function returns labels for all data points
        self.assertEqual(len(result), len(self.data))
        # Ensure the number of unique labels equals n_clusters
        unique_labels = unique(result)
        self.assertEqual(len(unique_labels), n_clusters)

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(columns=["x", "y"])
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError due to empty data
            agglomerative_clustering(empty_data, 1)

    def test_invalid_n_clusters(self):
        # Ensure function raises an error with invalid n_clusters value
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError for invalid n_clusters
            agglomerative_clustering(self.data, 0)
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError for invalid n_clusters
            agglomerative_clustering(self.data, -1)


class TestGetDistancesToCentroids(TestCase):
    def setUp(self):
        # Mock data setup
        self.data = DataFrame(
            {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "cluster": [0, 0, 1, 1]}
        ).set_index(["cluster", "x"])

        self.centroids = DataFrame({"x": [1.5, 3.5], "y": [1.5, 3.5]}, index=[0, 1])

    def test_positive_case(self):
        distances = get_distances_to_centroids(self.data, self.centroids, "cluster")
        expected_distances = DataFrame(
            [0.70710678, 0.70710678, 0.70710678, 0.70710678], index=[1, 2, 3, 4]
        )
        self.assertTrue((distances.round(8) == expected_distances.round(8)).all().all())

    def test_empty_dataframe(self):
        # Test to ensure function handles empty dataframe gracefully
        empty_data = DataFrame(columns=["x", "y", "cluster"]).set_index(
            ["cluster", "x"]
        )
        distances = get_distances_to_centroids(empty_data, self.centroids, "cluster")
        self.assertTrue(distances.empty)

    def test_negative_case(self):
        distances = get_distances_to_centroids(
            -1 * self.data, -1 * self.centroids, "cluster"
        )
        expected_distances = DataFrame(
            [0.70710678, 0.70710678, 0.70710678, 0.70710678], index=[1, 2, 3, 4]
        )
        self.assertTrue((distances.round(8) == expected_distances.round(8)).all().all())


class TestGetCosineSimilarities(TestCase):
    def setUp(self):
        # Mock data setup
        self.data = DataFrame(
            {
                "x": [1, 0, 0, 1],
                "y": [0, 1, 0, 0],
                "z": [0, 0, 1, 1],
                "cluster": [0, 0, 1, 1],
            }
        ).set_index("cluster")

    def test_positive_case(self):
        result = get_cosine_similarities(self.data, "cluster")
        # Verify for cluster 0
        expected_cluster_0 = cosine_similarity(self.data.loc[0])
        self.assertTrue((result[0] == expected_cluster_0).all())
        # Verify for cluster 1
        expected_cluster_1 = cosine_similarity(self.data.loc[1])
        self.assertTrue((result[1] == expected_cluster_1).all())

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(columns=["x", "y", "z", "cluster"]).set_index("cluster")
        result = get_cosine_similarities(empty_data, "cluster")
        self.assertTrue(empty_data.equals(result))

    def test_single_item_cluster(self):
        # Single item in a cluster should return a 1x1 DataFrame with value 1
        single_item_data = DataFrame(
            {"x": [1], "y": [0], "z": [0], "cluster": [0]}
        ).set_index("cluster")
        result = get_cosine_similarities(single_item_data, "cluster")
        expected = Series([[[[1.0]]]], name="cluster").astype("object")
        self.assertTrue(result.equals(expected))


class TestDisplayClustersSize(TestCase):
    def setUp(self):
        # Mock data setup
        self.data = DataFrame(
            {"value": [10, 20, 30, 40, 50], "cluster": [0, 0, 1, 1, 2]}
        )

    def test_positive_case(self):
        result = display_clusters_size(self.data, "cluster")
        expected = DataFrame(
            {N_ELEMENTS_COL: [2, 2, 1]},
            index=MultiIndex.from_tuples([(0,), (1,), (2,)], names=["cluster"]),
        )
        self.assertTrue(result.equals(expected))

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors and returns an empty dataframe
        empty_data = DataFrame(columns=["value", "cluster"])
        result = display_clusters_size(empty_data, "cluster")
        expected = DataFrame(
            columns=[N_ELEMENTS_COL],
            index=MultiIndex(levels=[[]], codes=[[]], names=["cluster"]),
        ).astype(int)
        self.assertTrue(result.equals(expected))

    def test_missing_cluster_column(self):
        # Ensure function raises an error when cluster_col is missing
        with self.assertRaises(KeyError):
            display_clusters_size(self.data, "missing_cluster_col")


class TestGetClustersCentroidsDistances(TestCase):
    def setUp(self):
        # Mock centroids setup
        self.centroids = DataFrame({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]})

    def test_positive_case(self):
        result = get_clusters_centroids_distances(self.centroids)
        expected = DataFrame(pairwise_distances(self.centroids))
        # Use numpy's allclose for floating point comparisons
        self.assertTrue((result.values == expected.values).all())

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(columns=["x", "y", "z"])
        with self.assertRaises(ArgumentStructureError):
            get_clusters_centroids_distances(empty_data)

    def test_single_centroid(self):
        # For a single centroid, the pairwise distance should be a 1x1 DataFrame with value 0
        single_centroid_data = DataFrame({"x": [1], "y": [0], "z": [0]})
        result = get_clusters_centroids_distances(single_centroid_data)
        expected = DataFrame([[0]])
        self.assertTrue((result.values == expected.values).all())


class TestGetClustersCentroids(TestCase):
    def setUp(self):
        # Mock data setup
        self.data = DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
        self.data["cluster"] = [0, 0, 1, 1, 2]
        self.data.set_index("cluster", inplace=True, append=True)

    def test_positive_case(self):
        result = get_clusters_centroids(self.data, "cluster")
        # Calculate the expected centroids manually
        expected_data = {"x": [1.5, 3.5, 5], "y": [2.5, 4.5, 6]}
        expected = DataFrame(expected_data, index=[0, 1, 2])
        # Ensure centroids are computed correctly
        self.assertTrue(
            array_equal(result.values, expected.values)
            and array_equal(result.index.values, expected.index.values)
        )

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(
            columns=["x", "y"],
            index=MultiIndex(levels=[[]], codes=[[]], names=["cluster"]),
        )
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError due to empty data
            get_clusters_centroids(empty_data, "cluster")

    def test_missing_cluster_column(self):
        # Ensure function raises an error when cluster_col is missing
        with self.assertRaises(KeyError):
            get_clusters_centroids(
                self.data.reset_index(level="cluster"), "missing_cluster_col"
            )

    def test_single_cluster(self):
        # For a single cluster, the centroid should be the mean of the data
        single_cluster_data = self.data[
            self.data.index.get_level_values("cluster") == 0
        ]
        with self.assertRaises(ArgumentStructureError):
            get_clusters_centroids(single_cluster_data, "cluster")


class TestAgglomerativeNClusterSearch(TestCase):
    def setUp(self):
        # Mock data setup: Create datasets with blobs
        self.data, _ = make_blobs(n_samples=100, centers=4, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])

    def test_positive_case(self):
        max_clusters = 5
        result = agglomerative_ncluster_search(self.data, max_clusters)

        # Ensure function returns scores for all cluster counts from 2 to max_clusters - 1
        self.assertEqual(len(result), max_clusters - 2)

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(columns=["x", "y"])
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError due to empty data
            agglomerative_ncluster_search(empty_data, 10)

    def test_invalid_max_clusters(self):
        # Ensure function raises an error with invalid max_clusters value
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError for invalid max_clusters
            agglomerative_ncluster_search(self.data, 1)


class TestKMeansNClusterSearch(TestCase):
    def setUp(self):
        # Mock data setup: create datasets with blobs that can be clustered
        self.data, _ = make_blobs(n_samples=100, centers=4, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])

    def test_positive_case(self):
        max_clusters = 4
        silhouette_scores, inertia_values = kmeans_ncluster_search(
            self.data, max_clusters
        )

        # Ensure function returns scores and inertia for all cluster counts from 2 to max_clusters - 1
        self.assertEqual(len(silhouette_scores), max_clusters - 2)
        self.assertEqual(len(inertia_values), max_clusters - 2)

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(columns=["x", "y"])
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError due to empty data
            kmeans_ncluster_search(empty_data, 10)

    def test_invalid_max_clusters(self):
        # Ensure function raises an error with invalid max_clusters value
        with self.assertRaises(
            ValueError
        ):  # Assuming it raises a ValueError for invalid max_clusters
            kmeans_ncluster_search(self.data, 1)


if __name__ == "__main__":
    unittest.main()
