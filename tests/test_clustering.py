import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose, array_equal, unique
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
    clustering_ncluster_search,
    display_clusters_size,
    get_clusters_centroids,
    get_cosine_similarities,
    get_distances_between_centroids,
    get_distances_to_centroids,
    kmeans_clustering,
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
        # Mock data setup: Create a simple dataset for clustering
        self.data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])
        # Empty dataframe for testing
        self.empty_data = DataFrame(columns=["x", "y"])

    def test_valid_clustering(self):
        n_clusters = 3
        agg_model, labels = agglomerative_clustering(self.data, n_clusters)
        # Check that the correct number of labels are generated
        self.assertEqual(len(labels), len(self.data))
        # Verify that the number of unique clusters matches `n_clusters`
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), n_clusters)
        # Ensure the model object is of type `AgglomerativeClustering`
        self.assertIsInstance(agg_model, AgglomerativeClustering)

    def test_empty_dataframe(self):
        # Ensure the function raises an error for an empty dataframe
        with self.assertRaises(ArgumentStructureError):
            agglomerative_clustering(self.empty_data, 3)

    def test_invalid_n_clusters(self):
        # Invalid `n_clusters` less than 2
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 1)

        # Invalid `n_clusters` greater than number of samples
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 101)

    def test_invalid_n_clusters_type(self):
        # Ensure function raises an error with non-integer n_clusters value
        with self.assertRaises(ArgumentTypeError):
            agglomerative_clustering(
                self.data, "three"
            )  # Passing a string instead of an int

    def test_ward_metric_error(self):
        # Ensure the function raises an error when using a non-euclidean metric with "ward" linkage
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 3, metric="manhattan", linkage="ward")

    def test_distance_threshold_error(self):
        # Ensure the function raises an error when `distance_threshold` is set along with `n_clusters`
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 3, distance_threshold=5.0)

        # Ensure the function raises an error when `distance_threshold` is set and `compute_full_tree` is False
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(
                self.data, 3, distance_threshold=5.0, compute_full_tree=False
            )

    def test_output_shape(self):
        # Ensure that the output has the expected shape
        agg_model, labels = agglomerative_clustering(self.data, 3)
        self.assertEqual(labels.shape, (len(self.data),))

    def test_custom_parameters(self):
        # Test with custom metric and linkage
        agg_model, labels = agglomerative_clustering(
            self.data,
            3,
            metric="cosine",
            linkage="complete",
            compute_full_tree=True,
        )
        # Ensure model has the correct parameters set
        self.assertEqual(agg_model.metric, "cosine")
        self.assertEqual(agg_model.linkage, "complete")

    def test_precomputed_metric(self):
        # Test with precomputed distance matrix
        distance_matrix = np.linalg.norm(
            self.data.values[:, np.newaxis] - self.data.values, axis=2
        )
        agg_model, labels = agglomerative_clustering(
            DataFrame(distance_matrix), 3, metric="precomputed", linkage="complete"
        )
        self.assertEqual(len(labels), len(self.data))


class TestClusteringNClusterSearch(unittest.TestCase):
    def setUp(self):
        # Mock data setup: Create datasets with blobs for clustering
        self.data, _ = make_blobs(n_samples=100, centers=4, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])
        # Empty dataframe for testing
        self.empty_data = DataFrame(columns=["x", "y"])

    def test_positive_case_kmeans(self):
        # Test positive case with KMeans clustering
        max_clusters = 5
        silhouette_scores, inertia_values = clustering_ncluster_search(
            self.data,
            max_clusters=max_clusters,
            clustering_method=kmeans_clustering,
            random_state=42,
            n_init=10,
        )
        # Ensure function returns scores and inertia for all cluster counts from 2 to max_clusters - 1
        self.assertEqual(len(silhouette_scores), max_clusters - 2)
        self.assertEqual(len(inertia_values), max_clusters - 2)

    def test_positive_case_agglomerative(self):
        # Test positive case with Agglomerative clustering
        max_clusters = 5
        silhouette_scores, inertia_values = clustering_ncluster_search(
            self.data,
            max_clusters=max_clusters,
            clustering_method=agglomerative_clustering,
            linkage="ward",
            metric="euclidean",
        )
        # Ensure function returns scores for all cluster counts from 2 to max_clusters - 1
        self.assertEqual(len(silhouette_scores), max_clusters - 2)
        self.assertIsNone(inertia_values)  # Inertia should be None for Agglomerative

    def test_empty_dataframe(self):
        # Ensure function raises an error for an empty dataframe for both clustering methods
        with self.assertRaises(ArgumentStructureError):
            clustering_ncluster_search(
                self.empty_data, 10, clustering_method=kmeans_clustering
            )

        with self.assertRaises(ArgumentStructureError):
            clustering_ncluster_search(
                self.empty_data, 10, clustering_method=agglomerative_clustering
            )

    def test_invalid_max_clusters(self):
        # Ensure function raises an error with invalid `max_clusters` value
        with self.assertRaises(ArgumentValueError):
            clustering_ncluster_search(
                self.data, 1, clustering_method=kmeans_clustering
            )

        with self.assertRaises(ArgumentValueError):
            clustering_ncluster_search(
                self.data, 1, clustering_method=agglomerative_clustering
            )

    def test_invalid_clustering_method(self):
        # Ensure function raises an error if an invalid clustering method is provided
        with self.assertRaises(TypeError):
            clustering_ncluster_search(
                self.data, 10, clustering_method="invalid_method"
            )

    def test_valid_custom_metric(self):
        # Test with a custom silhouette scoring metric
        def custom_metric(data, labels):
            # A mock scoring function that just returns a constant value
            return 0.5

        max_clusters = 5
        silhouette_scores, _ = clustering_ncluster_search(
            self.data,
            max_clusters=max_clusters,
            clustering_method=kmeans_clustering,
            score_metric=custom_metric,
            random_state=42,
            n_init=10,
        )

        # Ensure that all the silhouette scores match the custom metric value
        self.assertTrue(all(score == 0.5 for score in silhouette_scores))


class TestGetClustersCentroids(unittest.TestCase):
    def setUp(self):
        # Mock data setup: Create a simple dataset with a multi-index that includes cluster labels
        self.data = DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
        self.data["cluster"] = [0, 0, 1, 1, 2]
        # Set cluster labels as an index level (along with the default range index)
        self.data.set_index("cluster", inplace=True, append=True)

    def test_positive_case(self):
        # Compute centroids using the refactored function
        result = get_clusters_centroids(self.data, "cluster")
        # Expected centroids calculated manually
        expected_data = {"x": [1.5, 3.5, 5], "y": [2.5, 4.5, 6]}
        expected = DataFrame(expected_data, index=[0, 1, 2])
        # Ensure centroids are computed correctly
        self.assertTrue(
            array_equal(result.values, expected.values)
            and array_equal(result.index.values, expected.index.values)
        )

    def test_empty_dataframe(self):
        # Test with an empty dataframe with a 'cluster' index level
        empty_data = DataFrame(
            columns=["x", "y"],
            index=MultiIndex(levels=[[], []], codes=[[], []], names=[None, "cluster"]),
        )

        # Ensure function raises ArgumentStructureError for empty data
        with self.assertRaises(ArgumentStructureError):
            get_clusters_centroids(empty_data, "cluster")

    def test_missing_cluster_index_level(self):
        # Ensure function raises an error when the specified index level is missing
        # Here we remove the cluster index level and attempt to access it
        with self.assertRaises(KeyError):
            get_clusters_centroids(self.data.reset_index(level="cluster"), "cluster")

    def test_single_cluster(self):
        # Create a DataFrame with a single unique cluster label
        single_cluster_data = self.data.loc[
            self.data.index.get_level_values("cluster") == 0
        ]

        # Ensure function raises an ArgumentStructureError for single-group data
        with self.assertRaises(ArgumentStructureError):
            get_clusters_centroids(single_cluster_data, "cluster")


class TestGetDistancesBetweenCentroids(TestCase):
    def setUp(self):
        # Mock centroids setup: Create a simple DataFrame with 3 centroids
        self.centroids = DataFrame(
            {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}, index=["A", "B", "C"]
        )

    def test_positive_case(self):
        # Compute pairwise distances using the function
        result = get_distances_between_centroids(self.centroids)
        # Expected pairwise distance DataFrame
        expected = DataFrame(
            pairwise_distances(self.centroids.values),
            index=self.centroids.index,
            columns=self.centroids.index,
        )
        # Use numpy's allclose for floating-point comparisons
        self.assertTrue(allclose(result.values, expected.values))

    def test_empty_dataframe(self):
        # Test with an empty DataFrame with appropriate columns
        empty_data = DataFrame(columns=["x", "y", "z"], index=[])
        # Ensure the function raises ArgumentStructureError for empty DataFrame
        with self.assertRaises(ArgumentStructureError):
            get_distances_between_centroids(empty_data)

    def test_single_centroid(self):
        # Create a DataFrame with a single centroid
        single_centroid_data = DataFrame({"x": [1], "y": [0], "z": [0]}, index=["A"])
        # Compute distances using the function
        result = get_distances_between_centroids(single_centroid_data)
        # Expected pairwise distance matrix for a single point (1x1 matrix with 0)
        expected = DataFrame([[0]], index=["A"], columns=["A"])
        # Use numpy's allclose for floating-point comparisons
        self.assertTrue(allclose(result.values, expected.values))


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


if __name__ == "__main__":
    unittest.main()
