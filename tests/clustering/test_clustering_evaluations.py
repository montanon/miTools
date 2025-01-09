import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from mitools.clustering import (
    N_ELEMENTS_COL,
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
    agglomerative_clustering,
    clustering_ncluster_search,
    get_clusters_centroids,
    get_clusters_size,
    get_cosine_similarities_matrix,
    get_distances_between_centroids,
    get_distances_to_centroids,
    get_similarities_matrix,
    get_similarities_metric_vector,
    kmeans_clustering,
)


class TestGetClustersCentroids(TestCase):
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
            np.array_equal(result.values, expected.values)
            and np.array_equal(result.index.values, expected.index.values)
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
        self.assertTrue(np.allclose(result.values, expected.values))

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
        self.assertTrue(np.allclose(result.values, expected.values))


class TestGetDistancesToCentroids(TestCase):
    def setUp(self):
        # Mock data setup: Create a small sample DataFrame with a multi-index and centroids DataFrame
        self.data = DataFrame(
            {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "cluster": [0, 0, 1, 1]}
        ).set_index(["cluster"])
        self.centroids = DataFrame({"x": [1.5, 3.5], "y": [1.5, 3.5]}, index=[0, 1])

    def test_positive_case(self):
        # Calculate distances using the final function
        distances = get_distances_to_centroids(
            self.data, self.centroids, "cluster", metric="euclidean"
        )
        # Expected distances based on Euclidean distance calculations
        expected_distances = DataFrame(
            [0.70710678, 0.70710678, 0.70710678, 0.70710678],
            index=self.data.index,
            columns=["distance_to_cluster_centroid"],
        )
        # Compare rounded values to avoid floating point precision issues
        assert_frame_equal(distances, expected_distances, atol=1e-8)

    def test_empty_dataframes(self):
        # Create an empty DataFrame with the same structure as the original data
        empty_data = DataFrame(columns=["x", "y", "cluster"]).set_index(
            ["cluster", "x"]
        )
        # Ensure the function handles empty DataFrame correctly
        with self.assertRaises(ArgumentStructureError):
            get_distances_to_centroids(empty_data, self.centroids, "cluster")

        # Ensure the function handles empty centroids DataFrame correctly
        empty_centroids = DataFrame(columns=["x", "y"], index=[])
        with self.assertRaises(ArgumentStructureError):
            get_distances_to_centroids(self.data, empty_centroids, "cluster")

    def test_negative_case(self):
        # Test with negative values in data and centroids
        negative_data = -1 * self.data
        negative_centroids = -1 * self.centroids
        # Calculate distances for the negative values
        distances = get_distances_to_centroids(
            negative_data, negative_centroids, "cluster", metric="euclidean"
        )
        # Expected distances remain the same as absolute values are used in Euclidean distance calculation
        expected_distances = DataFrame(
            [0.70710678, 0.70710678, 0.70710678, 0.70710678],
            index=negative_data.index,
            columns=["distance_to_cluster_centroid"],
        )
        # Compare rounded values to avoid floating point precision issues
        assert_frame_equal(distances, expected_distances, atol=1e-8)

    def test_missing_index_level(self):
        # Create a DataFrame without the 'cluster' index level
        incorrect_index_data = self.data.reset_index(level="cluster")
        # Ensure function raises a KeyError when the specified cluster level is not present in the index
        with self.assertRaises(KeyError):
            get_distances_to_centroids(incorrect_index_data, self.centroids, "cluster")

    def test_different_metric(self):
        # Test with a different distance metric, e.g., 'cityblock' (Manhattan distance)
        distances = get_distances_to_centroids(
            self.data, self.centroids, "cluster", metric="cityblock"
        )
        # Expected distances based on Manhattan distance calculation
        expected_distances = DataFrame(
            [0.707107, 0.707107, 0.707107, 0.707107],
            index=self.data.index,
            columns=["distance_to_cluster_centroid"],
        )
        # Compare rounded values to avoid floating point precision issues
        assert_frame_equal(distances, expected_distances, atol=1e-8)


class TestGetSimilaritiesMetric(TestCase):
    def setUp(self):
        # Mock data setup: Create a simple DataFrame with multiple clusters
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("A", 3), ("B", 4), ("B", 5), ("B", 6)],
            names=["cluster", "sample_id"],
        )
        self.data = DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3, 0.1, 0.4, 0.5],
                "feature2": [0.4, 0.5, 0.6, 0.3, 0.7, 0.8],
            },
            index=index,
        )

    def test_positive_case_cosine_similarity(self):
        result = get_similarities_matrix(self.data, metric=cosine_similarity)
        # Manually calculate the expected cosine similarity matrix
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index,
            columns=self.data.index,
        )
        # Use pandas built-in function to assert that the matrices are equal
        assert_frame_equal(result, expected_result)

    def test_positive_case_with_id_level(self):
        result = get_similarities_matrix(
            self.data, metric=cosine_similarity, id_level="sample_id"
        )
        # Manually calculate expected cosine similarity using the specified level
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index.get_level_values("sample_id"),
            columns=self.data.index.get_level_values("sample_id"),
        )
        assert_frame_equal(result, expected_result)

    def test_positive_case_euclidean_distance(self):
        # Define a custom metric function that returns a similarity matrix using Euclidean distance
        def euclidean_similarity(X):
            distance_matrix = squareform(pdist(X, metric="euclidean"))
            return (
                1 / (1 + distance_matrix)
            )  # Convert distances to similarity (higher values indicate higher similarity)

        result = get_similarities_matrix(self.data, metric=euclidean_similarity)
        # Manually calculate the expected Euclidean similarity matrix
        expected_similarity = euclidean_similarity(self.data.values)
        expected_result = DataFrame(
            expected_similarity,
            index=self.data.index,
            columns=self.data.index,
        )
        assert_frame_equal(result, expected_result)

    def test_empty_dataframe(self):
        empty_data = DataFrame(columns=["feature1", "feature2"]).set_index(
            pd.MultiIndex.from_tuples([], names=["cluster", "sample_id"])
        )
        with self.assertRaises(ArgumentStructureError):
            get_similarities_matrix(empty_data, metric=cosine_similarity)

    def test_single_row_dataframe(self):
        single_row_df = self.data.iloc[[0]]
        with self.assertRaises(ArgumentStructureError):
            get_similarities_matrix(single_row_df, metric=cosine_similarity)

    def test_numeric_index_level(self):
        result = get_similarities_matrix(
            self.data, metric=cosine_similarity, id_level=1
        )
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index.get_level_values(1),
            columns=self.data.index.get_level_values(1),
        )
        assert_frame_equal(result, expected_result)

    def test_no_id_level_provided(self):
        result = get_similarities_matrix(
            self.data, metric=cosine_similarity, id_level=None
        )
        # Calculate expected cosine similarity values using the entire index
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index,
            columns=self.data.index,
        )
        assert_frame_equal(result, expected_result)

    def test_invalid_index_level(self):
        with self.assertRaises(KeyError):
            get_similarities_matrix(
                self.data, metric=cosine_similarity, id_level="invalid_level"
            )


class TestGetCosineSimilarities(TestCase):
    def setUp(self):
        # Mock data setup: Create a simple DataFrame with multiple clusters
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("A", 3), ("B", 4), ("B", 5), ("B", 6)],
            names=["cluster", "sample_id"],
        )
        self.data = DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3, 0.1, 0.4, 0.5],
                "feature2": [0.4, 0.5, 0.6, 0.3, 0.7, 0.8],
            },
            index=index,
        )

    def test_positive_case(self):
        result = get_cosine_similarities_matrix(self.data)
        # Manually calculate the expected cosine similarity matrix
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index,
            columns=self.data.index,
        )
        assert_frame_equal(result, expected_result)

    def test_empty_dataframe(self):
        empty_data = DataFrame(columns=["feature1", "feature2"]).set_index(
            pd.MultiIndex.from_tuples([], names=["cluster", "sample_id"])
        )
        with self.assertRaises(ArgumentStructureError):
            get_cosine_similarities_matrix(empty_data)

    def test_single_row_dataframe(self):
        single_row_df = self.data.iloc[[0]]
        with self.assertRaises(ArgumentStructureError):
            get_cosine_similarities_matrix(single_row_df)


class TestGetSimilaritiesMetricVector(TestCase):
    def setUp(self):
        # Mock data setup: Create a simple DataFrame with multiple clusters
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("A", 3), ("B", 4), ("B", 5), ("B", 6)],
            names=["cluster", "sample_id"],
        )
        self.data = DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3, 0.1, 0.4, 0.5],
                "feature2": [0.4, 0.5, 0.6, 0.3, 0.7, 0.8],
            },
            index=index,
        )

    def test_positive_case_cosine_similarity(self):
        result = get_similarities_metric_vector(
            self.data, metric=cosine_similarity, id_level="sample_id"
        )
        # Calculate expected cosine similarity values and create sample pairs
        similarity_matrix = cosine_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (
                self.data.index.get_level_values("sample_id")[i],
                self.data.index.get_level_values("sample_id")[j],
            )
            for i, j in zip(*upper_tri_indices)
        ]
        # Expected similarity vector DataFrame
        expected_result = DataFrame(
            similarity_matrix[upper_tri_indices],
            index=pd.MultiIndex.from_tuples(sample_pairs),
            columns=["cosine_similarity"],
        )
        assert_frame_equal(result, expected_result)

    def test_positive_case_euclidean_distance(self):
        # Define a custom metric function for Euclidean distance similarity
        def euclidean_similarity(X):
            distance_matrix = squareform(pdist(X, metric="euclidean"))
            return 1 / (1 + distance_matrix)  # Convert distances to similarity

        result = get_similarities_metric_vector(
            self.data, metric=euclidean_similarity, id_level="sample_id"
        )
        # Calculate expected Euclidean similarity values and create sample pairs
        similarity_matrix = euclidean_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (
                self.data.index.get_level_values("sample_id")[i],
                self.data.index.get_level_values("sample_id")[j],
            )
            for i, j in zip(*upper_tri_indices)
        ]
        # Expected similarity vector DataFrame
        expected_result = DataFrame(
            similarity_matrix[upper_tri_indices],
            index=pd.MultiIndex.from_tuples(sample_pairs),
            columns=["euclidean_similarity"],
        )
        assert_frame_equal(result, expected_result)

    def test_empty_dataframe(self):
        empty_data = DataFrame(columns=["feature1", "feature2"]).set_index(
            pd.MultiIndex.from_tuples([], names=["cluster", "sample_id"])
        )
        with self.assertRaises(ArgumentStructureError):
            get_similarities_metric_vector(
                empty_data, metric=cosine_similarity, id_level="sample_id"
            )

    def test_single_row_dataframe(self):
        single_row_df = self.data.iloc[[0]]
        with self.assertRaises(ArgumentStructureError):
            get_similarities_metric_vector(
                single_row_df, metric=cosine_similarity, id_level="sample_id"
            )

    def test_invalid_index_level(self):
        with self.assertRaises(KeyError):
            get_similarities_metric_vector(
                self.data, metric=cosine_similarity, id_level="invalid_level"
            )

    def test_numeric_index_level(self):
        result = get_similarities_metric_vector(
            self.data, metric=cosine_similarity, id_level=1
        )
        # Calculate expected cosine similarity values and create sample pairs
        similarity_matrix = cosine_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (
                self.data.index.get_level_values(1)[i],
                self.data.index.get_level_values(1)[j],
            )
            for i, j in zip(*upper_tri_indices)
        ]
        # Expected similarity vector DataFrame
        expected_result = DataFrame(
            similarity_matrix[upper_tri_indices],
            index=pd.MultiIndex.from_tuples(sample_pairs),
            columns=["cosine_similarity"],
        )
        assert_frame_equal(result, expected_result)

    def test_no_id_level_provided(self):
        result = get_similarities_metric_vector(
            self.data, metric=cosine_similarity, id_level=None
        )
        # Calculate expected cosine similarity values using the entire index
        similarity_matrix = cosine_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (self.data.index[i], self.data.index[j]) for i, j in zip(*upper_tri_indices)
        ]
        # Expected similarity vector DataFrame using the full index pairs
        expected_result = DataFrame(
            similarity_matrix[upper_tri_indices],
            index=pd.MultiIndex.from_tuples(sample_pairs),
            columns=["cosine_similarity"],
        )
        assert_frame_equal(result, expected_result)


class TestDisplayClustersSize(TestCase):
    def setUp(self):
        # Mock data setup
        self.data = DataFrame(
            {"value": [10, 20, 30, 40, 50], "cluster": [0, 0, 1, 1, 2]}
        ).set_index("cluster", append=True)

    def test_positive_case(self):
        result = get_clusters_size(self.data, "cluster")
        expected = DataFrame(
            {N_ELEMENTS_COL: [2, 2, 1]},
            index=pd.Index([0, 1, 2], name="cluster"),
        )
        assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors and returns an empty dataframe
        empty_data = DataFrame(columns=["value", "cluster"]).set_index(
            "cluster", append=True
        )
        with self.assertRaises(ArgumentStructureError):
            get_clusters_size(empty_data, "cluster")

    def test_missing_cluster_column(self):
        # Ensure function raises an error when cluster_col is missing
        with self.assertRaises(KeyError):
            get_clusters_size(self.data, "missing_cluster_col")


if __name__ == "__main__":
    unittest.main()
