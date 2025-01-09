import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from mitools.clustering import (
    N_ELEMENTS_COL,
    ArgumentStructureError,
    get_clusters_centroids,
    get_clusters_size,
    get_cosine_similarities_matrix,
    get_distances_between_centroids,
    get_distances_to_centroids,
    get_similarities_matrix,
    get_similarities_metric_vector,
)


class TestGetClustersCentroids(TestCase):
    def setUp(self):
        self.data = DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
        self.data["cluster"] = [0, 0, 1, 1, 2]
        self.data.set_index("cluster", inplace=True, append=True)

    def test_positive_case(self):
        result = get_clusters_centroids(self.data, "cluster")
        expected_data = {"x": [1.5, 3.5, 5], "y": [2.5, 4.5, 6]}
        expected = DataFrame(expected_data, index=[0, 1, 2])
        self.assertTrue(
            np.array_equal(result.values, expected.values)
            and np.array_equal(result.index.values, expected.index.values)
        )

    def test_empty_dataframe(self):
        empty_data = DataFrame(
            columns=["x", "y"],
            index=MultiIndex(levels=[[], []], codes=[[], []], names=[None, "cluster"]),
        )
        with self.assertRaises(ArgumentStructureError):
            get_clusters_centroids(empty_data, "cluster")

    def test_missing_cluster_index_level(self):
        with self.assertRaises(KeyError):
            get_clusters_centroids(self.data.reset_index(level="cluster"), "cluster")

    def test_single_cluster(self):
        single_cluster_data = self.data.loc[
            self.data.index.get_level_values("cluster") == 0
        ]
        with self.assertRaises(ArgumentStructureError):
            get_clusters_centroids(single_cluster_data, "cluster")


class TestGetDistancesBetweenCentroids(TestCase):
    def setUp(self):
        self.centroids = DataFrame(
            {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}, index=["A", "B", "C"]
        )

    def test_positive_case(self):
        result = get_distances_between_centroids(self.centroids)
        expected = DataFrame(
            pairwise_distances(self.centroids.values),
            index=self.centroids.index,
            columns=self.centroids.index,
        )
        self.assertTrue(np.allclose(result.values, expected.values))

    def test_empty_dataframe(self):
        empty_data = DataFrame(columns=["x", "y", "z"], index=[])
        with self.assertRaises(ArgumentStructureError):
            get_distances_between_centroids(empty_data)

    def test_single_centroid(self):
        single_centroid_data = DataFrame({"x": [1], "y": [0], "z": [0]}, index=["A"])
        result = get_distances_between_centroids(single_centroid_data)
        expected = DataFrame([[0]], index=["A"], columns=["A"])
        self.assertTrue(np.allclose(result.values, expected.values))


class TestGetDistancesToCentroids(TestCase):
    def setUp(self):
        self.data = DataFrame(
            {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "cluster": [0, 0, 1, 1]}
        ).set_index(["cluster"])
        self.centroids = DataFrame({"x": [1.5, 3.5], "y": [1.5, 3.5]}, index=[0, 1])

    def test_positive_case(self):
        distances = get_distances_to_centroids(
            self.data, self.centroids, "cluster", metric="euclidean"
        )
        expected_distances = DataFrame(
            [0.70710678, 0.70710678, 0.70710678, 0.70710678],
            index=self.data.index,
            columns=["distance_to_cluster_centroid"],
        )
        assert_frame_equal(distances, expected_distances, atol=1e-8)

    def test_empty_dataframes(self):
        empty_data = DataFrame(columns=["x", "y", "cluster"]).set_index(
            ["cluster", "x"]
        )
        with self.assertRaises(ArgumentStructureError):
            get_distances_to_centroids(empty_data, self.centroids, "cluster")
        empty_centroids = DataFrame(columns=["x", "y"], index=[])
        with self.assertRaises(ArgumentStructureError):
            get_distances_to_centroids(self.data, empty_centroids, "cluster")

    def test_negative_case(self):
        negative_data = -1 * self.data
        negative_centroids = -1 * self.centroids
        distances = get_distances_to_centroids(
            negative_data, negative_centroids, "cluster", metric="euclidean"
        )
        expected_distances = DataFrame(
            [0.70710678, 0.70710678, 0.70710678, 0.70710678],
            index=negative_data.index,
            columns=["distance_to_cluster_centroid"],
        )
        assert_frame_equal(distances, expected_distances, atol=1e-8)

    def test_missing_index_level(self):
        incorrect_index_data = self.data.reset_index(level="cluster")
        with self.assertRaises(KeyError):
            get_distances_to_centroids(incorrect_index_data, self.centroids, "cluster")

    def test_different_metric(self):
        distances = get_distances_to_centroids(
            self.data, self.centroids, "cluster", metric="cityblock"
        )
        expected_distances = DataFrame(
            [0.707107, 0.707107, 0.707107, 0.707107],
            index=self.data.index,
            columns=["distance_to_cluster_centroid"],
        )
        assert_frame_equal(distances, expected_distances, atol=1e-8)


class TestDisplayClustersSize(TestCase):
    def setUp(self):
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
        empty_data = DataFrame(columns=["value", "cluster"]).set_index(
            "cluster", append=True
        )
        with self.assertRaises(ArgumentStructureError):
            get_clusters_size(empty_data, "cluster")

    def test_missing_cluster_column(self):
        with self.assertRaises(KeyError):
            get_clusters_size(self.data, "missing_cluster_col")


class TestGetSimilaritiesMatrix(TestCase):
    def setUp(self):
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
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index,
            columns=self.data.index,
        )
        assert_frame_equal(result, expected_result)

    def test_positive_case_with_id_level(self):
        result = get_similarities_matrix(
            self.data, metric=cosine_similarity, id_level="sample_id"
        )
        expected_result = DataFrame(
            cosine_similarity(self.data.values),
            index=self.data.index.get_level_values("sample_id"),
            columns=self.data.index.get_level_values("sample_id"),
        )
        assert_frame_equal(result, expected_result)

    def test_positive_case_euclidean_distance(self):
        def euclidean_similarity(X):
            distance_matrix = squareform(pdist(X, metric="euclidean"))
            return (
                1 / (1 + distance_matrix)
            )  # Convert distances to similarity (higher values indicate higher similarity)

        result = get_similarities_matrix(self.data, metric=euclidean_similarity)
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
        similarity_matrix = cosine_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (
                self.data.index.get_level_values("sample_id")[i],
                self.data.index.get_level_values("sample_id")[j],
            )
            for i, j in zip(*upper_tri_indices)
        ]
        expected_result = DataFrame(
            similarity_matrix[upper_tri_indices],
            index=pd.MultiIndex.from_tuples(sample_pairs),
            columns=["cosine_similarity"],
        )
        assert_frame_equal(result, expected_result)

    def test_positive_case_euclidean_distance(self):
        def euclidean_similarity(X):
            distance_matrix = squareform(pdist(X, metric="euclidean"))
            return 1 / (1 + distance_matrix)  # Convert distances to similarity

        result = get_similarities_metric_vector(
            self.data, metric=euclidean_similarity, id_level="sample_id"
        )
        similarity_matrix = euclidean_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (
                self.data.index.get_level_values("sample_id")[i],
                self.data.index.get_level_values("sample_id")[j],
            )
            for i, j in zip(*upper_tri_indices)
        ]
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
        similarity_matrix = cosine_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (
                self.data.index.get_level_values(1)[i],
                self.data.index.get_level_values(1)[j],
            )
            for i, j in zip(*upper_tri_indices)
        ]
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
        similarity_matrix = cosine_similarity(self.data.values)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        sample_pairs = [
            (self.data.index[i], self.data.index[j]) for i, j in zip(*upper_tri_indices)
        ]
        expected_result = DataFrame(
            similarity_matrix[upper_tri_indices],
            index=pd.MultiIndex.from_tuples(sample_pairs),
            columns=["cosine_similarity"],
        )
        assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
