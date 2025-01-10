import unittest
from unittest import TestCase

import numpy as np
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs

from mitools.clustering import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
    agglomerative_clustering,
    clustering_ncluster_search,
    kmeans_clustering,
)


class TestKMeansClustering(TestCase):
    def setUp(self):
        self.data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])
        self.empty_data = DataFrame(columns=["x", "y"])

    def test_valid_clustering(self):
        n_clusters = 3
        kmeans_model, labels = kmeans_clustering(self.data, n_clusters)
        self.assertEqual(len(labels), len(self.data))
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), n_clusters)
        self.assertIsInstance(kmeans_model, KMeans)

    def test_empty_dataframe(self):
        with self.assertRaises(ArgumentStructureError):
            kmeans_clustering(self.empty_data, 3)

    def test_invalid_n_clusters(self):
        with self.assertRaises(ArgumentValueError):
            kmeans_clustering(self.data, 1)

        with self.assertRaises(ArgumentValueError):
            kmeans_clustering(self.data, 101)

    def test_invalid_n_clusters_type(self):
        with self.assertRaises(ArgumentTypeError):
            kmeans_clustering(self.data, "three")  # Passing a string instead of an int

    def test_algorithm_literal_values(self):
        for algo in ["auto", "full", "elkan"]:
            kmeans_model, labels = kmeans_clustering(self.data, 3, algorithm=algo)
            self.assertIsInstance(kmeans_model, KMeans)
            self.assertEqual(len(labels), len(self.data))

    def test_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            kmeans_clustering(self.data, 3, algorithm="invalid_algo")

    def test_custom_parameters(self):
        kmeans_model, labels = kmeans_clustering(
            self.data,
            3,
            random_state=42,
            n_init=20,
            max_iter=500,
            tol=1e-3,
            verbose=True,
        )
        self.assertEqual(kmeans_model.n_init, 20)
        self.assertEqual(kmeans_model.max_iter, 500)
        self.assertEqual(kmeans_model.tol, 1e-3)
        self.assertTrue(kmeans_model.verbose)

    def test_output_shape(self):
        kmeans_model, labels = kmeans_clustering(self.data, 3)
        self.assertEqual(labels.shape, (len(self.data),))


class TestAgglomerativeClustering(TestCase):
    def setUp(self):
        self.data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])
        self.empty_data = DataFrame(columns=["x", "y"])

    def test_valid_clustering(self):
        n_clusters = 3
        agg_model, labels = agglomerative_clustering(self.data, n_clusters)
        self.assertEqual(len(labels), len(self.data))
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), n_clusters)
        self.assertIsInstance(agg_model, AgglomerativeClustering)

    def test_empty_dataframe(self):
        with self.assertRaises(ArgumentStructureError):
            agglomerative_clustering(self.empty_data, 3)

    def test_invalid_n_clusters(self):
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 1)

        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 101)

    def test_invalid_n_clusters_type(self):
        with self.assertRaises(ArgumentTypeError):
            agglomerative_clustering(
                self.data, "three"
            )  # Passing a string instead of an int

    def test_ward_metric_error(self):
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 3, metric="manhattan", linkage="ward")

    def test_distance_threshold_error(self):
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(self.data, 3, distance_threshold=5.0)
        with self.assertRaises(ArgumentValueError):
            agglomerative_clustering(
                self.data, 3, distance_threshold=5.0, compute_full_tree=False
            )

    def test_output_shape(self):
        agg_model, labels = agglomerative_clustering(self.data, 3)
        self.assertEqual(labels.shape, (len(self.data),))

    def test_custom_parameters(self):
        agg_model, labels = agglomerative_clustering(
            self.data,
            3,
            metric="cosine",
            linkage="complete",
            compute_full_tree=True,
        )
        self.assertEqual(agg_model.metric, "cosine")
        self.assertEqual(agg_model.linkage, "complete")

    def test_precomputed_metric(self):
        distance_matrix = np.linalg.norm(
            self.data.values[:, np.newaxis] - self.data.values, axis=2
        )
        agg_model, labels = agglomerative_clustering(
            DataFrame(distance_matrix), 3, metric="precomputed", linkage="complete"
        )
        self.assertEqual(len(labels), len(self.data))


class TestClusteringNClusterSearch(TestCase):
    def setUp(self):
        self.data, _ = make_blobs(n_samples=100, centers=4, random_state=42)
        self.data = DataFrame(self.data, columns=["x", "y"])
        self.empty_data = DataFrame(columns=["x", "y"])

    def test_positive_case_kmeans(self):
        max_clusters = 5
        silhouette_scores, inertia_values = clustering_ncluster_search(
            self.data,
            max_clusters=max_clusters,
            clustering_method=kmeans_clustering,
            random_state=42,
            n_init=10,
        )
        self.assertEqual(len(silhouette_scores), max_clusters - 2)
        self.assertEqual(len(inertia_values), max_clusters - 2)

    def test_positive_case_agglomerative(self):
        max_clusters = 5
        silhouette_scores, inertia_values = clustering_ncluster_search(
            self.data,
            max_clusters=max_clusters,
            clustering_method=agglomerative_clustering,
            linkage="ward",
            metric="euclidean",
        )
        self.assertEqual(len(silhouette_scores), max_clusters - 2)
        self.assertIsNone(inertia_values)  # Inertia should be None for Agglomerative

    def test_empty_dataframe(self):
        with self.assertRaises(ArgumentStructureError):
            clustering_ncluster_search(
                self.empty_data, 10, clustering_method=kmeans_clustering
            )

        with self.assertRaises(ArgumentStructureError):
            clustering_ncluster_search(
                self.empty_data, 10, clustering_method=agglomerative_clustering
            )

    def test_invalid_max_clusters(self):
        with self.assertRaises(ArgumentValueError):
            clustering_ncluster_search(
                self.data, 1, clustering_method=kmeans_clustering
            )

        with self.assertRaises(ArgumentValueError):
            clustering_ncluster_search(
                self.data, 1, clustering_method=agglomerative_clustering
            )

    def test_invalid_clustering_method(self):
        with self.assertRaises(TypeError):
            clustering_ncluster_search(
                self.data, 10, clustering_method="invalid_method"
            )

    def test_valid_custom_metric(self):
        def custom_metric(data, labels):
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
        self.assertTrue(all(score == 0.5 for score in silhouette_scores))


if __name__ == "__main__":
    unittest.main()
