from mitools.clustering import *
import unittest
from pandas import DataFrame, Series
from sklearn.metrics.pairwise import cosine_similarity

class TestGetDistancesToCentroids(unittest.TestCase):

    def setUp(self):
        # Mock data setup
        self.data = DataFrame({
            'x': [1, 2, 3, 4],
            'y': [1, 2, 3, 4],
            'cluster': [0, 0, 1, 1]
        }).set_index(['cluster', 'x'])

        self.centroids = DataFrame({
            'x': [1.5, 3.5],
            'y': [1.5, 3.5]
        }, index=[0, 1])

    def test_positive_case(self):
        distances = get_distances_to_centroids(self.data, self.centroids, 'cluster')
        expected_distances = DataFrame([0.70710678, 0.70710678, 0.70710678, 0.70710678], index=[1, 2, 3, 4])
        self.assertTrue((distances.round(8) == expected_distances.round(8)).all().all())

    def test_empty_dataframe(self):
        # Test to ensure function handles empty dataframe gracefully
        empty_data = DataFrame(columns=['x', 'y', 'cluster']).set_index(['cluster', 'x'])
        distances = get_distances_to_centroids(empty_data, self.centroids, 'cluster')
        self.assertTrue(distances.empty)

    def test_negative_case(self):
        distances = get_distances_to_centroids(-1 * self.data, -1 * self.centroids, 'cluster')
        expected_distances = DataFrame([0.70710678, 0.70710678, 0.70710678, 0.70710678], index=[1, 2, 3, 4])
        self.assertTrue((distances.round(8) == expected_distances.round(8)).all().all())


class TestGetCosineSimilarities(unittest.TestCase):

    def setUp(self):
        # Mock data setup
        self.data = DataFrame({
            'x': [1, 0, 0, 1],
            'y': [0, 1, 0, 0],
            'z': [0, 0, 1, 1],
            'cluster': [0, 0, 1, 1]
        }).set_index('cluster')

    def test_positive_case(self):
        result = get_cosine_similarities(self.data, 'cluster')
        # Verify for cluster 0
        expected_cluster_0 = cosine_similarity(self.data.loc[0])
        self.assertTrue((result[0] == expected_cluster_0).all())
        # Verify for cluster 1
        expected_cluster_1 = cosine_similarity(self.data.loc[1])
        self.assertTrue((result[1] == expected_cluster_1).all())

    def test_empty_dataframe(self):
        # Ensure function handles an empty dataframe without errors
        empty_data = DataFrame(columns=['x', 'y', 'z', 'cluster']).set_index('cluster')
        result = get_cosine_similarities(empty_data, 'cluster')
        self.assertTrue(empty_data.equals(result))

    def test_single_item_cluster(self):
        # Single item in a cluster should return a 1x1 DataFrame with value 1
        single_item_data = DataFrame({
            'x': [1],
            'y': [0],
            'z': [0],
            'cluster': [0]
        }).set_index('cluster')
        result = get_cosine_similarities(single_item_data, 'cluster')
        expected = Series([[[[1.0]]]], name='cluster').astype('object')
        self.assertTrue(result.equals(expected))


if __name__ == "__main__":
    unittest.main()