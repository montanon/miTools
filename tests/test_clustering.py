from mitools.clustering import get_distances_to_centroids
from test_utils import generate_mock_dataframe
import unittest
from pandas import DataFrame


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
        empty_data = DataFrame(columns=['x', 'y', 'cluster']).set_index(['cluster', 'x'])
        distances = get_distances_to_centroids(empty_data, self.centroids, 'cluster')
        self.assertTrue(distances.empty)

    def test_negative_case(self):
        distances = get_distances_to_centroids(-1 * self.data, -1 * self.centroids, 'cluster')
        expected_distances = DataFrame([0.70710678, 0.70710678, 0.70710678, 0.70710678], index=[1, 2, 3, 4])
        self.assertTrue((distances.round(8) == expected_distances.round(8)).all().all())

if __name__ == '__main__':
    unittest.main()