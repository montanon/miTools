import shutil
import unittest
from pathlib import Path
from unittest import TestCase

import networkx as nx
from networkx import Graph
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from mitools.economic_complexity import (
    build_mst_graph,
    build_mst_graphs,
    build_nx_graph,
    build_nx_graphs,
    check_if_dataframe_sequence,
    proximity_vectors_sequence,
    store_dataframe_sequence,
    vectors_from_proximity_matrix,
)
from mitools.exceptions import ArgumentValueError


class TestVectorsFromProximityMatrix(TestCase):
    def setUp(self):
        self.proximity_matrix = DataFrame(
            {
                "Product A": [0.0, 0.8, 0.4],
                "Product B": [0.8, 0.0, 0.5],
                "Product C": [0.4, 0.5, 0.0],
            },
            index=["Product A", "Product B", "Product C"],
        )

    def test_valid_conversion(self):
        result = vectors_from_proximity_matrix(self.proximity_matrix)
        expected = DataFrame(
            {
                "product_i": ["Product A", "Product B", "Product A"],
                "product_j": ["Product B", "Product C", "Product C"],
                "weight": [0.8, 0.5, 0.4],
            }
        )
        assert_frame_equal(result, expected)

    def test_valid_asymmetric_conversion(self):
        asymmetric_matrix = self.proximity_matrix.copy()
        asymmetric_matrix.loc["Product B", "Product A"] = 0.3
        result = vectors_from_proximity_matrix(
            asymmetric_matrix,
            sort_by=["product_i", "product_j"],
            sort_ascending=True,
        )
        expected = DataFrame(
            {
                "product_i": [
                    "Product A",
                    "Product A",
                    "Product B",
                    "Product B",
                    "Product C",
                    "Product C",
                ],
                "product_j": [
                    "Product B",
                    "Product C",
                    "Product A",
                    "Product C",
                    "Product A",
                    "Product B",
                ],
                "weight": [0.3, 0.4, 0.8, 0.5, 0.4, 0.5],
            }
        )
        assert_frame_equal(result, expected)

    def test_empty_matrix(self):
        empty_matrix = DataFrame(dtype=float)
        result = vectors_from_proximity_matrix(empty_matrix)
        expected = DataFrame(columns=["product_i", "product_j", "weight"], dtype=float)
        assert_frame_equal(result, expected, check_dtype=False)

    def test_valid_conversion_with_renames(self):
        result = vectors_from_proximity_matrix(
            self.proximity_matrix,
            orig_product="origin",
            dest_product="destination",
            proximity_column="proximity",
        )
        expected = DataFrame(
            {
                "origin": ["Product A", "Product B", "Product A"],
                "destination": ["Product B", "Product C", "Product C"],
                "proximity": [0.8, 0.5, 0.4],
            }
        )
        assert_frame_equal(result, expected)

    def test_invalid_sort_by(self):
        with self.assertRaises(ArgumentValueError):
            vectors_from_proximity_matrix(self.proximity_matrix, sort_by="invalid")

    def test_invalid_sort_ascending(self):
        with self.assertRaises(ArgumentValueError):
            vectors_from_proximity_matrix(
                self.proximity_matrix, sort_ascending="invalid"
            )


class TestProximityVectorsSequence(TestCase):
    def setUp(self):
        self.temp_dir = Path("./tests/.test_assets/.data")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.proximity_matrices = {
            1: DataFrame({"A": [0, 0.8], "B": [0.8, 0]}, index=["A", "B"]),
            2: DataFrame({"A": [0, 0.4], "B": [0.4, 0]}, index=["A", "B"]),
        }

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_existing_vectors(self):
        vectors = {
            key: vectors_from_proximity_matrix(matrix)
            for key, matrix in self.proximity_matrices.items()
        }
        store_dataframe_sequence(vectors, "proximity_vectors", self.temp_dir)
        result = proximity_vectors_sequence(
            self.proximity_matrices, data_dir=self.temp_dir, recalculate=False
        )
        for key, vector in vectors.items():
            assert_frame_equal(result[key], vector)

    def test_recalculate_vectors(self):
        result = proximity_vectors_sequence(
            self.proximity_matrices, data_dir=self.temp_dir, recalculate=True
        )
        for key, matrix in self.proximity_matrices.items():
            expected = vectors_from_proximity_matrix(matrix)
            assert_frame_equal(result[key], expected)

    def test_store_vectors_after_calculation(self):
        proximity_vectors_sequence(
            self.proximity_matrices, data_dir=self.temp_dir, recalculate=True
        )
        self.assertTrue(
            check_if_dataframe_sequence(
                self.temp_dir, "proximity_vectors", list(self.proximity_matrices.keys())
            )
        )


class TestBuildNxGraph(TestCase):
    def setUp(self):
        self.proximity_vectors = DataFrame(
            {
                "product_i": ["A", "A", "B"],
                "product_j": ["B", "C", "C"],
                "weight": [0.8, 0.4, 0.5],
            }
        )

    def test_valid_graph(self):
        G = build_nx_graph(self.proximity_vectors)
        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 3)
        self.assertAlmostEqual(G["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(G["A"]["C"]["weight"], 0.4)
        self.assertAlmostEqual(G["B"]["C"]["weight"], 0.5)

    def test_missing_column(self):
        with self.assertRaises(ArgumentValueError):
            invalid_vectors = self.proximity_vectors.drop(columns=["product_i"])
            build_nx_graph(invalid_vectors)

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=["product_i", "product_j", "weight"])
        G = build_nx_graph(empty_df)
        self.assertEqual(len(G.nodes), 0)
        self.assertEqual(len(G.edges), 0)

    def test_graph_with_additional_attributes(self):
        vectors_with_extra_attr = self.proximity_vectors.assign(year=[2020, 2021, 2022])
        G = build_nx_graph(vectors_with_extra_attr)
        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 3)
        self.assertEqual(G["A"]["B"]["year"], 2020)
        self.assertEqual(G["A"]["C"]["year"], 2021)
        self.assertEqual(G["B"]["C"]["year"], 2022)


class TestBuildNxGraphs(TestCase):
    def setUp(self):
        self.networks_folder = Path("./tests/.test_assets/.data")
        self.networks_folder.mkdir(parents=True, exist_ok=True)
        self.proximity_vectors = {
            1: DataFrame(
                {"product_i": ["A", "A"], "product_j": ["B", "C"], "weight": [0.8, 0.4]}
            ),
            2: DataFrame({"product_i": ["B"], "product_j": ["C"], "weight": [0.5]}),
        }

    def tearDown(self):
        if self.networks_folder.exists():
            shutil.rmtree(self.networks_folder)

    def test_build_and_store_graphs(self):
        graphs, graph_files = build_nx_graphs(
            self.proximity_vectors,
            orig_product="product_i",
            dest_product="product_j",
            networks_folder=self.networks_folder,
            recalculate=True,
        )
        for key, gml_path in graph_files.items():
            self.assertTrue(Path(gml_path).exists())
            self.assertTrue(isinstance(graphs[key], nx.Graph))
        self.assertEqual(len(graphs[1].nodes), 3)
        self.assertEqual(len(graphs[1].edges), 2)
        self.assertAlmostEqual(graphs[1]["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(graphs[1]["A"]["C"]["weight"], 0.4)
        self.assertEqual(len(graphs[2].nodes), 2)
        self.assertEqual(len(graphs[2].edges), 1)
        self.assertAlmostEqual(graphs[2]["B"]["C"]["weight"], 0.5)

    def test_load_existing_graphs(self):
        build_nx_graphs(
            self.proximity_vectors,
            orig_product="product_i",
            dest_product="product_j",
            networks_folder=self.networks_folder,
            recalculate=True,
        )
        graphs, graph_files = build_nx_graphs(
            self.proximity_vectors,
            orig_product="product_i",
            dest_product="product_j",
            networks_folder=self.networks_folder,
            recalculate=False,
        )
        for key, gml_path in graph_files.items():
            self.assertTrue(Path(gml_path).exists())
            self.assertTrue(isinstance(graphs[key], nx.Graph))
        self.assertEqual(len(graphs[1].nodes), 3)
        self.assertEqual(len(graphs[1].edges), 2)
        self.assertAlmostEqual(graphs[1]["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(graphs[1]["A"]["C"]["weight"], 0.4)
        self.assertEqual(len(graphs[2].nodes), 2)
        self.assertEqual(len(graphs[2].edges), 1)
        self.assertAlmostEqual(graphs[2]["B"]["C"]["weight"], 0.5)

    def test_missing_network_folder(self):
        with self.assertRaises(ArgumentValueError):
            build_nx_graphs(
                self.proximity_vectors,
                orig_product="product_i",
                dest_product="product_j",
                networks_folder="non_existent_folder",
                recalculate=False,
            )

    def test_empty_proximity_vectors(self):
        graphs, graph_files = build_nx_graphs(
            {},
            orig_product="product_i",
            dest_product="product_j",
            networks_folder=self.networks_folder,
            recalculate=True,
        )
        self.assertEqual(len(graphs), 0)
        self.assertEqual(len(graph_files), 0)


class TestBuildMSTGraph(TestCase):
    def setUp(self):
        self.proximity_vectors = DataFrame(
            {
                "product_i": ["A", "A", "B", "C"],
                "product_j": ["B", "C", "C", "D"],
                "weight": [0.8, 0.4, 0.5, 0.6],
            }
        )
        self.G = build_nx_graph(self.proximity_vectors)

    def test_mst_no_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors)
        self.assertEqual(len(mst.edges), 3)  # Expected 3 edges in the MST

    def test_mst_with_attribute_threshold(self):
        mst = build_mst_graph(self.proximity_vectors, attribute_th=0.5)
        self.assertEqual(len(mst.edges), 3)  # All edges with weight >= 0.5

    def test_mst_with_n_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors, n_extra_edges=1)
        self.assertEqual(len(mst.edges), 4)  # 3 MST edges + 1 extra edge

    def test_mst_with_pct_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors, pct_extra_edges=1.0)
        self.assertEqual(len(mst.edges), 4)  # 50% of remaining edges added

    def test_mst_with_all_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors, pct_extra_edges=1.0)
        self.assertEqual(len(mst.edges), 4)  # All edges added to the MST

    def test_missing_columns(self):
        invalid_vectors = self.proximity_vectors.drop(columns=["weight"])
        with self.assertRaises(ArgumentValueError):
            build_mst_graph(invalid_vectors)

    def test_empty_proximity_vectors(self):
        empty_vectors = DataFrame(columns=["product_i", "product_j", "weight"])
        mst = build_mst_graph(empty_vectors)
        self.assertEqual(len(mst.edges), 0)

    def test_preserve_original_weights(self):
        mst = build_mst_graph(self.proximity_vectors, n_extra_edges=1)
        self.assertAlmostEqual(mst["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(mst["C"]["D"]["weight"], 0.6)

    def test_graph_with_custom_attributes(self):
        vectors_with_attr = self.proximity_vectors.assign(year=[2020, 2021, 2022, 2023])
        mst = build_mst_graph(vectors_with_attr, attribute="year")
        self.assertTrue(all("year" in data for _, _, data in mst.edges(data=True)))


class TestBuildMSTGraphs(unittest.TestCase):
    def setUp(self):
        self.networks_folder = Path("./tests/.test_assets/.data")
        self.networks_folder.mkdir(parents=True, exist_ok=True)
        self.proximity_vectors = {
            1: DataFrame(
                {
                    "product_i": ["A", "A", "B", "C", "D"],
                    "product_j": ["B", "C", "C", "D", "A"],
                    "weight": [0.8, 0.4, 0.5, 0.6, 0.1],
                }
            ),
            2: DataFrame(
                {
                    "product_i": ["A", "A", "B", "C", "D"],
                    "product_j": ["B", "C", "C", "D", "A"],
                    "weight": [0.8, 0.4, 0.5, 0.6, 0.1],
                }
            ),
        }

    def tearDown(self):
        if self.networks_folder.exists():
            shutil.rmtree(self.networks_folder)

    def test_build_and_store_mst_graphs(self):
        graphs, graph_files = build_mst_graphs(
            self.proximity_vectors,
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            recalculate=True,
        )
        for key, gml_path in graph_files.items():
            self.assertTrue(Path(gml_path).exists())
            self.assertIsInstance(graphs[key], Graph)

    def test_load_existing_mst_graphs(self):
        build_mst_graphs(
            self.proximity_vectors,
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            recalculate=True,
        )
        graphs, graph_files = build_mst_graphs(
            self.proximity_vectors,
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            recalculate=False,
        )
        for key, graph in graphs.items():
            self.assertIsInstance(graph, Graph)

    def test_with_n_extra_edges(self):
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            n_extra_edges=1,
            recalculate=True,
        )
        for graph in graphs.values():
            self.assertEqual(len(graph.edges), 4)  # 2 MST edges + 1 extra edge

    def test_attribute_threshold(self):
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            attribute_th=0.4,
            recalculate=True,
        )
        for graph in graphs.values():
            for _, _, data in graph.edges(data=True):
                self.assertGreaterEqual(data["weight"], 0.4)

    def test_with_pct_extra_edges(self):
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            pct_extra_edges=0.0,
            recalculate=True,
        )
        for graph in graphs.values():
            self.assertEqual(len(graph.edges), 3)  # 0% of remaining edges added

    def test_missing_network_folder(self):
        with self.assertRaises(ArgumentValueError):
            build_mst_graphs(
                self.proximity_vectors,
                networks_folder="non_existent_folder",
                orig_product="product_i",
                dest_product="product_j",
                attribute="weight",
            )

    def test_empty_proximity_vectors(self):
        graphs, graph_files = build_mst_graphs(
            {},
            networks_folder=self.networks_folder,
            orig_product="product_i",
            dest_product="product_j",
            attribute="weight",
            recalculate=True,
        )
        self.assertEqual(len(graphs), 0)  # No graphs should be built
        self.assertEqual(len(graph_files), 0)


if __name__ == "__main__":
    unittest.main()
