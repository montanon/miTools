import shutil
import unittest
from pathlib import Path
from unittest import TestCase

import networkx as nx
from networkx import DiGraph, Graph
from pandas import DataFrame, Interval
from pandas.testing import assert_frame_equal
from pyvis.network import Network as VisNetwork

from mitools.economic_complexity import (
    EdgesWidthsBins,
    NodesColors,
    NodesLabels,
    NodesSizes,
    assign_net_edges_attributes,
    assign_net_nodes_attributes,
    build_mst_graph,
    build_mst_graphs,
    build_nx_graph,
    build_nx_graphs,
    build_vis_graph,
    build_vis_graphs,
    check_if_dataframe_sequence,
    draw_nx_colored_graph,
    proximity_vectors_sequence,
    pyvis_to_networkx,
    store_dataframe_sequence,
    vectors_from_proximity_matrix,
)
from mitools.exceptions import ArgumentTypeError, ArgumentValueError


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


class TestBuildMSTGraphs(TestCase):
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


class TestAssignNetNodesAttributes(TestCase):
    def setUp(self):
        self.net = VisNetwork()
        self.net.add_node(1)
        self.net.add_node(2)
        self.net.add_node(3)

    def test_assign_valid_sizes(self):
        sizes = {1: 10, 2: 15, 3: 20}
        assign_net_nodes_attributes(self.net, sizes=sizes)
        for node in self.net.nodes:
            self.assertEqual(node["size"], sizes[node["id"]])

    def test_assign_single_size(self):
        assign_net_nodes_attributes(self.net, sizes=12)
        for node in self.net.nodes:
            self.assertEqual(node["size"], 12)

    def test_invalid_sizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, sizes="invalid_size")

    def test_missing_node_in_sizes(self):
        sizes = {1: 10, 2: 15}  # Missing size for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, sizes=sizes)

    def test_assign_valid_colors(self):
        colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        assign_net_nodes_attributes(self.net, colors=colors)
        for node in self.net.nodes:
            self.assertEqual(node["color"], colors[node["id"]])

    def test_invalid_colors_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, colors="invalid_color")

    def test_missing_node_in_colors(self):
        colors = {1: (255, 0, 0), 2: (0, 255, 0)}  # Missing color for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, colors=colors)

    def test_assign_valid_labels(self):
        labels = {1: "Node A", 2: "Node B", 3: "Node C"}
        assign_net_nodes_attributes(self.net, labels=labels)
        for node in self.net.nodes:
            self.assertEqual(node["label"], labels[node["id"]])

    def test_single_label_assignment(self):
        assign_net_nodes_attributes(self.net, labels="Common Label")
        for node in self.net.nodes:
            self.assertEqual(node["label"], "Common Label")

    def test_invalid_labels_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, labels=123)

    def test_missing_node_in_labels(self):
        labels = {1: "Node A", 2: "Node B"}  # Missing label for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, labels=labels)

    def test_assign_valid_label_sizes(self):
        label_sizes = {1: 15, 2: 20, 3: 25}
        assign_net_nodes_attributes(self.net, label_sizes=label_sizes)
        for node in self.net.nodes:
            self.assertEqual(node["font"], f"{label_sizes[node['id']]}px arial black")

    def test_single_label_size_assignment(self):
        assign_net_nodes_attributes(self.net, label_sizes=18)
        for node in self.net.nodes:
            self.assertEqual(node["font"], "18px arial black")

    def test_invalid_label_sizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, label_sizes="invalid_size")

    def test_missing_node_in_label_sizes(self):
        label_sizes = {1: 15, 2: 20}  # Missing size for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, label_sizes=label_sizes)


class TestAssignNetEdgesAttributes(TestCase):
    def setUp(self):
        self.net = VisNetwork()
        self.net.add_node(1)
        self.net.add_node(2)
        self.net.add_node(3)
        self.net.add_edge(1, 2, width=3.0)
        self.net.add_edge(2, 3, width=7.0)

        self.edges_widths = {
            Interval(0, 5, closed="both"): 2.0,
            Interval(5, 10, closed="both"): 5.0,
        }

    def test_assign_valid_edges_widths(self):
        assign_net_edges_attributes(self.net, self.edges_widths)
        expected_widths = [2.0, 5.0]
        for edge, expected in zip(self.net.edges, expected_widths):
            self.assertEqual(edge["width"], expected)

    def test_edge_width_not_in_bins(self):
        self.net.add_edge(3, 1, width=12.0)
        with self.assertRaises(ArgumentValueError):
            assign_net_edges_attributes(self.net, self.edges_widths)

    def test_empty_edges_widths(self):
        with self.assertRaises(ArgumentValueError):
            assign_net_edges_attributes(self.net, {})

    def test_no_matching_bins_for_edges(self):
        invalid_bins = {Interval(20, 30, closed="both"): 10.0}
        with self.assertRaises(ArgumentValueError):
            assign_net_edges_attributes(self.net, invalid_bins)

    def test_multiple_edges_with_same_width(self):
        self.net.add_edge(1, 3, width=3.0)  # Another edge with width 3.0
        assign_net_edges_attributes(self.net, self.edges_widths)
        for edge, val in zip(self.net.edges, [2.0, 5.0, 2.0]):
            self.assertEqual(edge["width"], val)


class TestBuildVisGraph(TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2, weight=3.0)
        self.graph.add_edge(2, 3, weight=7.0)
        self.nodes_sizes: NodesSizes = {1: 15, 2: 20, 3: 25}
        self.nodes_colors: NodesColors = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
        }
        self.nodes_labels: NodesLabels = {1: "A", 2: "B", 3: "C"}
        self.edges_widths: EdgesWidthsBins = {
            Interval(0, 5, closed="both"): 2.0,
            Interval(5, 10, closed="both"): 5.0,
        }

    def test_build_with_valid_attributes(self):
        net = build_vis_graph(
            graph=self.graph,
            nodes_sizes=self.nodes_sizes,
            nodes_colors=self.nodes_colors,
            nodes_labels=self.nodes_labels,
            edges_widths=self.edges_widths,
        )
        self.assertIsInstance(net, VisNetwork)
        self.assertEqual(len(net.nodes), 3)  # Ensure all nodes are added
        self.assertEqual(len(net.edges), 2)  # Ensure all edges are added

    def test_build_with_single_size(self):
        net = build_vis_graph(graph=self.graph, nodes_sizes=12)
        for node in net.nodes:
            self.assertEqual(node["size"], 12)

    def test_build_with_single_color(self):
        net = build_vis_graph(graph=self.graph, nodes_colors=(255, 0, 0))
        for node in net.nodes:
            self.assertEqual(node["color"], (255, 0, 0))

    def test_build_with_single_label(self):
        net = build_vis_graph(graph=self.graph, nodes_labels="Common Label")
        for node in net.nodes:
            self.assertEqual(node["label"], "Common Label")

    def test_invalid_nodes_sizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            build_vis_graph(graph=self.graph, nodes_sizes="invalid_size")

    def test_invalid_nodes_colors_type(self):
        with self.assertRaises(ArgumentTypeError):
            build_vis_graph(graph=self.graph, nodes_colors="invalid_color")

    def test_invalid_nodes_labels_type(self):
        with self.assertRaises(ArgumentTypeError):
            build_vis_graph(graph=self.graph, nodes_labels=123)

    def test_physics_settings(self):
        physics_kwargs = {
            "gravity": -5000,
            "spring_length": 300,
            "damping": 0.2,
        }
        net = build_vis_graph(
            graph=self.graph, physics=True, physics_kwargs=physics_kwargs
        )
        self.assertIsInstance(net, VisNetwork)  # Ensure a network is returned

    def test_empty_graph(self):
        empty_graph = nx.Graph()
        net = build_vis_graph(graph=empty_graph)
        self.assertEqual(len(net.nodes), 0)
        self.assertEqual(len(net.edges), 0)

    def test_missing_edges_widths(self):
        net = build_vis_graph(graph=self.graph)
        for edge in net.edges:
            self.assertIn("width", edge)  # Ensure width attribute exists

    def test_edge_width_not_in_bins(self):
        self.graph.add_edge(3, 4, weight=12.0)  # Add an edge with a weight not in bins
        with self.assertRaises(ArgumentValueError):
            build_vis_graph(graph=self.graph, edges_widths=self.edges_widths)

    def test_assign_label_sizes(self):
        label_sizes = {1: 15, 2: 20, 3: 25}
        net = build_vis_graph(graph=self.graph, node_label_size=label_sizes)
        for node in net.nodes:
            expected_size = label_sizes[node["id"]]
            self.assertEqual(node["font"], f"{expected_size}px arial black")

    def test_assign_single_label_size(self):
        net = build_vis_graph(graph=self.graph, node_label_size=18)
        for node in net.nodes:
            self.assertEqual(node["font"], "18px arial black")


class TestBuildVisGraphs(TestCase):
    def setUp(self):
        self.networks_folder = Path("./tests/.test_assets/.data")
        self.networks_folder.mkdir(parents=True, exist_ok=True)
        self.graphs_data = {
            1: nx.Graph([(1, 2, {"weight": 0.8}), (1, 3, {"weight": 0.4})]),
            2: nx.Graph([(2, 3, {"weight": 0.5})]),
        }

    def tearDown(self):
        if self.networks_folder.exists():
            shutil.rmtree(self.networks_folder)

    def test_build_and_store_vis_graphs(self):
        vis_graphs, graph_files = build_vis_graphs(
            self.graphs_data, networks_folder=self.networks_folder
        )
        for key, html_path in graph_files.items():
            self.assertTrue(Path(html_path).exists())
            self.assertIsInstance(vis_graphs[key], VisNetwork)
        self.assertEqual(len(vis_graphs[1].nodes), 3)
        self.assertEqual(len(vis_graphs[1].edges), 2)
        self.assertEqual(len(vis_graphs[2].nodes), 2)
        self.assertEqual(len(vis_graphs[2].edges), 1)

    def test_missing_network_folder(self):
        with self.assertRaises(ArgumentValueError):
            build_vis_graphs(self.graphs_data, networks_folder="non_existent_folder")

    def test_empty_graph_data(self):
        vis_graphs, graph_files = build_vis_graphs(
            {},
            networks_folder=self.networks_folder,
        )
        self.assertEqual(len(vis_graphs), 0)
        self.assertEqual(len(graph_files), 0)

    def test_build_with_custom_node_sizes(self):
        node_sizes = {1: 10, 2: 15, 3: 20}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            networks_folder=self.networks_folder,
            nodes_sizes=node_sizes,
        )
        for node in vis_graphs[1].nodes:
            self.assertEqual(node["size"], node_sizes[node["id"]])

    def test_build_with_custom_node_colors(self):
        node_colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            networks_folder=self.networks_folder,
            nodes_colors=node_colors,
        )
        for node in vis_graphs[1].nodes:
            self.assertEqual(node["color"], node_colors[node["id"]])

    def test_build_with_custom_labels(self):
        node_labels = {1: "A", 2: "B", 3: "C"}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            networks_folder=self.networks_folder,
            nodes_labels=node_labels,
        )
        for node in vis_graphs[1].nodes:
            self.assertEqual(node["label"], node_labels[node["id"]])

    def test_build_with_custom_physics(self):
        physics_kwargs = {"gravity": -5000, "spring_length": 300}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            networks_folder=self.networks_folder,
            physics=True,
            physics_kwargs=physics_kwargs,
        )
        self.assertIsInstance(vis_graphs[1], VisNetwork)


class TestPyvisToNetworkx(TestCase):
    def setUp(self):
        self.undirected_network = VisNetwork()
        self.undirected_network.add_node(1, label="Node A", size=10)
        self.undirected_network.add_node(2, label="Node B", size=15)
        self.undirected_network.add_edge(1, 2, width=2.0, title="Edge A-B")

        self.directed_network = VisNetwork(directed=True)
        self.directed_network.add_node(3, label="Node C", size=20)
        self.directed_network.add_node(4, label="Node D", size=25)
        self.directed_network.add_edge(3, 4, width=3.0, title="Edge C-D")

    def test_convert_undirected_network(self):
        nx_graph = pyvis_to_networkx(self.undirected_network)
        self.assertIsInstance(nx_graph, Graph)
        self.assertEqual(len(nx_graph.nodes), 2)
        self.assertEqual(len(nx_graph.edges), 1)
        self.assertEqual(nx_graph.nodes[1]["name"], "Node A")
        self.assertAlmostEqual(nx_graph[1][2]["weight"], 2.0)
        self.assertEqual(nx_graph[1][2]["title"], "Edge A-B")

    def test_convert_directed_network(self):
        nx_graph = pyvis_to_networkx(self.directed_network)
        self.assertIsInstance(nx_graph, DiGraph)
        self.assertEqual(len(nx_graph.nodes), 2)
        self.assertEqual(len(nx_graph.edges), 1)
        self.assertEqual(nx_graph.nodes[3]["name"], "Node C")
        self.assertAlmostEqual(nx_graph[3][4]["weight"], 3.0)
        self.assertEqual(nx_graph[3][4]["title"], "Edge C-D")

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            pyvis_to_networkx([1, 2, 3])

    def test_default_edge_weight(self):
        self.undirected_network.add_edge(2, 1)
        nx_graph = pyvis_to_networkx(self.undirected_network)
        self.assertAlmostEqual(nx_graph[2][1]["weight"], 2.0)


class TestDrawNxColoredGraph(unittest.TestCase):
    def setUp(self):
        self.G = Graph()
        self.G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        self.pos_G = nx.spring_layout(self.G)
        self.node_colors = {"red": [1, 2], "blue": [3, 4]}
        self.edge_widths = {
            2.0: [(1, 2), (3, 4)],
            3.0: [(2, 3)],
        }

    def test_valid_input(self):
        draw_nx_colored_graph(self.G, self.pos_G, self.node_colors, self.edge_widths)

    def test_invalid_graph_type(self):
        with self.assertRaises(ArgumentTypeError):
            draw_nx_colored_graph(
                "not_a_graph", self.pos_G, self.node_colors, self.edge_widths
            )

    def test_invalid_pos_G_type(self):
        with self.assertRaises(ArgumentTypeError):
            draw_nx_colored_graph(
                self.G, "not_a_dict", self.node_colors, self.edge_widths
            )

    def test_missing_nodes(self):
        self.node_colors["red"].append(5)  # Node 5 is not in the graph
        with self.assertRaises(ArgumentValueError):
            draw_nx_colored_graph(
                self.G, self.pos_G, self.node_colors, self.edge_widths
            )

    def test_missing_edges(self):
        self.edge_widths[2.0].append((5, 6))  # Edge (5, 6) is not in the graph
        with self.assertRaises(ArgumentValueError):
            draw_nx_colored_graph(
                self.G, self.pos_G, self.node_colors, self.edge_widths
            )

    def test_custom_node_size(self):
        draw_nx_colored_graph(
            self.G, self.pos_G, self.node_colors, self.edge_widths, node_size=20
        )

    def test_custom_width_scale(self):
        draw_nx_colored_graph(
            self.G, self.pos_G, self.node_colors, self.edge_widths, width_scale=5.0
        )

    def test_edge_alpha(self):
        draw_nx_colored_graph(
            self.G, self.pos_G, self.node_colors, self.edge_widths, edge_alpha=0.5
        )


if __name__ == "__main__":
    unittest.main()
