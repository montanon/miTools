from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from networkx import DiGraph, Graph
from pandas import DataFrame, Interval
from pyvis.network import Network as VisNetwork

from mitools.economic_complexity import (
    check_if_dataframe_sequence,
    load_dataframe_sequence,
    store_dataframe_sequence,
)
from mitools.exceptions import ArgumentTypeError, ArgumentValueError

NodeID = Any
NodeColor = Union[Tuple[float, ...], Tuple[int, ...]]
NodesColors = Dict[NodeID, NodeColor]
NodesLabels = Dict[NodeID, str]
NodesSizes = Dict[NodeID, Union[int, float]]
EdgesWidthsBins = Dict[Interval, float]


def vectors_from_proximity_matrix(
    proximity_matrix: DataFrame,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    proximity_column: str = "weight",
    sort_by: Union[str, List[str], Tuple[str]] = None,
    sort_ascending: Union[bool, List[bool], Tuple[bool]] = False,
) -> DataFrame:
    if sort_by is not None:
        if isinstance(sort_by, str) and sort_by not in [
            orig_product,
            dest_product,
            proximity_column,
        ]:
            raise ArgumentValueError(
                f"Column '{sort_by}' not available in output DataFrame."
            )
        elif isinstance(sort_by, (list, tuple)) and not all(
            [
                col
                in [
                    orig_product,
                    dest_product,
                    proximity_column,
                ]
                for col in sort_by
            ]
        ):
            raise ArgumentValueError(
                f"Columns '{sort_by}' not available in output DataFrame."
            )
    if sort_ascending is not None:
        if not isinstance(sort_ascending, bool) or (
            isinstance(sort_ascending, list)
            and all(isinstance(b, bool) for b in sort_ascending)
        ):
            raise ArgumentValueError(
                "sort_ascending must be a boolean or a list of booleans."
            )
    is_symmetric = proximity_matrix.equals(proximity_matrix.T)
    proximity_vectors = proximity_matrix.unstack().reset_index()
    proximity_vectors.columns = [orig_product, dest_product, proximity_column]
    if is_symmetric:
        proximity_vectors = proximity_vectors[
            proximity_vectors[orig_product] <= proximity_vectors[dest_product]
        ]
    proximity_vectors = proximity_vectors.loc[proximity_vectors[proximity_column] > 0]
    proximity_vectors = proximity_vectors.drop_duplicates()
    proximity_vectors = proximity_vectors.sort_values(
        by=proximity_column if sort_by is None else sort_by, ascending=sort_ascending
    ).reset_index(drop=True)
    proximity_vectors = proximity_vectors.rename(
        columns={proximity_column: proximity_column}
    )
    proximity_vectors[orig_product] = proximity_vectors[orig_product].astype(str)
    proximity_vectors[dest_product] = proximity_vectors[dest_product].astype(str)

    return proximity_vectors


def proximity_vectors_sequence(
    proximity_matrices: Dict[Union[str, int], DataFrame],
    data_dir: PathLike = None,
    recalculate: bool = False,
    sequence_name: str = "proximity_vectors",
) -> Dict[Union[str, int], DataFrame]:
    sequence_values = list(proximity_matrices.keys())
    if (
        not recalculate
        and data_dir is not None
        and check_if_dataframe_sequence(
            data_dir=data_dir, name=sequence_name, sequence_values=sequence_values
        )
    ):
        proximity_vectors = load_dataframe_sequence(
            data_dir=data_dir, name=sequence_name, sequence_values=sequence_values
        )
    else:
        proximity_vectors = {
            key: vectors_from_proximity_matrix(proximity_matrix)
            for key, proximity_matrix in proximity_matrices.items()
        }
        if data_dir is not None:
            store_dataframe_sequence(
                proximity_vectors, data_dir=data_dir, name=sequence_name
            )
    return proximity_vectors


def build_nx_graph(
    proximity_vectors: DataFrame,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
) -> Graph:
    required_columns = {orig_product, dest_product}
    if not required_columns.issubset(proximity_vectors.columns):
        missing_cols = required_columns - set(proximity_vectors.columns)
        raise ArgumentValueError(f"Missing columns in DataFrame: {missing_cols}")
    G = nx.from_pandas_edgelist(
        proximity_vectors, source=orig_product, target=dest_product, edge_attr=True
    )

    return G


def build_nx_graphs(
    proximity_vectors: Dict[Union[str, int], DataFrame],
    networks_folder: PathLike,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    recalculate: bool = False,
) -> Tuple[Dict[Union[str, int], Graph], Dict[Union[str, int], Path]]:
    networks_folder = Path(networks_folder)
    if not networks_folder.exists():
        raise ArgumentValueError(f"Folder '{networks_folder}' does not exist.")
    graphs = {}
    graph_files = {}
    for key, vectors in proximity_vectors.items():
        gml_name = f"{key}_G_graph.gml".replace(" ", "_")
        gml_path = networks_folder / gml_name
        if not gml_path.exists() or recalculate:
            G = build_nx_graph(
                vectors, orig_product=orig_product, dest_product=dest_product
            )
            nx.write_gml(G, gml_path)  # Store the graph in GML format
        else:
            G = nx.read_gml(gml_path)  # Load the graph from disk
        graphs[key] = G
        graph_files[key] = str(gml_path)

    return graphs, graph_files


def build_mst_graph(
    proximity_vectors: DataFrame,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    attribute: str = "weight",
    attribute_th: float = None,
    n_extra_edges: int = None,
    pct_extra_edges: float = None,
) -> Graph:
    required_columns = {orig_product, dest_product, attribute}
    if not required_columns.issubset(proximity_vectors.columns):
        missing_cols = required_columns - set(proximity_vectors.columns)
        raise (f"Missing columns in DataFrame: {missing_cols}")
    sorted_vectors = proximity_vectors.sort_values(by=attribute, ascending=False)
    G = build_nx_graph(
        sorted_vectors, orig_product=orig_product, dest_product=dest_product
    )
    MST = nx.maximum_spanning_tree(G, weight=attribute)
    extra_edges = None
    if attribute_th is not None:
        extra_edges = sorted_vectors.query(f"{attribute} >= @attribute_th")
    elif n_extra_edges is not None:
        n_total_edges = len(MST.edges) + n_extra_edges
        extra_edges = sorted_vectors.iloc[:n_total_edges]
    elif pct_extra_edges is not None:
        n_total_edges = int(
            (sorted_vectors.shape[0] - len(MST.edges)) * pct_extra_edges
        )
        extra_edges = sorted_vectors.iloc[
            len(MST.edges) : len(MST.edges) + n_total_edges
        ]
    if extra_edges is not None:
        extra_graph = build_nx_graph(
            extra_edges, orig_product=orig_product, dest_product=dest_product
        )
        combined_graph = nx.compose(MST, extra_graph)
        for u, v, data in G.edges(data=True):
            if combined_graph.has_edge(u, v):
                combined_graph[u][v][attribute] = data[attribute]
        MST = combined_graph
    return MST


def build_mst_graphs(
    proximity_vectors: Dict[Union[str, int], DataFrame],
    networks_folder: PathLike,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    attribute: str = "weight",
    attribute_th: float = None,
    n_extra_edges: int = None,
    pct_extra_edges: float = None,
    recalculate: bool = False,
) -> Tuple[Dict[Union[str, int], Graph], Dict[Union[str, int], Path]]:
    networks_folder = Path(networks_folder)
    if not networks_folder.exists():
        raise ArgumentValueError(f"Folder '{networks_folder}' does not exist.")
    graphs = {}
    graph_files = {}
    for key, vectors in proximity_vectors.items():
        gml_name = f"{key}_MST_graph.gml".replace(" ", "_")
        gml_path = networks_folder / gml_name
        if not gml_path.exists() or recalculate:
            MST = build_mst_graph(
                vectors,
                orig_product=orig_product,
                dest_product=dest_product,
                attribute=attribute,
                attribute_th=attribute_th,
                n_extra_edges=n_extra_edges,
                pct_extra_edges=pct_extra_edges,
            )
            nx.write_gml(MST, gml_path)
        else:
            MST = nx.read_gml(gml_path)
        graphs[key] = MST
        graph_files[key] = str(gml_path)

    return graphs, graph_files


def build_vis_graph(
    graph: Graph,
    nodes_sizes: Union[NodesSizes, int, float] = None,
    nodes_colors: Union[NodesColors, NodeColor] = None,
    nodes_labels: Union[NodesLabels, str] = None,
    node_label_size: Union[Dict[NodeID, int], int] = None,
    edges_widths: EdgesWidthsBins = None,
    net_height: int = 700,
    notebook: bool = True,
    physics: bool = False,
    physics_kwargs: Dict[str, Any] = None,
) -> VisNetwork:
    if physics_kwargs is None:
        physics_kwargs = {
            "gravity": -1000000,
            "central_gravity": 0.0,
            "spring_length": 500,
            "spring_strength": 2,
            "damping": 0.1,
            "overlap": 1,
        }
    net = VisNetwork(height=f"{net_height}px", notebook=notebook)
    net.from_nx(graph)

    assign_net_nodes_attributes(
        net=net,
        sizes=nodes_sizes,
        colors=nodes_colors,
        labels=nodes_labels,
        label_sizes=node_label_size,
    )

    assign_net_edges_attributes(net=net, edges_widths=edges_widths)

    net.barnes_hut(**physics_kwargs)
    if physics:
        net.show_buttons(filter_=["physics"])
    return net


def assign_net_edges_attributes(net: VisNetwork, edges_widths: EdgesWidthsBins):
    if edges_widths is not None:
        for edge in net.edges:
            try:
                edge["width"] = next(
                    w for b, w in edges_widths.items() if edge["width"] in b
                )
            except StopIteration:
                raise ArgumentValueError(
                    "Some edge width values are not present in the corresponding 'edges_widths' argument."
                )


def assign_net_nodes_attributes(
    net: VisNetwork,
    sizes: Union[NodesSizes, int, float] = None,
    colors: Union[NodesColors, NodeColor] = None,
    labels: Union[NodesLabels, str] = None,
    label_sizes: Union[Dict[NodeID, int], int] = None,
):
    if sizes is not None and not isinstance(sizes, (int, float, dict)):
        raise ArgumentTypeError("Nodes 'sizes' must be a int, float or dict.")
    if isinstance(sizes, dict) and not all(node["id"] in sizes for node in net.nodes):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'sizes' argument."
        )
    if colors is not None and not isinstance(colors, (tuple, list, dict)):
        raise ArgumentTypeError(
            "Nodes 'colors' must be a tuple, list, NodeColor or dict."
        )
    if isinstance(colors, dict) and not all(node["id"] in colors for node in net.nodes):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'colors' argument."
        )
    if labels is not None and not isinstance(labels, (str, dict)):
        raise ArgumentTypeError("Nodes 'labels' must be a str or dict.")
    if isinstance(labels, dict) and not all(node["id"] in labels for node in net.nodes):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'labels' argument."
        )
    if label_sizes is not None and not isinstance(label_sizes, (int, dict)):
        raise ArgumentTypeError("Nodes 'label_sizes' must be a int or dict.")
    if isinstance(label_sizes, dict) and not all(
        node["id"] in label_sizes for node in net.nodes
    ):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'label_sizes' argument."
        )
    if sizes is not None:
        for node in net.nodes:
            node["size"] = sizes if not isinstance(sizes, dict) else sizes[node["id"]]
    if colors is not None:
        for node in net.nodes:
            node["color"] = (
                colors if not isinstance(colors, dict) else colors[node["id"]]
            )
    if labels is not None:
        for node in net.nodes:
            node["label"] = (
                labels if not isinstance(labels, dict) else labels[node["id"]]
            )
    if label_sizes is not None:
        for node in net.nodes:
            node["font"] = (
                f"{label_sizes}px arial black"
                if not isinstance(label_sizes, dict)
                else f"{label_sizes[node['id']]}px arial black"
            )


def build_vis_graphs(
    graphs_data: Dict[Union[str, int], Graph],
    networks_folder: PathLike,
    nodes_sizes: Union[NodesSizes, int, float] = None,
    nodes_colors: Union[NodesColors, NodeColor] = None,
    nodes_labels: Union[NodesLabels, str] = None,
    node_label_size: Union[Dict[NodeID, int], int] = None,
    edges_widths: EdgesWidthsBins = None,
    net_height: int = 700,
    notebook: bool = True,
    physics: bool = False,
    physics_kwargs: Dict[str, Any] = None,
) -> Tuple[Dict[Union[str, int], VisNetwork], Dict[Union[str, int], str]]:
    networks_folder = Path(networks_folder)
    if not networks_folder.exists():
        raise ArgumentValueError(f"Folder '{networks_folder}' does not exist.")

    vis_graphs = {}
    graph_files = {}

    for key, graph in graphs_data.items():
        gml_name = f"{key}_vis_graph.html".replace(" ", "_")
        gml_path = networks_folder / gml_name

        if not gml_path.exists():
            net = build_vis_graph(
                graph=graph,
                nodes_sizes=nodes_sizes,
                nodes_colors=nodes_colors,
                nodes_labels=nodes_labels,
                node_label_size=node_label_size,
                edges_widths=edges_widths,
                net_height=net_height,
                notebook=notebook,
                physics=physics,
                physics_kwargs=physics_kwargs,
            )
            net.save_graph(str(gml_path))  # Save the graph as an HTML file

        vis_graphs[key] = net
        graph_files[key] = str(gml_path)

    return vis_graphs, graph_files


def distribute_products_in_communities(series, n_communities):
    values = series.tolist()
    np.random.shuffle(values)
    size = len(values) // n_communities
    remainder = len(values) % n_communities

    communities = []
    start = 0

    for i in range(n_communities):
        end = start + size + (1 if i < remainder else 0)
        communities.append(values[start:end])
        start = end

    return communities


def average_strenght_of_links_within_community(G, community):
    links = G.edges(community, data=True)
    try:
        return np.mean([d["width"] for u, v, d in links if v in community])
    except Exception:
        return np.mean([d["weight"] for u, v, d in links if v in community])


def average_strenght_of_links_within_communities(G, communities):
    strenghts = [
        average_strenght_of_links_within_community(G, community)
        for community in communities
    ]
    strenghts = [s for s in strenghts if not np.isnan(s)]
    return {
        "mean": np.mean(strenghts),
        "std": np.std(strenghts),
        "max": np.max(strenghts),
        "min": np.min(strenghts),
    }


def average_strength_of_links_from_community(G, community):
    links = G.edges(data=True)
    try:
        return np.mean(
            [d["width"] for u, v, d in links if u in community and v not in community]
        )
    except Exception:
        return np.mean(
            [d["weight"] for u, v, d in links if u in community and v not in community]
        )


def average_strength_of_links_from_communities(G, communities):
    strenghts = [
        average_strength_of_links_from_community(G, community)
        for community in communities
    ]
    strenghts = [s for s in strenghts if not np.isnan(s)]
    return {
        "mean": np.mean(strenghts),
        "std": np.std(strenghts),
        "max": np.max(strenghts),
        "min": np.min(strenghts),
    }


def create_nx_nodes_color_dict(G, color_bins, sitc_codes):
    node_colors = {}

    for node in G.nodes:
        sitc_id = sitc_codes.loc[
            sitc_codes["sitc_product_code"] == node, "sitc_product_code"
        ].values[0]
        if not sitc_id.isdigit():
            node_colors[node] = "#000000"
        else:
            for b, c in color_bins.items():
                if int(sitc_id) in b:
                    node_colors[node] = c
                    continue

    return node_colors


def create_nx_edges_widths(G, proximity_vectors, bin_widths):
    edge_widths = {w: [] for w in bin_widths.values()}

    for edge in G.edges:
        prox = proximity_vectors.loc[
            (proximity_vectors["product_i"].isin(edge))
            & (proximity_vectors["product_j"].isin(edge)),
            "proximity",
        ].values[0]
        for b, w in bin_widths.items():
            if prox in b:
                edge_widths[w].append(edge)
                continue


def draw_nx_colored_graph(G, pos_G, node_colors, edge_widths):
    for c, nodes in node_colors.items():
        nx.draw_networkx_nodes(G, pos_G, node_size=10, nodelist=nodes, node_color=c)
    for w, nodes in edge_widths.items():
        nx.draw_networkx_edges(G, pos_G, alpha=1, nodelist=nodes, width=w / 10)


def display_country_nodes(net, country, masked_rca_matrix, export_matrix, bins=3):
    _net = VisNetwork(height="700px", notebook=True)

    net_data = net.get_network_data()

    for node in net_data[0]:
        _net.add_node(node["id"], **node)

    for edge in net_data[1]:
        _net.add_edge(
            edge["from"],
            edge["to"],
            **{k: v for k, v in edge.items() if k not in ["from", "to"]},
        )

    _net.set_options(net_data[5])

    country_rcas = masked_rca_matrix.loc[[country]]
    country_exports = export_matrix.loc[[country]]

    rca_nodes = country_rcas.loc[:, (country_rcas == 1.0).any()].columns.tolist()
    rca_nodes_exports = country_exports.loc[:, rca_nodes].T
    rca_nodes_exports["bins"] = pd.cut(
        rca_nodes_exports.values[:, 0], bins, labels=[n + 1 for n in range(bins)]
    )
    rca_nodes_exports = {n: e for n, e in zip(rca_nodes, rca_nodes_exports["bins"])}

    for node in _net.nodes:
        if isinstance(node["id"], str) and node["id"].isdigit():
            not_in_nodes = int(node["id"]) not in rca_nodes
        else:
            not_in_nodes = node["id"] not in rca_nodes
        if not_in_nodes:
            sector_color = node["color"]
            node["color"] = {"border": sector_color, "background": "#cccccc"}
            node["borderWidth"] = 1
        else:
            node["shape"] = "square"
            try:
                node["size"] = node["size"] * rca_nodes_exports[int(node["id"])]
            except Exception:
                node["size"] = node["size"] * rca_nodes_exports[node["id"]]

    return _net


def pyvis_to_networkx(pyvis_network):
    nx_graph = DiGraph() if pyvis_network.directed else Graph()

    for node in pyvis_network.nodes:
        node_id = node["id"]
        node_attrs = {k: v for k, v in node.items() if k != "id"}
        node_attrs["name"] = node_attrs["label"]
        nx_graph.add_node(node_id, **node_attrs)

    for edge in pyvis_network.edges:
        source = edge["from"]
        target = edge["to"]
        edge_attrs = {"weight": edge["width"]}
        if "title" in edge:
            edge_attrs["title"] = edge["title"]
        nx_graph.add_edge(source, target, **edge_attrs)

    return nx_graph
