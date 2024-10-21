from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from networkx import Graph
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


# def width_bins():
#     width_bins = pd.cut(
#         proximity_vectors[edge_attribute].sort_values(), bins=len(widths), precision=0
#     ).unique()
#     width_bins = {b: w for b, w in zip(width_bins, widths)}

#     if products_codes is not None and color_bins is not None:
#         for node in net.nodes:
#             try:
#                 product_id = int(node["id"])
#                 product_code = products_codes.loc[
#                     products_codes[product_code_col] == product_id, product_code_col
#                 ].values[0]

#                 node["color"] = next(
#                     c for b, c in color_bins.items() if int(product_code) in b
#                 )
#             except (IndexError, StopIteration):
#                 node["color"] = "gray"  # Default color


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


def build_vis_graphs(
    mst_graphs,
    proximity_vectors_dfs,
    id_col,
    product_name_col,
    products_codes,
    physics=False,
    node_size=10,
    label_size=20,
):
    nets = {}

    color_bins, bins_names = build_color_bins(id_col)

    for (temp_id, MST), (_, proximity_vectors) in zip(
        mst_graphs.items(), proximity_vectors_dfs.items()
    ):
        net = build_vis_graph(
            MST,
            proximity_vectors,
            product_name_col,
            id_col,
            products_codes,
            color_bins,
            physics=physics,
            node_size=node_size,
            node_label_size=label_size,
        )

        nets[temp_id] = net

    return nets


def build_color_bins(id_col):
    bins = _bins_colors.get(id_col, None)

    if bins is not None:
        return bins["color"], bins["names"]
    else:
        return None, None


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
    nx_graph = nx.DiGraph() if pyvis_network.directed else Graph()

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


_bins_colors = {
    "HS2 ID": {
        "color": {
            pd.Interval(left=101, right=105, closed="right"): "#F57373",
            pd.Interval(left=206, right=214, closed="right"): "#AAD75A",
            pd.Interval(left=315, right=315, closed="right"): "#D7B55A",
            pd.Interval(left=416, right=424, closed="right"): "#FAF51B",
            pd.Interval(left=525, right=527, closed="right"): "#643F0D",
            pd.Interval(left=628, right=638, closed="right"): "#EA14B0",
            pd.Interval(left=739, right=740, closed="right"): "#854CEB",
            pd.Interval(left=841, right=843, closed="right"): "#B87B7B",
            pd.Interval(left=944, right=946, closed="right"): "#204F0B",
            pd.Interval(left=1047, right=1049, closed="right"): "#F8F0A4",
            pd.Interval(left=1150, right=1163, closed="right"): "#00CC1E",
            pd.Interval(left=1264, right=1267, closed="right"): "#00FF25",
            pd.Interval(left=1368, right=1370, closed="right"): "#AB9E97",
            pd.Interval(left=1471, right=1471, closed="right"): "#76E5E7",
            pd.Interval(left=1572, right=1583, closed="right"): "#4E6D6E",
            pd.Interval(left=1684, right=1685, closed="right"): "#314480",
            pd.Interval(left=1786, right=1789, closed="right"): "#85F6FF",
            pd.Interval(left=1890, right=1892, closed="right"): "#3F004C",
            pd.Interval(left=1993, right=1993, closed="right"): "#B3B3B3",
            pd.Interval(left=2094, right=2096, closed="right"): "#614906",
            pd.Interval(left=2197, right=2197, closed="right"): "#B584C4",
            pd.Interval(left=2299, right=2299, closed="right"): "#000000",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
        ],
    },
    "HS4 ID": {
        "color": {
            pd.Interval(left=10101, right=10511, closed="right"): "#F57373",
            pd.Interval(left=20601, right=21404, closed="right"): "#AAD75A",
            pd.Interval(left=31501, right=31522, closed="right"): "#D7B55A",
            pd.Interval(left=41601, right=42403, closed="right"): "#FAF51B",
            pd.Interval(left=52501, right=52716, closed="right"): "#643F0D",
            pd.Interval(left=62801, right=63826, closed="right"): "#EA14B0",
            pd.Interval(left=73901, right=74017, closed="right"): "#854CEB",
            pd.Interval(left=84101, right=84304, closed="right"): "#B87B7B",
            pd.Interval(left=94401, right=94602, closed="right"): "#204F0B",
            pd.Interval(left=104701, right=104911, closed="right"): "#F8F0A4",
            pd.Interval(left=115001, right=116310, closed="right"): "#00CC1E",
            pd.Interval(left=126401, right=126703, closed="right"): "#00FF25",
            pd.Interval(left=136801, right=137020, closed="right"): "#AB9E97",
            pd.Interval(left=147101, right=147118, closed="right"): "#76E5E7",
            pd.Interval(left=157201, right=158311, closed="right"): "#4E6D6E",
            pd.Interval(left=168401, right=168548, closed="right"): "#314480",
            pd.Interval(left=178601, right=178908, closed="right"): "#85F6FF",
            pd.Interval(left=189001, right=189209, closed="right"): "#3F004C",
            pd.Interval(left=199301, right=199307, closed="right"): "#B3B3B3",
            pd.Interval(left=209401, right=209619, closed="right"): "#614906",
            pd.Interval(left=219701, right=219706, closed="right"): "#B584C4",
            pd.Interval(left=229901, right=229999, closed="right"): "#000000",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
        ],
    },
    "HS6 ID": {
        "color": {
            pd.Interval(left=1010100, right=1051199, closed="right"): "#F57373",
            pd.Interval(left=2060100, right=2140499, closed="right"): "#AAD75A",
            pd.Interval(left=3150100, right=3152299, closed="right"): "#D7B55A",
            pd.Interval(left=4160100, right=4240399, closed="right"): "#FAF51B",
            pd.Interval(left=5250100, right=5271699, closed="right"): "#643F0D",
            pd.Interval(left=6280100, right=6382699, closed="right"): "#EA14B0",
            pd.Interval(left=7390100, right=7401799, closed="right"): "#854CEB",
            pd.Interval(left=8410100, right=8430499, closed="right"): "#B87B7B",
            pd.Interval(left=9440100, right=9460299, closed="right"): "#204F0B",
            pd.Interval(left=10470100, right=10491199, closed="right"): "#F8F0A4",
            pd.Interval(left=11500100, right=11631099, closed="right"): "#00CC1E",
            pd.Interval(left=12640100, right=12670399, closed="right"): "#00FF25",
            pd.Interval(left=13680100, right=13702099, closed="right"): "#AB9E97",
            pd.Interval(left=14710100, right=14711899, closed="right"): "#76E5E7",
            pd.Interval(left=15720100, right=15831199, closed="right"): "#4E6D6E",
            pd.Interval(left=16840100, right=16854899, closed="right"): "#314480",
            pd.Interval(left=17860100, right=17890899, closed="right"): "#85F6FF",
            pd.Interval(left=18900100, right=18920999, closed="right"): "#3F004C",
            pd.Interval(left=19930100, right=19930799, closed="right"): "#B3B3B3",
            pd.Interval(left=20940100, right=20961999, closed="right"): "#614906",
            pd.Interval(left=21970100, right=21970699, closed="right"): "#B584C4",
            pd.Interval(left=22990100, right=22999999, closed="right"): "#000000",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
        ],
    },
    "sitc_product_code": {
        "color": {
            pd.Interval(left=0, right=999, closed="right"): "#65D100",
            pd.Interval(left=1000, right=1999, closed="right"): "#FFE400",
            pd.Interval(left=2000, right=2999, closed="right"): "#DE1313",
            pd.Interval(left=3000, right=3999, closed="right"): "#89422F",
            pd.Interval(left=4000, right=4999, closed="right"): "#F7A5F4",
            pd.Interval(left=5000, right=5999, closed="right"): "#FF199A",
            pd.Interval(left=6000, right=6999, closed="right"): "#6219FF",
            pd.Interval(left=7000, right=7999, closed="right"): "#6EABFF",
            pd.Interval(left=8000, right=8999, closed="right"): "#27895E",
            pd.Interval(left=9000, right=9999, closed="right"): "#000000",
        },
        "names": [
            "Food & Live Animals",
            "Beverages & Tobacco",
            "Raw Materials",
            "Mineral fuels, Lubricants & Related Materials",
            "Animal & Vegetable Oils, Fats & Waxes",
            "Chemicals",
            "Manufactured Goods by Material",
            "Machinery & Transport Equipment",
            "Miscellaneous Manufactured Articles",
            "Miscellaneous",
        ],
    },
    "Sector": {
        "color": {
            pd.Interval(left=101, right=105, closed="right"): "#F57373",
            pd.Interval(left=206, right=214, closed="right"): "#AAD75A",
            pd.Interval(left=315, right=315, closed="right"): "#D7B55A",
            pd.Interval(left=416, right=424, closed="right"): "#FAF51B",
            pd.Interval(left=525, right=527, closed="right"): "#643F0D",
            pd.Interval(left=628, right=638, closed="right"): "#EA14B0",
            pd.Interval(left=739, right=740, closed="right"): "#854CEB",
            pd.Interval(left=841, right=843, closed="right"): "#B87B7B",
            pd.Interval(left=944, right=946, closed="right"): "#204F0B",
            pd.Interval(left=1047, right=1049, closed="right"): "#F8F0A4",
            pd.Interval(left=1150, right=1163, closed="right"): "#00CC1E",
            pd.Interval(left=1264, right=1267, closed="right"): "#00FF25",
            pd.Interval(left=1368, right=1370, closed="right"): "#AB9E97",
            pd.Interval(left=1471, right=1471, closed="right"): "#76E5E7",
            pd.Interval(left=1572, right=1583, closed="right"): "#4E6D6E",
            pd.Interval(left=1684, right=1685, closed="right"): "#314480",
            pd.Interval(left=1786, right=1789, closed="right"): "#85F6FF",
            pd.Interval(left=1890, right=1892, closed="right"): "#3F004C",
            pd.Interval(left=1993, right=1993, closed="right"): "#B3B3B3",
            pd.Interval(left=2094, right=2096, closed="right"): "#614906",
            pd.Interval(left=2197, right=2197, closed="right"): "#B584C4",
            pd.Interval(left=2298, right=2298, closed="right"): "#000000",
            pd.Interval(left=9999, right=99105, closed="right"): "#643F0D",
            pd.Interval(left=99106, right=99107, closed="right"): "#242c57",
            pd.Interval(left=99108, right=99110, closed="right"): "#85F6FF",
            pd.Interval(left=99111, right=99111, closed="right"): "#a949fc",
            pd.Interval(left=99113, right=99115, closed="right"): "#ff036c",
            pd.Interval(left=99116, right=99116, closed="right"): "#3e3d40",
            pd.Interval(left=99117, right=99119, closed="right"): "#ffe0ab",
            pd.Interval(left=99120, right=99120, closed="right"): "#59dea2",
            pd.Interval(left=99121, right=99121, closed="right"): "#ebadff",
            pd.Interval(left=99122, right=99124, closed="right"): "#056ef7",
            pd.Interval(left=99125, right=99125, closed="right"): "#b157cf",
            pd.Interval(left=99126, right=99127, closed="right"): "#7391f5",
            pd.Interval(left=99128, right=99128, closed="right"): "#4a5947",
            pd.Interval(left=99129, right=99129, closed="right"): "#a3021a",
            pd.Interval(left=99130, right=99130, closed="right"): "#a9db04",
            pd.Interval(left=99131, right=99131, closed="right"): "#ff059f",
            pd.Interval(left=99132, right=99132, closed="right"): "#addeff",
            pd.Interval(left=99133, right=99133, closed="right"): "#0019bf",
            pd.Interval(left=99134, right=99134, closed="right"): "#bf5900",
            pd.Interval(left=99135, right=99135, closed="right"): "#fcb3eb",
        },
        "names": [
            "Animal Products",
            "Vegetable Products",
            "Animal and Vegetable Bi-Products",
            "Foodstuffs",
            "Mineral Products",
            "Chemical Products",
            "Plastics and Rubbers",
            "Animal Hides",
            "Wood Products",
            "Paper Goods",
            "Textiles",
            "Footwear and Headwear",
            "Stone And Glass",
            "Precious Metals",
            "Metals",
            "Machines",
            "Transportation",
            "Instruments",
            "Weapons",
            "Miscellaneous",
            "Arts and Antiques",
            "Unspecified",
            "Mineral Fuels, Oils, Distillation Products, etc",
            "Manufacturing & Maintenance",
            "Transport",
            "Postal & Courier Services",
            "Travel Goods",
            "Local Transport, Acommodation, Food-Serving Services",
            "Construction",
            "Insurance, Pension, Financial Services",
            "Real State",
            "Intellectual Property",
            "Information & Technology Services",
            "Research & Development",
            "Consulting & Engineering Services",
            "Waste Treatment",
            "Operational Leasing Services",
            "Trade & Business Services",
            "Audiovisual Services",
            "Health Services",
            "Education Services",
            "Heritage & Recreational Services",
            "Government Goods and Services",
        ],
    },
}
