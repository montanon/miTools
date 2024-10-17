import os

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
from pyvis.network import Network

from mitools.etl import check_if_table, read_sql_table
from mitools.exceptions import ArgumentValueError


def vectors_from_proximity_matrix(
    proximity_matrix: DataFrame,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    proximity_column: str = "weight",
    sort_by: str = None,
    sort_ascending: bool = False,
) -> DataFrame:
    if not isinstance(proximity_matrix, DataFrame):
        raise ArgumentValueError("Input must be a pandas DataFrame.")
    if sort_by is not None and sort_by not in [
        orig_product,
        dest_product,
        proximity_column,
    ]:
        raise ArgumentValueError(
            f"Column '{sort_by}' not available in output DataFrame."
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


def build_mst_graph(
    G,
    proximity_vectors,
    weight="weight",
    weights_th=None,
    n_extra_edges=None,
    pct_extra_edges=None,
):
    proximity_vectors = proximity_vectors.sort_values(by="weight", ascending=False)

    MST = nx.maximum_spanning_tree(G, weight=weight)

    extra_edges = None
    if weights_th is not None:
        extra_edges = proximity_vectors.query("weight >= @weights_th")
    elif n_extra_edges is not None:
        n_extra_edges += len(MST.edges)
        extra_edges = proximity_vectors.iloc[:n_extra_edges, :]
    elif pct_extra_edges is not None:
        n_extra_edges = int(
            (proximity_vectors.shape[0] - len(MST.edges)) * pct_extra_edges
        )
        extra_edges = proximity_vectors.iloc[:n_extra_edges, :]

    if extra_edges is not None:
        exG = build_nx_graph(extra_edges)

        _G = nx.compose(MST, exG)

        for u, v, d in G.edges(data=True):
            if _G.has_edge(u, v):
                _G[u][v]["weight"] = d["weight"]

        return _G
    return MST


def build_vis_graph(
    mst,
    proximity_vectors,
    product_name_col,
    id_col,
    products_codes=None,
    color_bins=None,
    physics=False,
    node_size=10,
    label_size=20,
):
    # , filter_menu=True)#, select_menu=True)
    net = Network(height="700px", notebook=True)
    net.from_nx(mst)

    set_net_nodes_size(net, size=node_size)
    set_net_nodes_labels_size(net, size=label_size)

    width_bins = build_width_bins(proximity_vectors)

    if products_codes is not None and color_bins is not None:
        set_net_nodes_color(net, products_codes, color_bins, product_code_col=id_col)
    set_net_edges_width(net, width_bins)
    if products_codes is not None:
        set_net_nodes_label(
            net, products_codes, label_col=product_name_col, product_code_col=id_col
        )

    net.barnes_hut(
        gravity=-1000000,
        central_gravity=0.0,
        spring_length=500,
        spring_strength=2,
        damping=0.1,
        overlap=1,
    )

    if physics:
        net.show_buttons(filter_=["physics"])

    return net


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


def vectors_from_proximity_matrices(
    proximity_matrices, tablenames, conn, recalculate=False
):
    proximity_vectors_dfs = {}

    for (temp_id, proximity_matrix), (_, tablename) in zip(
        proximity_matrices.items(), tablenames.items()
    ):
        if not check_if_table(conn, tablename) or recalculate:
            proximity_vectors = vectors_from_proximity_matrix(proximity_matrix)

            proximity_vectors.to_sql(tablename, conn, if_exists="replace")

        else:
            proximity_vectors = read_sql_table(tablename, conn)

        proximity_vectors_dfs[temp_id] = proximity_vectors

    return proximity_vectors_dfs


def build_nx_graphs(
    proximity_vectors_dfs, id_col, value_col, networks_folder, recalculate=False
):
    graph_files = {}
    graphs = {}

    for temp_id, proximity_vectors in proximity_vectors_dfs.items():
        gml_name = f"{temp_id}_{id_col}_{value_col}_G_graph.gml".replace(" ", "_")
        gml_path = os.path.join(networks_folder, gml_name)

        create_graph_network = not os.path.exists(gml_path)

        if create_graph_network or recalculate:
            G = build_nx_graph(proximity_vectors)
            nx.write_gml(G, gml_path)
        else:
            G = nx.read_gml(gml_path)

        graph_files[temp_id] = gml_path
        graphs[temp_id] = G

    return graph_files, graphs


def build_nx_graph(
    proximity_vectors, orig_product="product_i", dest_product="product_j"
):
    G = nx.from_pandas_edgelist(
        proximity_vectors, source=orig_product, target=dest_product, edge_attr=True
    )

    return G


def build_mst_graphs(
    proximity_vectors_dfs,
    graphs,
    id_col,
    value_col,
    networks_folder,
    thresholds=None,
    recalculate=False,
):
    weights_th = thresholds["weights_th"]
    n_extra_edges = thresholds["n_extra_edges"]
    pct_extra_edges = thresholds["pct_extra_edges"]

    mst_files = {}
    mst_graphs = {}

    for (temp_id, G), (_, proximity_vectors) in zip(
        graphs.items(), proximity_vectors_dfs.items()
    ):
        mst_gml_name = f"{temp_id}_{id_col}_{value_col}_mst"
        mst_gml_name += f"_{str(weights_th).replace('.', '')}_{str(n_extra_edges).replace('.', '')}_{str(pct_extra_edges).replace('.', '')}.gml"
        mst_gml_name = mst_gml_name.replace(" ", "_")
        mst_gml_path = os.path.join(networks_folder, mst_gml_name)

        create_mst_network = not os.path.exists(mst_gml_path)

        if create_mst_network or recalculate:
            MST = build_mst_graph(
                G,
                proximity_vectors,
                weights_th=weights_th,
                n_extra_edges=n_extra_edges,
                pct_extra_edges=pct_extra_edges,
            )
            nx.write_gml(MST, mst_gml_path)
        else:
            MST = nx.read_gml(mst_gml_path)

        mst_files[temp_id] = mst_gml_path
        mst_graphs[temp_id] = MST

    return mst_files, mst_graphs


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
            label_size=label_size,
        )

        nets[temp_id] = net

    return nets


def build_color_bins(id_col):
    bins = _bins_colors.get(id_col, None)

    if bins is not None:
        return bins["color"], bins["names"]
    else:
        return None, None


def build_width_bins(proximity_vectors, widths=[2, 5, 10, 15, 30]):
    bins = pd.cut(
        proximity_vectors.sort_values(by="weight", ascending=True)["weight"], 5
    )
    bins = bins.unique()

    return {b: w for w, b in zip(widths, bins)}


def set_net_nodes_size(net, size=10):
    for node in net.nodes:
        node["size"] = size


def set_net_nodes_labels_size(net, size=20):
    for node in net.nodes:
        node["font"] = f"{size}px arial black"


def set_net_nodes_color(
    net, sitc_codes, color_bins, product_code_col="sitc_product_code"
):
    if color_bins is not None:
        for node in net.nodes:
            sitc_id = sitc_codes.loc[
                sitc_codes[product_code_col] == int(node["id"]), product_code_col
            ].values[0]
            for b, c in color_bins.items():
                if int(sitc_id) in b:
                    node["color"] = c
                    continue


def set_net_edges_width(net, width_bins):
    for edge in net.edges:
        for b, w in width_bins.items():
            if edge["width"] in b:
                edge["width"] = w
                continue


def set_net_nodes_label(
    net,
    sitc_codes,
    label_col="sitc_product_name_short_en",
    product_code_col="sitc_product_code",
):
    for node in net.nodes:
        product_label = sitc_codes.loc[
            sitc_codes[product_code_col] == int(node["id"]), label_col
        ].values[0]
        node["label"] = product_label


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
    _net = Network(height="700px", notebook=True)

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
    nx_graph = nx.DiGraph() if pyvis_network.directed else nx.Graph()

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
