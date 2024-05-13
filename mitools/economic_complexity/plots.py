import random
import statistics
from string import ascii_uppercase, digits
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from pandas import DataFrame
from scipy.spatial.distance import squareform

from ..pandas import idxslice
from ..utils import stretch_string
from ..visuals import (
    adjust_axes_lims,
    is_axes_empty,
)
from .objects import Product, ProductsBasket

Color = Union[Tuple[int, int, int], str]


def plot_country_sector_distributions_evolution(
    data, country, years, colors, value=False, kde=False
):
    sectors = data["Sector"].unique()
    sectors.sort()
    nrows, ncols = len(years), len(sectors)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(100, 30))
    years_string = f'{", ".join([str(year) for year in years[:-1]])} & {years[-1]}'
    title = f"{country} Products by Sector between during {years_string}"
    fig.suptitle(title, fontsize=28, y=0.925)
    fig.subplots_adjust(hspace=0.75)
    plot_data = data[
        data.index.get_level_values("Year").isin(years)
        & (data.index.get_level_values("Country") == country)
    ]
    for nrow, year in enumerate(years):
        yearly_data = plot_data[plot_data.index.get_level_values("Year") == year]
        for ncol, sector in enumerate(sectors):
            ax = axes[nrow, ncol]
            products = []
            for idx, row in yearly_data.query(
                f"Sector == '{sector}' and RCA >= 1.0"
            ).iterrows():
                products.append(
                    Product(
                        name=row["HS6"],
                        code=row["HS6"],
                        pci=row["PCI"],
                        value=row["Trade Value"],
                    )
                )
            basket = ProductsBasket(products)
            if not value:
                plot_distribution(
                    basket,
                    bins=30,
                    n=7,
                    color=colors[sector],
                    title=f"{year}-{country}-{sector}",
                    ax=ax,
                )
            else:
                plot_value_distribution(
                    plot_data.query(
                        f"Sector == '{sector}' and RCA >= 1.0 and Year == {year}"
                    ),
                    basket,
                    bins=30,
                    n=7,
                    color=colors[sector],
                    title=f"{year}-{country}-{sector}",
                    ax=ax,
                    kde=kde,
                )
    axes = adjust_axes_lims(axes, y=False)
    return axes


def plot_country_sector_distributions_by_year(
    data, country, year, colors, value=False, kde=False
):
    sectors = data["Sector"].unique()
    sectors.sort()
    plot_data = data.query(f'Year == {year} and Country == "{country}"')
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(60, 30))
    title = f"{country} Products by Sector for the year {year}"
    fig.suptitle(title, fontsize=28, y=0.925)
    fig.subplots_adjust(hspace=0.75)
    indices = [(r, c) for r in range(nrows) for c in range(ncols)]
    for nrow, ncol in indices:
        sector_idx = nrow * ncols + ncol
        if sector_idx < len(sectors):
            ax = axes[nrow, ncol]
            sector = sectors[sector_idx]
            products = []
            for idx, row in plot_data.query(
                f"Sector == '{sector}' and RCA >= 1.0 and Year == {year}"
            ).iterrows():
                products.append(
                    Product(
                        name=row["HS6"],
                        code=row["HS6"],
                        pci=row["PCI"],
                        value=row["Trade Value"],
                    )
                )
            basket = ProductsBasket(products)
            if not value:
                plot_distribution(
                    basket,
                    bins=30,
                    n=7,
                    color=colors[sector],
                    title=f"{year}-{country}-{sector}",
                    ax=ax,
                )
            else:
                plot_value_distribution(
                    plot_data.query(
                        f"Sector == '{sector}' and RCA >= 1.0 and Year == {year}"
                    ),
                    basket,
                    bins=30,
                    n=4,
                    color=colors[sector],
                    title=f"{year}-{country}-{sector}",
                    ax=ax,
                    kde=kde,
                )
    axes = adjust_axes_lims(axes, y=False)
    return axes


def plot_distribution(
    basket,
    n=10,
    bins=100,
    color=None,
    title=None,
    ax=None,
    info=True,
    quantiles=True,
    line_kws=None,
    **kwargs,
):
    # Extract pci values
    pcis = [product.pci for product in basket.products]
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 6))
    ax = sns.histplot(
        pcis,
        bins=bins,
        kde=True,
        color=color,
        ax=ax,
        line_kws=line_kws or {},
        **kwargs,
        zorder=1,
    )
    if info:
        mean_val = statistics.mean(pcis) if pcis else np.nan
        std_val = statistics.stdev(pcis) if pcis else np.nan
        min_val = min(pcis) if pcis else np.nan
        max_val = max(pcis) if pcis else np.nan
        # Add text for statistics
        stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
        )
    if quantiles:
        y_text_position = (
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.075
        )
        quantile_products = basket.products_closest_to_quantiles(n)
        for product in quantile_products:
            ax.axvline(x=product.pci, linestyle="--", color="gray", zorder=0)
            ax.text(
                product.pci,
                y_text_position,
                product.name,
                rotation=45,
                fontsize=12,
                horizontalalignment="right",
                verticalalignment="top",
            )
    # Final plot adjustments
    if title:
        title = stretch_string(title, 60)
        ax.set_title(title)
    ax.set_xlabel("PCI", ha="left")
    ax.xaxis.set_label_coords(0, -0.02)
    ax.set_ylabel("Frequency")
    return ax


def generate_random_products(num_products):
    products = []
    existing_codes = set()
    existing_names = set()
    while len(products) < num_products:
        code = random.randint(1000, 9999)
        name = "".join(random.choices(ascii_uppercase + digits, k=5))
        pci = random.uniform(-5.0, 5.0)
        value = random.uniform(1_000, 1_000_000)
        if code not in existing_codes and name not in existing_names:
            products.append(Product(code, name, pci, value))
            existing_codes.add(code)
            existing_names.add(name)
    return products


def plot_value_distribution(
    data,
    basket,
    n=10,
    bins=100,
    color=None,
    title=None,
    ax=None,
    info=True,
    quantiles=True,
    major_exports=True,
    kde=False,
    weighted_mean=True,
    bin_step=0.001,
    line_kws=None,
    **kwargs,
):
    data = data.copy(deep=True).reset_index()
    data["Trade Value"] = data["Trade Value"] * 1e-6
    data = data[
        [
            c
            for c in data.columns
            if c
            not in [
                "Year",
                "Section",
                "Income Group",
                "Current Income Group",
                "ECI",
                "normECI",
                "stdECI",
                "normPCI",
                "stdPCI",
                "Relatedness",
                "PCI*RCA",
                "normPCI*RCA",
                "stdPCI*RCA",
                "notRCA",
                "notRCA*Rel*PCI",
                "notRCA*Rel*normPCI",
                "notRCA*Rel*stdPCI",
                "SECI",
                "normSECI",
                "stdSECI",
                "SCI",
                "normSCI",
                "stdSCI",
                "SCP",
                "SCP",
                "normSCP",
                "stdSCP",
            ]
        ]
    ]
    bin_edges = np.arange(data["PCI"].min(), data["PCI"].max() + bin_step, bin_step)
    data["PCIbin"] = pd.cut(data.loc[data["RCA"] != 0.0, "PCI"], bins=bin_edges)
    agg_bins = data.groupby("PCIbin")[["Trade Value"]].sum().sort_index()
    agg_bins.index = pd.IntervalIndex(agg_bins.index.categories).mid
    agg_bins["Trade Value"] = agg_bins["Trade Value"]
    # Extract pci values
    pcis = [product.pci for product in basket.products]
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 6))

    scatter_plot = True
    if not scatter_plot:
        ax.bar(
            agg_bins.index,
            agg_bins["Trade Value"],
            width=bin_step,
            color=color,
            edgecolor="k",
            **kwargs,
            zorder=1,
        )
    else:
        ax = sns.scatterplot(data, x="PCI", y="Trade Value", color=color, ax=ax)
        ax.set_ylim(-1.0, ax.get_ylim()[1] * 1.1)
        ax.set_zorder(1)
    if kde:
        ax1 = ax.twinx()
        ax1.set_zorder(0)
        sns.kdeplot(pcis, ax=ax1, color=color, zorder=1, bw_adjust=0.5, linewidth=2)
    if info:
        mean_val = statistics.mean(pcis) if pcis else np.nan
        std_val = statistics.stdev(pcis) if pcis else np.nan
        min_val = min(pcis) if pcis else np.nan
        max_val = max(pcis) if pcis else np.nan
        # Add text for statistics
        stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=16,
        )
    if quantiles:
        y_text_position = (
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.075
        )  # 5% below the x-axis
        quantile_products = basket.products_closest_to_quantiles(n)
        for product in quantile_products:
            ax.axvline(
                x=product.pci, linestyle="--", color="gray", zorder=0, alpha=0.85
            )
            ax.text(
                product.pci,
                y_text_position,
                stretch_string(product.name, 35),
                rotation=45,
                fontsize=16,
                horizontalalignment="right",
                verticalalignment="top",
            )

    if major_exports:
        x_min, x_max = ax.get_xlim()
        x_mean = (x_min + x_max) / 2
        if not scatter_plot:
            top_exports = agg_bins["Trade Value"].nlargest(2)
        else:
            top_exports = data.nlargest(2, "Trade Value")
        if not scatter_plot:
            if top_exports.iloc[0] * 0.5 > top_exports.iloc[1]:
                top_exports = top_exports.iloc[[0]]
        for n, (pci_value, trade_value) in top_exports[
            ["PCI", "Trade Value"]
        ].iterrows():
            if not scatter_plot:
                in_pci_range = (
                    data["PCIbin"].apply(check_interval, args=(pci_value,)) == True
                )
                product = data.loc[in_pci_range]
                total_products = product.shape[0]
                product = product.sort_values("Trade Value").iloc[-1]
            else:
                product = top_exports.loc[n]
            product_name = product["HS6"]
            product_value = product["Trade Value"]
            product_pct = round(100 * (product_value / trade_value), 2)
            product_name = f"{stretch_string(product_name, 40)} \n"
            if not scatter_plot:
                product_name += f"[{product_pct}% of {total_products} products]"
            h_al = "right" if pci_value > x_mean else "left"
            ax.text(
                pci_value,
                trade_value,
                product_name,
                rotation=0,
                fontsize=12,
                horizontalalignment=h_al,
                verticalalignment="bottom",
                zorder=10,
            )

    if weighted_mean:
        weighted_value = sum(product.pci * product.value for product in basket.products)
        weighted_value /= sum(product.value for product in basket.products)
        ax.axvline(
            x=weighted_value,
            linestyle="--",
            linewidth=5.0,
            color="green",
            zorder=10,
            alpha=0.75,
        )
        ax.text(
            weighted_value * 1.1,
            ax.get_ylim()[1] * 0.55,
            f"SECI={round(weighted_value, 3)}",
            fontsize=24,
            zorder=99,
            verticalalignment="bottom",
            horizontalalignment="left",
            color="green",
            alpha=0.75,
        )

    if title:
        title = stretch_string(title, 30)
        ax.set_title(title)
    ax.set_xlabel("PCI", ha="left")
    ax.xaxis.set_label_coords(0, -0.02)
    ax.set_ylabel("Trade Value (million USD)")
    return ax


def check_interval(interval, value):
    if pd.isna(interval):
        return False
    return interval.left <= value < interval.right


def custom_agg_rca(group, n=5):
    return 1.0 if group.sum() >= n else 0.0


def display_rca_matrix(rca_matrix: DataFrame, tag=""):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    rca_matrix_visualization(rca_matrix, tag, ax=axes[0])
    rca_matrix_histogram(rca_matrix, tag, ax=axes[1])


def rca_matrix_visualization(rca_matrix: DataFrame, tag="", ax=None):
    if ax is None:
        _, axes = plt.subplots(1, 1, figsize=(16, 6))

    sns.heatmap(rca_matrix, cmap="viridis", vmin=0.0, ax=ax)
    ax.set_title(f"{tag} - RCA Matrix")


def rca_matrix_histogram(rca_matrix: DataFrame, tag="", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 6))

    data_to_plot = rca_matrix.unstack()
    data_to_plot = data_to_plot[data_to_plot != 0]
    data_to_plot.hist(ax=ax, bins=100)
    ax.set_title(f"{tag} - RCA Values")
    ax.set_ylabel("Quantity")
    ax.set_xlabel("RCA")


def display_proximity_matrix(proximity_matrix: DataFrame, tag=""):
    proximity_matrix_visualization(proximity_matrix, tag)
    proximity_matrix_histogram(proximity_matrix, tag)


def proximity_matrix_visualization(proximity_matrix, tag=""):
    _, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes = [axes]

    sns.heatmap(proximity_matrix, cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[0])
    axes[0].set_title(f"{tag} - Proximity Matrix")


def proximity_matrix_histogram(proximity_matrix, tag=""):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    condensed_proximity = squareform(proximity_matrix)
    axes[0].hist(condensed_proximity, bins=50)
    axes[0].set_xlabel("Proximity Value")
    axes[0].set_title(f"{tag} - All Proximity Values from Condensed Matrix")

    axes[-1].hist(condensed_proximity[condensed_proximity > 0], bins=50)
    axes[-1].set_xlabel("Proximity Value")
    axes[-1].set_title(f"{tag} - Non-Zero Proximity Values from Condensed Matrix")


def plot_countries_ecis_indicator_scatter(
    data: DataFrame,
    entities: List[str],
    x_vars_cols: List[str],
    y_var_col: str,
    name_tag: str,
    groups: List[str],
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
    n_steps: Optional[int] = 1,
    ncols: Optional[int] = 3,
    year_labels: Optional[bool] = False,
    colors: Optional[List[Color]] = None,
    groups_colors: Optional[Dict[str, Color]] = None,
    figsize: Optional[Tuple[float, float]] = (7, 7),
    arrows: Optional[bool] = False,
    arrow_style: Optional[bool] = None,
    arrow_kwargs: Optional[Dict[str, Any]] = None,
    set_arrows_ax_limits: Optional[bool] = False,
    marker_kwargs: Optional[Dict[str, Any]] = None,
    axes: Optional[Axes] = None,
) -> Axes:
    nrows = (len(x_vars_cols) + 1) // ncols
    if axes is None:
        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows)
        )
    for entity in entities:
        country_data = data.query(f"{entity_col} == @entity")
        if country_data.empty:
            continue
        axes = plot_country_ecis_indicator_scatter(
            country_data=country_data,
            x_vars_cols=x_vars_cols,
            y_var_col=y_var_col,
            name_tag=name_tag,
            groups=groups,
            groups_col=groups_col,
            entity_col=entity_col,
            time_col=time_col,
            n_steps=n_steps,
            ncols=ncols,
            year_labels=year_labels,
            colors=colors,
            groups_colors=groups_colors,
            figsize=figsize,
            arrows=arrows,
            arrow_style=arrow_style,
            arrow_kwargs=arrow_kwargs,
            set_arrows_ax_limits=set_arrows_ax_limits,
            marker_kwargs=marker_kwargs,
            axes=axes,
        )
    axes.flat[0].figure.suptitle(
        f"Countries {name_tag} vs {y_var_col}",
        fontsize=22,
        y=0.9,
        verticalalignment="bottom",
        horizontalalignment="center",
    )
    return axes


def plot_country_ecis_indicator_scatter(
    country_data: DataFrame,
    x_vars_cols: List[str],
    y_var_col: str,
    name_tag: str,
    groups: List[str],
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
    n_steps: Optional[int] = 1,
    ncols: Optional[int] = 3,
    year_labels: Optional[bool] = True,
    colors: Optional[List[Color]] = None,
    groups_colors: Optional[Dict[str, Color]] = None,
    figsize: Optional[Tuple[float, float]] = (9, 9),
    arrows: Optional[bool] = True,
    arrow_style: Optional[ArrowStyle] = None,
    arrow_kwargs: Optional[Dict[str, Any]] = None,
    set_arrows_ax_limits: Optional[bool] = True,
    marker_kwargs: Optional[Dict[str, Any]] = None,
    axes: Optional[Axes] = None,
) -> Axes:
    entity = country_data.index.get_level_values(entity_col).unique()[0]
    nrows = (len(x_vars_cols) + 1) // ncols
    if axes is None:
        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows)
        )
    for n, (ax, x_var_col) in enumerate(zip(axes.flat, x_vars_cols)):
        ax = plot_country_eci_indicator_scatter(
            country_data=country_data,
            x_var_col=x_var_col,
            y_var_col=y_var_col,
            groups=groups,
            groups_col=groups_col,
            time_col=time_col,
            n_steps=n_steps,
            year_labels=year_labels,
            color=colors[n] if colors else None,
            groups_colors=groups_colors,
            figsize=figsize,
            arrows=arrows,
            arrow_style=arrow_style,
            arrow_kwargs=arrow_kwargs,
            set_arrows_ax_limits=set_arrows_ax_limits,
            marker_kwargs=marker_kwargs,
            ax=ax,
        )
    axes.flat[0].figure.suptitle(
        f"{entity} {name_tag}s vs {y_var_col} Evolution",
        fontsize=24,
        y=0.925,
        verticalalignment="bottom",
        horizontalalignment="center",
    )
    last_ax = axes.flat[-1]
    if is_axes_empty(last_ax) and groups_colors is not None:
        last_ax.cla()
        last_ax.set_xticks([])
        last_ax.set_yticks([])
        last_ax.axis("off")
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle="None",
                markersize=10,
                markeredgecolor="k",
                label=label,
            )
            for label, color in groups_colors.items()
        ]
        last_ax.legend(handles=legend_handles, fontsize=16, loc="center left", ncols=1)
    return axes


def plot_country_eci_indicator_scatter(
    country_data: DataFrame,
    x_var_col: str,
    y_var_col: str,
    groups: List[str],
    groups_col: Optional[str] = "Income Group",
    time_col: Optional[str] = "Year",
    n_steps: Optional[int] = 1,
    year_labels: Optional[bool] = True,
    color: Optional[Color] = None,
    groups_colors: Optional[Dict[str, Color]] = None,
    figsize: Optional[Tuple[float, float]] = (9, 9),
    arrows: Optional[bool] = True,
    arrow_style: Optional[ArrowStyle] = None,
    arrow_kwargs: Optional[Dict[str, Any]] = None,
    set_arrows_ax_limits: Optional[bool] = True,
    marker_kwargs: Optional[Dict[str, Any]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
    if marker_kwargs is None:
        marker_kwargs = {
            "marker": "o",
            "markeredgewidth": 2,
            "markersize": 10,
            "label_fontsize": 12,
        }
    if arrow_style is None:
        arrow_style = ArrowStyle("Fancy", head_length=10, head_width=5, tail_width=0.4)
    if arrow_kwargs is None:
        arrow_kwargs = dict(
            connectionstyle="arc3", color="grey", linewidth=1, linestyle=":", alpha=0.75
        )
    years = country_data.index.get_level_values(time_col)[::n_steps]
    steps_index = country_data.index[::n_steps]
    groups = [
        level
        for level in groups
        if level in country_data.loc[steps_index, :].index.get_level_values(groups_col)
    ]
    for group in groups:
        group_idxslice = idxslice(country_data, level=groups_col, value=group, axis=0)
        ax.plot(
            country_data.loc[steps_index, :].loc[group_idxslice, x_var_col].values,
            country_data.loc[steps_index, :].loc[group_idxslice, y_var_col].values,
            markeredgecolor=color if color else "k",
            markerfacecolor=groups_colors[group] if groups_colors else "white",
            marker=marker_kwargs["marker"],
            markeredgewidth=marker_kwargs["markeredgewidth"],
            markersize=marker_kwargs["markersize"],
            linestyle="",
            alpha=0.75,
        )
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    x = country_data[x_var_col].values[::n_steps]
    y = country_data[y_var_col].values[::n_steps]
    if arrows:
        for i in range(len(x) - 1):
            if any(
                [np.isnan(x[i]), np.isnan(x[i + 1]), np.isnan(y[i]), np.isnan(y[i + 1])]
            ):
                continue
            arrow = FancyArrowPatch(
                (x[i], y[i]),
                (x[i + 1], y[i + 1]),
                arrowstyle=arrow_style,
                connectionstyle=arrow_kwargs["connectionstyle"],
                color=arrow_kwargs["color"],
                linewidth=arrow_kwargs["linewidth"],
                linestyle=arrow_kwargs["linestyle"],
                alpha=arrow_kwargs["alpha"],
            )
            ax.add_patch(arrow)
        if set_arrows_ax_limits:
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
    if year_labels:
        offset = 0.01 * (
            y_lims[1] - y_lims[0]
        )  # Offset for annotation so it doesn't sit right on the point
        for xi, yi, year in zip(x, y, years):
            ax.annotate(
                str(year),
                (xi, yi - offset),
                textcoords="offset points",
                xytext=(0, -10),
                ha="center",
                fontsize=marker_kwargs["label_fontsize"],
            )
    ax.set_ylabel(y_var_col)
    ax.set_xlabel(x_var_col)
    return ax


def plot_income_levels_ecis_indicator_scatter(
    data: DataFrame,
    x_vars_cols: List[str],
    y_var_col: str,
    name_tag: str,
    groups: List[str],
    all_groups: str,
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
    colors: List[Color] = None,
    groups_colors: Dict[str, Color] = None,
    figsize: Optional[Tuple[float, float]] = (9, 7),
    marker_kwargs: Optional[Dict[str, Any]] = None,
    adjust_axes_lims_kwargs: Optional[Dict[str, Any]] = None,
) -> Axes:
    if adjust_axes_lims_kwargs is None:
        adjust_axes_lims_kwargs = {"mode": "rows", "x": True, "y": True}
    nrows = len(x_vars_cols)
    ncols = len(groups)
    _, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows)
    )
    for n, group in enumerate(groups):
        if group != all_groups:
            group_data = data.loc[
                idxslice(data, level=groups_col, value=group, axis=0), :
            ].copy(deep=True)
            groups = [group]
        else:
            group_data = data.copy(deep=True)
            groups = group_data.index.get_level_values(groups_col).unique().tolist()
        entities = data.index.get_level_values(entity_col).unique().tolist()
        plot_countries_ecis_indicator_scatter(
            data=group_data,
            entities=entities,
            x_vars_cols=x_vars_cols,
            y_var_col=y_var_col,
            name_tag=name_tag,
            groups=groups,
            groups_col=groups_col,
            entity_col=entity_col,
            time_col=time_col,
            n_steps=1,
            ncols=1,
            year_labels=False,
            colors=colors,
            groups_colors=groups_colors,
            figsize=figsize,
            arrows=False,
            arrow_style=None,
            arrow_kwargs=None,
            set_arrows_ax_limits=False,
            marker_kwargs=marker_kwargs,
            axes=axes.flat[n::ncols],
        )
    axes = adjust_axes_lims(axes, **adjust_axes_lims_kwargs)
    return axes
