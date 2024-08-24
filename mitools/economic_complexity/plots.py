import random
import statistics
import string
from io import BytesIO
from math import ceil
from pathlib import Path
from string import ascii_uppercase, digits
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cairosvg
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
from PIL import Image
from scipy.spatial.distance import squareform

from ..country_converter import cc
from ..pandas import idxslice, quantize_group
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


def plot_countries_ecis_indicators_scatter(
    data: DataFrame,
    income_level: str,
    y_var_col: str,
    name_tag: str,
    n_steps: int,
    n_cols: int,
    groups: List[str],
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
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
):
    x_vars_cols = [c for c in data.columns if c.endswith(f" {name_tag}")]
    if income_level != "All income":
        _data = data.loc[pd.IndexSlice[:, :, income_level, :, :], :].copy(deep=True)
    else:
        _data = data
    countries = _data.index.get_level_values("Country").unique().tolist()
    axes = plot_countries_ecis_indicator_scatter(
        data=_data,
        entities=countries,
        x_vars_cols=x_vars_cols,
        y_var_col=y_var_col,
        name_tag=name_tag,
        groups=groups,
        groups_col=groups_col,
        entity_col=entity_col,
        time_col=time_col,
        n_steps=n_steps,
        ncols=n_cols,
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

    return axes


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
    nrows = ceil((len(x_vars_cols)) / ncols)
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


def remove_ylabel_for_non_leftmost(axes):
    for ax in axes.flat:
        if ax.get_subplotspec().colspan.start > 0:
            ax.set_ylabel("")
    return axes


def remove_xlabel_for_non_bottom(axes):
    total_rows = axes.shape[0]
    for ax in axes.flat:
        if ax.get_subplotspec().rowspan.stop < total_rows:
            ax.set_xlabel("")
    return axes


def log_tick_formatter(val, pos):
    return f"$10^{{{int(val)}}}$"


def set_labels_and_titles(ax, letter, new_xlabel, label_size, title_size, tick_size):
    ax.set_xlabel(r"$\mathrm{SCI}_{log}$", fontsize=label_size)
    ax.set_ylabel(
        r"$\mathrm{CO}_2\,\mathrm{emissions\ (metric\ tons\ per\ capita)}_{log}$",
        fontsize=label_size,
    )
    new_title = new_xlabel.replace(" SCI", " Sector").replace("\n", "")
    ax.set_title(f"{letter}) {new_title}:", fontsize=title_size, loc="left")
    ax.tick_params(axis="both", labelsize=tick_size)


def set_log_scale(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    ax.set_xticks([-3, -2, -1, 0])


def create_legend(fig, income_colors, legend_fontsize):
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=12,
            markeredgecolor="k",
            label=label,
        )
        for label, color in income_colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        fontsize=legend_fontsize,
        ncol=4,
    )


def format_axes(
    axes,
    axis_label_fontsize,
    axis_tick_fontsize,
    axis_title_fontsize,
    legend_fontsize,
    new_labels,
    income_colors,
    axes_to_remove,
    relevant_axes,
):
    for letter, ax, new_xlabel in zip(string.ascii_lowercase, axes.flat, new_labels):
        set_labels_and_titles(
            ax,
            letter,
            new_xlabel,
            axis_label_fontsize,
            axis_title_fontsize,
            axis_tick_fontsize,
        )
    axes = remove_ylabel_for_non_leftmost(axes)
    for ax in axes.flat:
        set_log_scale(ax)
    if axes_to_remove > 0:
        for ax in axes.flat[-axes_to_remove:]:
            ax.remove()
    for ax in axes.flat[relevant_axes:]:
        if ax is not None:
            ax.clear()
            ax.axis("off")
    create_legend(axes.flat[0].get_figure(), income_colors, legend_fontsize)
    return axes


def adjust_figure_size(fig, width_mm):
    current_width, current_height = fig.get_size_inches()
    aspect_ratio = current_height / current_width
    new_width_inches = width_mm / 25.4
    new_height_inches = new_width_inches * aspect_ratio
    fig.set_size_inches(new_width_inches, new_height_inches)


def adjust_axes_alpha(axes, alpha):
    for ax in axes.flat:
        for line in ax.get_lines():
            line.set_alpha(alpha)
    return axes


def format_eci_scatter_plot(
    axes,
    new_labels,
    axis_label_fontsize,
    axis_tick_fontsize,
    axis_title_fontsize,
    axis_legend_fontsize,
    figure_suptitle_fontsize,
    income_colors,
    width_mm,
    axes_to_remove,
    relevant_axes,
):
    axes = format_axes(
        axes,
        axis_label_fontsize,
        axis_tick_fontsize,
        axis_title_fontsize,
        axis_legend_fontsize,
        new_labels,
        income_colors,
        axes_to_remove,
        relevant_axes,
    )

    fig = axes.flat[0].get_figure()
    fig.suptitle(
        "Economic Sectors SCI vs CO2 Emissions for all countries in log scale",
        fontsize=figure_suptitle_fontsize,
        y=1.005,
    )

    adjust_figure_size(fig, width_mm)

    plt.tight_layout()
    return axes


def calculate_regression_predictions(
    var_coeffs,
    x_axis,
    group,
    v,
    colors,
    linear_significane_linestyles,
    quadratic_significance_colors,
):
    predictions = []
    for q, color in zip(var_coeffs.columns.get_level_values("Quantile"), colors):
        q_coeffs = var_coeffs.loc[:, [(group, q)]].droplevel([0, 1, 2, 3, 4])
        significances = q_coeffs[(group, q)].str.count("\*")

        x_value = float(q_coeffs.loc[v, :].str.split("(", expand=True)[0].values[0])
        x_sq_value = float(
            q_coeffs.loc[f"{v}_square", :].str.split("(", expand=True)[0].values[0]
        )

        pred = x_value * x_axis + x_sq_value * x_axis**2 + q
        style = linear_significane_linestyles[significances.loc[v]]
        color = quadratic_significance_colors[significances.loc[f"{v}_square"]]

        predictions.append((pred, style, color))
    return predictions


def plot_regression_predictions(ax, predictions, x_axis):
    for pred, style, color in predictions:
        ax.plot(x_axis, pred, color=color, linestyle=style, linewidth=3)


def set_regression_axis_properties(
    ax, letter, v, axis_title_fontsize, axis_tick_fontsize, axis_label_fontsize, group
):
    ax.set_title(
        f"{letter}) {v.replace(' SCI', ' Sector')}:",
        fontsize=axis_title_fontsize,
        loc="left",
    )
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticklabels(
        [f"{round(t, 1)} Q" for t in np.arange(0.1, 1.0, 0.1)],
        fontsize=axis_tick_fontsize,
    )
    ax.set_xticks([])
    ax.set_ylabel(
        r"$\mathrm{CO}_2\,\mathrm{emissions\ (metric\ tons\ per\ capita)}_{log}$",
        fontsize=axis_label_fontsize,
    )
    ax.set_xlabel(r"$\mathrm{SCI}$", fontsize=axis_label_fontsize)
    ax.set_xlim(0, 1.0 if group != "Low income" else 0.6)


def create_regression_legend_handles_labels(
    linear_significane_linestyles, quadratic_significance_colors
):
    significance_levels = [0, 1, 2, 3]
    significance_labels = [
        r"$\chi(-)$",
        r"$\chi({*})$",
        r"$\chi({*}{*})$",
        r"$\chi({*}{*}{*})$",
        r"$\chi^2(-)$",
        r"$\chi^2({*})$",
        r"$\chi^2({*}{*})$",
        r"$\chi^2({*}{*}{*})$",
    ]
    linestyle_handles = [
        mlines.Line2D(
            [0],
            [0],
            color="black",
            linestyle=linear_significane_linestyles[significance],
            linewidth=3,
        )
        for significance in significance_levels
    ] + [
        mlines.Line2D([0], [0], color=color, linestyle="-", linewidth=3)
        for color in quadratic_significance_colors.values()
    ]
    return linestyle_handles, significance_labels


def reorder_regression_legend_handles_labels(linestyle_handles, significance_labels):
    reorder_indices = [0, 4, 1, 5, 2, 6, 3, 7]
    reordered_handles = [linestyle_handles[n] for n in reorder_indices]
    reordered_labels = [significance_labels[v] for v in reorder_indices]
    return reordered_handles, reordered_labels


def regression_coefficients_plots(
    coeffs,
    group,
    sci_labels,
    colors,
    width_mm,
    linear_significance_linestyles,
    quadratic_significance_colors,
    nrows,
    ncols,
    axis_title_fontsize,
    axis_tick_fontsize,
    axis_label_fontsize,
    axis_legend_fontsize,
    figure_suptitle_fontsize,
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 8 * nrows))
    x_axis = (
        np.arange(0, 1.01, 0.01) if group != "Low income" else np.arange(0, 0.61, 0.01)
    )
    group_coeffs = coeffs[group].coefficients_quantiles_table()

    for letter, ax, v in zip(string.ascii_lowercase, axes.flat, sci_labels):
        var_coeffs = group_coeffs.loc[
            [v in i for i in group_coeffs.index.get_level_values("Independent Vars")], :
        ]
        predictions = calculate_regression_predictions(
            var_coeffs,
            x_axis,
            group,
            v,
            colors,
            linear_significance_linestyles,
            quadratic_significance_colors,
        )
        plot_regression_predictions(ax, predictions, x_axis)
        set_regression_axis_properties(
            ax,
            letter,
            v,
            axis_title_fontsize,
            axis_tick_fontsize,
            axis_label_fontsize,
            group,
        )

    axes = remove_ylabel_for_non_leftmost(axes)

    linestyle_handles, significance_labels = create_regression_legend_handles_labels(
        linear_significance_linestyles, quadratic_significance_colors
    )
    reordered_handles, reordered_labels = reorder_regression_legend_handles_labels(
        linestyle_handles, significance_labels
    )
    fig.legend(
        reordered_handles,
        reordered_labels,
        loc="lower center",
        fontsize=axis_legend_fontsize,
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
    )
    axes.flat[0].get_figure().suptitle(
        f"Quantile Regressions of Each Sector for {group} grouping.",
        fontsize=figure_suptitle_fontsize,
        y=1.005,
    )

    adjust_figure_size(fig, width_mm)

    plt.tight_layout()

    return axes


def get_pci_quantiles(pcis, quantiles=[0.05, 0.5, 0.95]):
    return {q: pcis.quantile(q) for q in quantiles}


def pci_quantiles_to_str(pci_quantiles):
    str_mapping = {0.05: "5%", 0.5: "Mean", 0.95: "95%"}
    return {
        q: f"{str_mapping.get(q, q)} PCI:\n {quantile:.3f}"
        for q, quantile in pci_quantiles.items()
    }


def get_closest_products_to_pci_quantiles(products_data, pci_quantiles, n_products=3):
    return {
        q: products_data.iloc[
            (products_data["PCI"] - pci_quantiles[q]).abs().argsort()[:n_products]
        ]["HS6"].values
        for q, pci in pci_quantiles.items()
    }


def plot_sector_products(
    sector_products,
    sector,
    color,
    vline_width,
    products_fontsize,
    ticks_fontsize,
    n_products=3,
    ax=None,
    pci_bins=None,
):
    ax.set_title(sector, fontsize=22)

    pci_quantiles = get_pci_quantiles(sector_products["PCI"])
    pci_quantiles_str = pci_quantiles_to_str(pci_quantiles)
    quantiles_products = get_closest_products_to_pci_quantiles(
        sector_products, pci_quantiles, n_products=n_products
    )

    ax.hist(sector_products["PCI"], bins=pci_bins, color=color, alpha=0.65)

    sector_quantiles_str_loc = {
        "Agriculture": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Electronics & Instruments": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Fishing": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (0.085, 0.75, "left"),
            0.95: (0.085, 0.75, "left"),
        },
        "Food & Beverages": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Iron & Steel": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Machinery": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Metal Products": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Mining & Quarrying": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Other Manufacturing": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Petroleum, Chemicals & Non-Metals": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
        "Textiles & Wearing Apparel": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (0.085, 0.75, "left"),
            0.95: (0.085, 0.75, "left"),
        },
        "Wood & Paper": {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        },
    }
    quantiles_str_loc = (
        sector_quantiles_str_loc[sector]
        if sector != "All Sectors"
        else {
            0.05: (-0.085, 0.75, "right"),
            0.5: (-0.085, 0.75, "right"),
            0.95: (0.085, 0.75, "left"),
        }
    )

    products_valignment = [0.7, 0.475, 0.275]

    for q, quantile in pci_quantiles.items():
        ax.axvline(quantile, color="k", linestyle="--", linewidth=vline_width)
        ax.text(
            quantile + quantiles_str_loc[q][0],
            ax.get_ylim()[1] * quantiles_str_loc[q][1],
            pci_quantiles_str[q],
            color="k",
            ha=quantiles_str_loc[q][2],
            fontsize=products_fontsize,
            fontweight="bold",
        )
        for n, product in enumerate(quantiles_products[q]):
            ax.text(
                quantile + quantiles_str_loc[q][0],
                ax.get_ylim()[1] * products_valignment[n],
                stretch_string(f"- {product}", 18),
                color="k",
                ha=quantiles_str_loc[q][2],
                va="top",
                fontsize=products_fontsize,
            )

    ax.tick_params(axis="x", labelsize=ticks_fontsize)
    ax.tick_params(axis="y", labelsize=ticks_fontsize)

    legend_handles = [mpatches.Patch(color=color, label=sector)]
    ax.legend(handles=legend_handles, fontsize=products_fontsize, loc="upper left")

    return ax


def calculate_pci_bins(products_data, pci_gap_scale, pci_n_bins):
    pci = products_data["PCI"]
    global_min = pci.min() * pci_gap_scale
    global_max = pci.max() * pci_gap_scale
    pci_bins = np.linspace(global_min, global_max, pci_n_bins)
    return pci, global_min, global_max, pci_bins


def set_axes_limits(axes, global_min, global_max):
    for ax in axes.flat:
        ax.set_xlim(global_min, global_max)
    return axes


def plot_background_histograms(axes, pci, pci_bins):
    for ax in axes.flat:
        ax.hist(pci, bins=pci_bins, color="lightgray", alpha=0.65)
        ax.legend().remove()
    return axes


def plot_sector_products_on_axes(
    axes,
    sectors,
    products_data,
    n_products,
    colors,
    vline_width,
    axis_tick_fontsize,
    axis_label_fontsize,
    axis_title_fontsize,
    pci_bins,
):
    for n, (ax, sector) in enumerate(zip(axes.flat, sectors)):
        ax = plot_sector_products(
            products_data.loc[products_data["Sector"] == sector],
            sector,
            colors[sector],
            vline_width,
            axis_tick_fontsize,
            axis_label_fontsize,
            n_products=n_products,
            ax=ax,
            pci_bins=pci_bins,
        )
        ax.set_title("")
        ax.set_title(
            f"{string.ascii_lowercase[n]}) {sector} Sector PCI Histogram:",
            fontsize=axis_title_fontsize,
            loc="left",
        )
    return axes


def set_distribution_axes_labels(axes, axis_label_fontsize):
    for ax in axes.flat:
        ax.set_xlabel("PCI", fontsize=axis_label_fontsize)
        ax.set_ylabel("N° Products", fontsize=axis_label_fontsize)
    return axes


def plot_sectors_pci_distribution(
    sectors,
    products_data,
    n_products,
    pci_gap_scale,
    pci_n_bins,
    colors,
    vline_width,
    axis_tick_fontsize,
    axis_label_fontsize,
    axis_title_fontsize,
    figure_suptitle_fontsize,
    a4_width_inch,
    a4_height_inch,
):
    ncols = 2
    nrows = ceil(len(sectors) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(a4_width_inch * 3 * 1.35, a4_height_inch * 3 * 1.335)
    )

    pci, global_min, global_max, pci_bins = calculate_pci_bins(
        products_data, pci_gap_scale, pci_n_bins
    )
    axes = set_axes_limits(axes, global_min, global_max)

    axes = plot_background_histograms(axes, pci, pci_bins)
    axes = plot_sector_products_on_axes(
        axes,
        sectors,
        products_data,
        n_products,
        colors,
        vline_width,
        axis_tick_fontsize,
        axis_label_fontsize,
        axis_title_fontsize,
        pci_bins,
    )

    axes = set_distribution_axes_labels(axes, axis_label_fontsize)

    axes = remove_ylabel_for_non_leftmost(axes)

    fig.suptitle(
        "Product Complexity Indexes Distribution by Sector",
        fontsize=figure_suptitle_fontsize,
        y=1.0,
    )

    adjust_figure_size(fig, 180.0 * 3.75)
    plt.tight_layout()

    return axes


def extract_svg_country_mapping(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    container = soup.find("div", class_="container")

    svg_country_dict = {}

    for div in container.find_all("div"):
        img = div.find("img")
        country = div.find("p").get_text(strip=True).split(" (")[0].strip()
        country = cc.convert(country, to="name_short")
        svg_file = Path(img["src"]).name
        svg_country_dict[country] = svg_file

    return svg_country_dict


def add_svg_marker(country, x, y, flags_folder_path, flag_country_dict, ax, zoom=0.1):
    flag_path = flags_folder_path / flag_country_dict[country]
    png_data = cairosvg.svg2png(url=str(flag_path))
    image = Image.open(BytesIO(png_data))
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False)
    ax.add_artist(ab)


def scatter_countries_plot(
    data,
    years,
    x,
    y,
    flags_folder,
    income_level,
    all_income,
    flag_country_dict,
    ax=None,
    figsize=(21, 14),
    zoom=0.035,
    title_fontsize=18,
    labels_fontsize=16,
    ticks_fontsize=10,
    marker: Optional[Literal["flag", "name"]] = "flag",
    size_col=None,
    size_bins=5,
):
    plot_data = data.loc[
        (data.index.get_level_values("Year").isin(years) if years else data.index)
    ]
    if income_level != all_income:
        plot_data = plot_data.loc[
            plot_data.index.get_level_values("Income Group") == income_level,
            [y, x, size_col] if size_col else [y, x],
        ]
    else:
        plot_data = plot_data[[y, x, size_col] if size_col else [y, x]]
    size_data = plot_data[[size_col]] if size_col else None
    if size_col:
        quantized_size = (
            size_data.groupby(level="Year")
            .apply(quantize_group, column=size_col, N=size_bins)
            .droplevel(0)
        )
        quantized_size = quantized_size * zoom
    plot_data = plot_data[[x, y]]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(plot_data[x], plot_data[y], alpha=0)
    for idx in plot_data.index:
        x_values = plot_data.loc[idx, x]
        y_values = plot_data.loc[idx, y]
        if marker == "flag":
            add_svg_marker(
                idx[1],
                x_values,
                y_values,
                flags_folder,
                flag_country_dict,
                ax,
                zoom=quantized_size.loc[idx, size_col] if size_col else zoom,
            )
        elif marker == "name":
            country = stretch_string(idx[1], 13)
            text = ax.text(
                x_values,
                y_values,
                country,
                color="k",
                ha="center",
                va="center",
                fontsize=12,
            )
            text.set_bbox(dict(facecolor="white", edgecolor="none", pad=5, alpha=0.66))
    ax.set_xlabel(x, fontsize=labels_fontsize)
    ax.set_ylabel(y, fontsize=labels_fontsize)
    ax.set_facecolor("#c8c8c8")
    ax.tick_params(axis="x", labelsize=ticks_fontsize)
    ax.tick_params(axis="y", labelsize=ticks_fontsize)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="black")
    ax.set_title(f"{y} vs {x} for {income_level} countries", fontsize=title_fontsize)
    return ax


def plot_export_pct_evolution_by_income(
    exports,
    income_level,
    colors=None,
    stacked=True,
    ax=None,
    label_fontsize=16,
    title_fontsize=24,
):
    if income_level != "All income":
        exports_pct = exports.groupby(["Year", "Income Group", "Sector"])[
            "Sector_Exports_pct"
        ].agg(["mean", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        exports_pct.columns = ["mean", "q25", "q75"]
        df_reset = (
            exports_pct.loc[(slice(None), income_level), :].reset_index().reset_index()
        )
    else:
        exports_pct = exports.groupby(["Year", "Sector"])["Sector_Exports_pct"].agg(
            ["mean", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        )
        exports_pct.columns = ["mean", "q25", "q75"]
        df_reset = exports_pct.reset_index().reset_index()

    df_pivot = df_reset.pivot(index="Year", columns="Sector", values="mean")
    df_q05 = df_reset.pivot(index="Year", columns="Sector", values="q25")
    df_q95 = df_reset.pivot(index="Year", columns="Sector", values="q75")

    if colors:
        plot_colors = [colors.get(sector, "#333333") for sector in df_pivot.columns]
    else:
        plot_colors = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(21, 7))
    if stacked:
        ax.stackplot(
            df_pivot.index,
            df_pivot.T,
            labels=df_pivot.columns,
            colors=plot_colors,
            alpha=0.66,
        )
        ax.set_title(
            f"Export Share by Sector for {income_level} Countries",
            fontsize=title_fontsize,
        )
    else:
        for sector in df_pivot.columns:
            ax.plot(
                df_pivot.index,
                df_pivot[sector],
                label=sector,
                color=colors.get(sector, "#333333"),
                marker="o",
            )
            ax.fill_between(
                df_pivot.index,
                df_q05[sector],
                df_q95[sector],
                color=colors.get(sector, "#333333"),
                alpha=0.3,
            )
        ax.set_title(
            f"Time Series Evolution of Export Percentages by Sector for {income_level} Countries"
        )

    ax.set_xlabel("Year", fontsize=label_fontsize)
    ax.set_ylabel("Export Share (%) by Sector", fontsize=label_fontsize)

    ax.set_yticks([0.1 * i for i in range(1, 11)])
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=1.0, color="k")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(1995, 2020)

    return ax
