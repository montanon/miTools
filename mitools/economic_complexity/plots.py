import functools
import random
import re
import statistics
from dataclasses import dataclass
from os import PathLike
from string import ascii_uppercase, digits
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from numpy import ndarray
from pandas import DataFrame, MultiIndex, Series
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tqdm import tqdm

from ..economic_complexity import StringMapper
from ..pandas import idxslice
from ..regressions import generate_hash_from_dataframe
from ..utils import clean_str, stretch_string
from .objects import Product, ProductsBasket

Color = Union[Tuple(int,int,int),str]


def plot_country_sector_distributions_evolution(data, country, years, colors, value=False, kde=False):
    sectors = data['Sector'].unique()
    sectors.sort()
    nrows, ncols = len(years), len(sectors)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(100,30))
    years_string = f'{", ".join([str(year) for year in years[:-1]])} & {years[-1]}'
    title = f'{country} Products by Sector between during {years_string}'
    fig.suptitle(title, fontsize=28, y=0.925)
    fig.subplots_adjust(hspace=0.75)
    plot_data = data[
        data.index.get_level_values('Year').isin(years) & (
            data.index.get_level_values('Country') == country
            )]
    for nrow, year in enumerate(years):
        yearly_data = plot_data[plot_data.index.get_level_values('Year') == year]
        for ncol, sector in enumerate(sectors):
            ax = axes[nrow, ncol]
            products = []
            for idx, row in yearly_data.query(f"Sector == '{sector}' and RCA >= 1.0").iterrows():
                products.append(Product(name=row['HS6'], code=row['HS6'], pci=row['PCI'], value=row['Trade Value']))
            basket = ProductsBasket(products)
            if not value:
                plot_distribution(basket, 
                                  bins=30, 
                                  n=7, 
                                  color=colors[sector], 
                                  title=f'{year}-{country}-{sector}', 
                                  ax=ax)
            else:
                plot_value_distribution(plot_data.query(f"Sector == '{sector}' and RCA >= 1.0 and Year == {year}"), basket, 
                                        bins=30, 
                                        n=7, 
                                        color=colors[sector], 
                                        title=f'{year}-{country}-{sector}', 
                                        ax=ax, 
                                        kde=kde)
    axes = adjust_axes_lims(axes, y=False)
    return axes

def plot_country_sector_distributions_by_year(data, country, year, colors, value=False, kde=False):
    sectors = data['Sector'].unique()
    sectors.sort()
    plot_data = data.query(f'Year == {year} and Country == "{country}"')
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(60,30))
    title =f'{country} Products by Sector for the year {year}'
    fig.suptitle(title, fontsize=28, y=0.925)
    fig.subplots_adjust(hspace=0.75)
    indices = [(r, c) for r in range(nrows) for c in range(ncols)]
    for nrow, ncol in indices:
        sector_idx = nrow*ncols + ncol
        if sector_idx < len(sectors):
            ax = axes[nrow, ncol]
            sector = sectors[sector_idx]
            products = []
            for idx, row in plot_data.query(f"Sector == '{sector}' and RCA >= 1.0 and Year == {year}").iterrows():
                products.append(Product(name=row['HS6'], code=row['HS6'], pci=row['PCI'], value=row['Trade Value']))
            basket = ProductsBasket(products)
            if not value:
                plot_distribution(basket, 
                                  bins=30, 
                                  n=7, 
                                  color=colors[sector], 
                                  title=f'{year}-{country}-{sector}', 
                                  ax=ax)
            else:
                plot_value_distribution(plot_data.query(f"Sector == '{sector}' and RCA >= 1.0 and Year == {year}"), basket, 
                                        bins=30, 
                                        n=4, 
                                        color=colors[sector], 
                                        title=f'{year}-{country}-{sector}', 
                                        ax=ax, 
                                        kde=kde)
    axes = adjust_axes_lims(axes, y=False)
    return axes

def plot_distribution(basket, n=10, bins=100, color=None, title=None, ax=None, info=True, quantiles=True, line_kws=None, **kwargs):
    # Extract pci values
    pcis = [product.pci for product in basket.products]
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 6))
    ax = sns.histplot(pcis, bins=bins, kde=True, color=color, ax=ax, line_kws=line_kws or {}, **kwargs, zorder=1)
    if info:
        mean_val = statistics.mean(pcis) if pcis else np.nan
        std_val = statistics.stdev(pcis) if pcis else np.nan
        min_val = min(pcis) if pcis else np.nan
        max_val = max(pcis) if pcis else np.nan
        # Add text for statistics
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', horizontalalignment='right')
    if quantiles:
        y_text_position = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.075 
        quantile_products = basket.products_closest_to_quantiles(n)
        for product in quantile_products:
            ax.axvline(x=product.pci, linestyle='--', color='gray', zorder=0)
            ax.text(product.pci, y_text_position, product.name,
            rotation=45, fontsize=12, horizontalalignment='right', verticalalignment='top')
    # Final plot adjustments
    if title:
        title = stretch_string(title, 60)
        ax.set_title(title)
    ax.set_xlabel('PCI', ha='left')
    ax.xaxis.set_label_coords(0, -0.02)
    ax.set_ylabel('Frequency')
    return ax

def generate_random_products(num_products):
    products = []
    existing_codes = set()
    existing_names = set()
    while len(products) < num_products:
        code = random.randint(1000, 9999)
        name = ''.join(random.choices(ascii_uppercase + digits, k=5))
        pci = random.uniform(-5.0, 5.0)
        value = random.uniform(1_000, 1_000_000)
        if code not in existing_codes and name not in existing_names:
            products.append(Product(code, name, pci, value))
            existing_codes.add(code)
            existing_names.add(name)
    return products

def plot_value_distribution(data, basket, n=10, bins=100, color=None, title=None, ax=None, info=True, quantiles=True, 
                            major_exports=True, kde=False, weighted_mean=True, bin_step=0.001, line_kws=None, **kwargs):
    data = data.copy(deep=True).reset_index()
    data['Trade Value'] = data['Trade Value'] * 1e-6
    data = data[[c for c in data.columns if c not in [ 'Year',
        'Section', 'Income Group', 'Current Income Group',
        'ECI', 'normECI', 'stdECI', 'normPCI', 'stdPCI', 'Relatedness', 'PCI*RCA', 'normPCI*RCA', 'stdPCI*RCA', 'notRCA',
    'notRCA*Rel*PCI', 'notRCA*Rel*normPCI', 'notRCA*Rel*stdPCI', 'SECI', 'normSECI', 'stdSECI', 'SCI', 'normSCI', 'stdSCI',
    'SCP', 'SCP', 'normSCP', 'stdSCP']]]
    bin_edges = np.arange(data['PCI'].min(), data['PCI'].max() + bin_step, bin_step)
    data['PCIbin'] = pd.cut(data.loc[data['RCA'] != 0.0, 'PCI'], bins=bin_edges)
    agg_bins = data.groupby('PCIbin')[['Trade Value']].sum().sort_index()
    agg_bins.index = pd.IntervalIndex(agg_bins.index.categories).mid
    agg_bins['Trade Value'] = agg_bins['Trade Value']
    # Extract pci values
    pcis = [product.pci for product in basket.products]
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 6))

    scatter_plot = True
    if not scatter_plot:
        ax.bar(agg_bins.index, agg_bins['Trade Value'], width=bin_step, color=color, edgecolor='k', **kwargs, zorder=1)
    else:
        ax = sns.scatterplot(data, x='PCI', y='Trade Value', color=color, ax=ax)
        ax.set_ylim(-1.0, ax.get_ylim()[1]*1.1)
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
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', horizontalalignment='right', fontsize=16)
    if quantiles:
        y_text_position = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.075  # 5% below the x-axis
        quantile_products = basket.products_closest_to_quantiles(n)
        for product in quantile_products:
            ax.axvline(x=product.pci, linestyle='--', color='gray', zorder=0, alpha=0.85)
            ax.text(product.pci, y_text_position, stretch_string(product.name, 35),
            rotation=45, fontsize=16, horizontalalignment='right', verticalalignment='top')

    if major_exports:
        x_min, x_max = ax.get_xlim()
        x_mean = (x_min + x_max) / 2
        if not scatter_plot:
            top_exports = agg_bins['Trade Value'].nlargest(2)
        else:
            top_exports = data.nlargest(2, 'Trade Value')
        if not scatter_plot:
            if top_exports.iloc[0]*0.5 > top_exports.iloc[1]:
                top_exports = top_exports.iloc[[0]]
        for n, (pci_value, trade_value) in top_exports[['PCI', 'Trade Value']].iterrows():
            if not scatter_plot:
                in_pci_range = (data['PCIbin'].apply(check_interval, args=(pci_value,)) == True)
                product = data.loc[in_pci_range]
                total_products = product.shape[0]
                product = product.sort_values('Trade Value').iloc[-1]
            else:
                product = top_exports.loc[n]
            product_name = product['HS6']
            product_value = product['Trade Value']
            product_pct = round(100 * (product_value / trade_value), 2)
            product_name = f"{stretch_string(product_name, 40)} \n"
            if not scatter_plot:
                product_name += f"[{product_pct}% of {total_products} products]"
            h_al = 'right' if pci_value > x_mean else 'left'
            ax.text(pci_value, trade_value, product_name,
                rotation=0, fontsize=12, horizontalalignment=h_al, verticalalignment='bottom', zorder=10)
            
    if weighted_mean:
        weighted_value = sum(product.pci * product.value for product in basket.products)
        weighted_value /= sum(product.value for product in basket.products)
        ax.axvline(x=weighted_value, linestyle='--', linewidth=5.0, color='green', zorder=10, alpha=0.75)
        ax.text(weighted_value*1.1, ax.get_ylim()[1]*0.55, f"SECI={round(weighted_value, 3)}", fontsize=24, zorder=99, 
                verticalalignment='bottom', horizontalalignment='left', color='green', alpha=0.75)
            
    # Final plot adjustments
    if title:
        title = stretch_string(title, 30)
        ax.set_title(title)
    ax.set_xlabel('PCI', ha='left')
    ax.xaxis.set_label_coords(0, -0.02)
    ax.set_ylabel('Trade Value (million USD)')
    return ax

def check_interval(interval, value):
    if pd.isna(interval):
        return False
    return interval.left <= value < interval.right

def custom_agg_rca(group, n=5):
    return 1.0 if group.sum() >= n else 0.0

def get_axes_limits(axes, get_lim_func):
    lim_min, lim_max = float('inf'), float('-inf')
    for ax in axes:
        if not is_ax_empty(ax):
            lim1, lim2 = get_lim_func(ax)
            lim_min, lim_max = min(lim1, lim_min), max(lim2, lim_max)
    return lim_min, lim_max

def set_axes_limits(axes, set_lim_func, lim_min, lim_max):
    for ax in axes:
        set_lim_func(ax, (lim_min, lim_max))

def adjust_axes_lims(axes, mode='all', x=True, y=True):
    if not (x or y):
        return axes
    nrows, ncols = axes.shape
    if mode == 'all':
        if x:
            xlim_min, xlim_max = get_axes_limits(axes.flat, lambda ax: ax.get_xlim())
            set_axes_limits(axes.flat, lambda ax, lim: ax.set_xlim(*lim), xlim_min, xlim_max)
        if y:
            ylim_min, ylim_max = get_axes_limits(axes.flat, lambda ax: ax.get_ylim())
            set_axes_limits(axes.flat, lambda ax, lim: ax.set_ylim(*lim), ylim_min, ylim_max)
    elif mode == 'rows':
        for i in range(nrows):
            if x:
                xlim_min, xlim_max = get_axes_limits(axes[i, :], lambda ax: ax.get_xlim())
                set_axes_limits(axes[i, :], lambda ax, lim: ax.set_xlim(*lim), xlim_min, xlim_max)
            if y:
                ylim_min, ylim_max = get_axes_limits(axes[i, :], lambda ax: ax.get_ylim())
                set_axes_limits(axes[i, :], lambda ax, lim: ax.set_ylim(*lim), ylim_min, ylim_max)
    elif mode == 'columns':
        for j in range(ncols):
            if x:
                xlim_min, xlim_max = get_axes_limits(axes[:, j], lambda ax: ax.get_xlim())
                set_axes_limits(axes[:, j], lambda ax, lim: ax.set_xlim(*lim), xlim_min, xlim_max)
            if y:
                ylim_min, ylim_max = get_axes_limits(axes[:, j], lambda ax: ax.get_ylim())
                set_axes_limits(axes[:, j], lambda ax, lim: ax.set_ylim(*lim), ylim_min, ylim_max)
    return axes

def is_ax_empty(ax):
    return not (ax.lines or ax.patches or ax.collections)

def is_axes_empty(ax: Axes):
    return (len(ax.get_lines()) == 0 and
            len(ax.patches) == 0 and
            len(ax.texts) == 0 and
            ax.get_legend() is None and
            not ax.get_xlabel() and
            not ax.get_ylabel())

def plot_country_eci_indicator_scatter(country_data: DataFrame, 
                                       x_var_col: str, 
                                       y_var_col: str, 
                                       groups: List[str],
                                       groups_col: Optional[str]='Income Group',
                                       time_col: Optional[str]='Year',
                                       n_steps: Optional[int]=1,
                                       year_labels: Optional[bool]=True, 
                                       color: Optional[Color]=None, 
                                       groups_colors: Optional[Dict[str, Color]]=None, 
                                       figsize: Optional[Tuple(float, float)]=(9, 9), 
                                       arrows: Optional[bool]=True, 
                                       arrow_style: Optional[ArrowStyle]=None, 
                                       arrow_kwargs: Optional[Dict[str, Any]]=None, 
                                       set_arrows_ax_limits: Optional[bool]=True, 
                                       marker_kwargs: Optional[Dict[str, Any]]=None,
                                       ax: Optional[Axes]=None
                                       ):
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
    if marker_kwargs is None:
        marker_kwargs = {'marker': 'o', 'markeredgewidth': 2, 'markersize': 10, 'label_fontsize': 12}
    if arrow_style is None:
        arrow_style = ArrowStyle("Fancy", head_length=10, head_width=5, tail_width=.4)
    if arrow_kwargs is None:
        arrow_kwargs = dict(connectionstyle='arc3', color='grey', linewidth=1, linestyle=':', alpha=0.75)
    years = country_data.index.get_level_values(time_col)[::n_steps]
    steps_index = country_data.index[::n_steps]
    groups = [level for level in groups if level in country_data.loc[steps_index, :].index.get_level_values(groups_col)]
    for group in groups:
        group_idxslice = idxslice(country_data, level=groups_col, value=group, axis=0)
        ax.plot(country_data.loc[steps_index, :].loc[group_idxslice, x_var_col].values, 
                country_data.loc[steps_index, :].loc[group_idxslice, y_var_col].values, 
                markeredgecolor=color if color else 'k',
                markerfacecolor=groups_colors[group] if groups_colors else 'white', 
                marker=marker_kwargs['marker'],
                markeredgewidth=marker_kwargs['markeredgewidth'],
                markersize=marker_kwargs['markersize'],
                linestyle='',
                alpha=0.75
                )
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    x = country_data[x_var_col].values[::n_steps]
    y = country_data[y_var_col].values[::n_steps]
    if arrows:
        for i in range(len(x) - 1):
            if any([np.isnan(x[i]), np.isnan(x[i+1]), np.isnan(y[i]), np.isnan(y[i+1])]):
                continue
            arrow = FancyArrowPatch((x[i], y[i]), (x[i+1], y[i+1]),
                                    arrowstyle=arrow_style, 
                                    connectionstyle=arrow_kwargs['connectionstyle'], 
                                    color=arrow_kwargs['color'], 
                                    linewidth=arrow_kwargs['linewidth'], 
                                    linestyle=arrow_kwargs['linestyle'], 
                                    alpha=arrow_kwargs['alpha'])
            ax.add_patch(arrow)
        if set_arrows_ax_limits:
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims) 
    if year_labels:
        offset = 0.01 * (y_lims[1] - y_lims[0])  # Offset for annotation so it doesn't sit right on the point
        for xi, yi, year in zip(x, y, years):
            ax.annotate(str(year), (xi, yi - offset), textcoords="offset points", xytext=(0, -10), ha='center', 
                        fontsize=marker_kwargs['label_fontsize'])
    ax.set_ylabel(y_var_col)
    ax.set_xlabel(x_var_col)
    return ax

def plot_country_ecis_indicator_scatter(country_data: DataFrame,
                                        x_vars_cols: List[str],
                                        y_var_col: str,
                                        name_tag: str,
                                        groups: List[str],
                                        groups_col: Optional[str]='Income Group',
                                        entity_col: Optional[str]='Country',
                                        time_col: Optional[str]='Year',
                                        n_steps: Optional[int]=1,
                                        ncols: Optional[int]=3, 
                                        year_labels: Optional[bool]=True, 
                                        colors: Optional[List[Color]]=None,
                                        groups_colors: Optional[Dict[str, Color]]=None, 
                                        figsize: Optional[Tuple(float, float)]=(9, 9), 
                                        arrows: Optional[bool]=True, 
                                        arrow_style: Optional[ArrowStyle]=None, 
                                        arrow_kwargs: Optional[Dict[str, Any]]=None,
                                        set_arrows_ax_limits: Optional[bool]=True, 
                                        marker_kwargs: Optional[Dict[str, Any]]=None, 
                                        axes: Optional[Axes]=None
                                        ):
    entity = country_data.index.get_level_values(entity_col).unique()[0]
    nrows = (len(x_vars_cols) + 1) // ncols
    if axes is None:
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    for n, (ax, x_var_col) in enumerate(zip(axes.flat, x_vars_cols)):
        ax = plot_country_eci_indicator_scatter(country_data=country_data, 
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
    axes.flat[0].figure.suptitle(f"{entity} {name_tag}s vs {y_var_col} Evolution", 
                 fontsize=24, 
                 y=0.925, 
                 verticalalignment='bottom', 
                 horizontalalignment='center'
                                 )
    last_ax = axes.flat[-1]
    if is_axes_empty(last_ax) and groups_colors is not None:
        last_ax.cla()
        last_ax.set_xticks([]) 
        last_ax.set_yticks([]) 
        last_ax.axis('off') 
        legend_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                        markersize=10, markeredgecolor='k', label=label) for label, color in groups_colors.items()]
        last_ax.legend(handles=legend_handles, fontsize=16, loc='center left', ncols=1)
    return axes

def plot_countries_ecis_indicator_scatter(data: DataFrame, 
                                          entities: List[str], 
                                          x_vars_cols: List[str], 
                                          y_var_col: str, 
                                          name_tag: str,
                                          groups: List[str],
                                          groups_col: Optional[str]='Income Group',
                                          entity_col: Optional[str]='Country',
                                          time_col: Optional[str]='Year', 
                                          n_steps: Optional[int]=1, 
                                          ncols: Optional[int]=3,
                                          year_labels: Optional[bool]=False, 
                                          colors: Optional[List[Color]]=None, 
                                          groups_colors: Optional[Dict[str, Color]]=None, 
                                          figsize: Optional[Tuple(float, float)]=(7,7), 
                                          arrows: Optional[bool]=False,
                                          arrow_style: Optional[bool]=None, 
                                          arrow_kwargs: Optional[Dict[str, Any]]=None,
                                          set_arrows_ax_limits: Optional[bool]=False,
                                          marker_kwargs: Optional[Dict[str, Any]]=None, 
                                          axes: Optional[Axes]=None
                                          ):
    nrows = (len(x_vars_cols) + 1) // ncols
    if axes is None:
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    for entity in entities:
        country_data = data.query(f'{entity_col} == @entity')
        axes = plot_country_ecis_indicator_scatter(country_data=country_data, 
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
                                                   axes=axes
                                                   )
    axes.flat[0].figure.suptitle(f'Countries {name_tag} vs {y_var_col}', fontsize=22, y=0.9, 
                    verticalalignment='bottom', 
                    horizontalalignment='center')
    return axes

def prepare_regression_data(data: DataFrame, 
                            dependent_variable: str, 
                            independent_variables: List[str], 
                            control_variables: Optional[List[str]]=None,
                            str_mapper: Optional[StringMapper]=None, 
                            ) -> Tuple[DataFrame, str, List[str], List[str]]:
    if control_variables is None:
        control_variables = []
    regression_data = data.loc[:, [dependent_variable, *independent_variables, *control_variables]].copy(deep=True)
    if str_mapper is not None:
        regression_data.columns = [str_mapper.uglify_str(var) for var in regression_data.columns]
        dependent_variable = str_mapper.uglify_str(dependent_variable)
        independent_variables = [str_mapper.uglify_str(var) for var in independent_variables]
        control_variables = [str_mapper.uglify_str(var) for var in control_variables]
    return regression_data, dependent_variable, independent_variables, control_variables
    
def get_quantile_regression_results(data: DataFrame, 
                                    dependent_variable: str, 
                                    independent_variables: List[str], 
                                    control_variables: Optional[List[str]]=None, 
                                    quantiles: Optional[List[float]]=None, 
                                    quadratic=False, 
                                    max_iter: Optional[int]=2_500
                                    ) -> Dict[float, RegressionResultsWrapper]:
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    formula_terms = independent_variables.copy()
    if quadratic:
        formula_terms += [f"I({var} ** 2)" for var in formula_terms]
    if control_variables: 
        formula_terms += control_variables
    formula = f"{dependent_variable} ~ " + " + ".join(formula_terms)
    results = {q: smf.quantreg(formula, data).fit(q=q, max_iter=max_iter) for q in quantiles}
    return results

def quantile_regression_value(row: Series) -> Series:
    coeff = round(row['coeff'], 5)
    t_value = round(row['t'], 5)
    p_value = row['P>|t|']
    if p_value < 0.001:
        return f"{coeff}({t_value})***"
    elif p_value < 0.01:
        return f"{coeff}({t_value})**"
    elif p_value < 0.05:
        return f"{coeff}({t_value})*"
    else:
        return f"{coeff}({t_value})"

@dataclass(frozen=True)    
class QuantileRegStrs:
    UNNAMED: str = 'Unnamed: 0'
    COEF: str = 'coef'
    T_VALUE: str = 't'
    P_VALUE: str = 'P>|t|'
    VALUE: str = 'Value'
    QUANTILE: str = 'Quantile'
    INDEPENDENT_VARS: str = 'Independent Vars'
    REGRESSION_TYPE: str = 'Regression Type'
    REGRESSION_DEGREE: str = 'Regression Degree'
    DEPENDENT_VAR: str = 'Dependent Var'
    VARIABLE_TYPE: str = 'Variable Type'
    EXOG_VAR: str = 'Exog'
    CONTROL_VAR: str = 'Control'
    ID: str = 'Id'
    QUADRATIC_REG: str = 'quadratic'
    LINEAR_REG: str = 'linear'
    QUADRATIC_VAR_SUFFIX: str = ' ** 2'
    INDEPENDENT_VARS_PATTERN: str = r'^I\((.*)\)$'
    
def get_quantile_regression_results_coeffs(results: Dict[int, RegressionResultsWrapper],
                                           independent_variables: List[str]
                                           ) -> DataFrame:
    regression_coeffs = []
    for q, result in results.items():
        coeffs = pd.concat(pd.read_html(result.summary().tables[1].as_html(), header=0))
        coeffs = coeffs.set_index(QuantileRegStrs.UNNAMED)
        coeffs[QuantileRegStrs.VALUE] = coeffs[
            [QuantileRegStrs.COEF, QuantileRegStrs.T_VALUE, QuantileRegStrs.P_VALUE]
                            ].apply(quantile_regression_value, axis=1)
        coeffs = coeffs[[QuantileRegStrs.VALUE]]
        coeffs.columns = pd.MultiIndex.from_tuples([(str(q), c) for c in coeffs.columns])
        regression_coeffs.append(coeffs)
    regression_coeffs = pd.concat(regression_coeffs, axis=1)
    regression_coeffs = regression_coeffs.reset_index().melt(id_vars=QuantileRegStrs.UNNAMED, 
                                                             var_name=[QuantileRegStrs.QUANTILE], 
                                                             value_name=QuantileRegStrs.VALUE)
    regression_coeffs.columns = [QuantileRegStrs.INDEPENDENT_VARS, *regression_coeffs.columns[1:]]
    regression_coeffs[QuantileRegStrs.INDEPENDENT_VARS] = (regression_coeffs[QuantileRegStrs.INDEPENDENT_VARS]
                                   .replace(QuantileRegStrs.INDEPENDENT_VARS_PATTERN, r'\1', regex=True)
                                            )
    regression_coeffs[QuantileRegStrs.REGRESSION_TYPE] = type(results[q].model).__name__
    reg_degree = QuantileRegStrs.QUADRATIC_REG if all(
        [f"{var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}" in regression_coeffs[
            QuantileRegStrs.INDEPENDENT_VARS].values for var in independent_variables]
        ) else QuantileRegStrs.LINEAR_REG
    regression_coeffs[QuantileRegStrs.REGRESSION_DEGREE] = reg_degree
    regression_coeffs[QuantileRegStrs.DEPENDENT_VAR] = results[q].model.endog_names
    regression_coeffs[QuantileRegStrs.VARIABLE_TYPE] = regression_coeffs[QuantileRegStrs.INDEPENDENT_VARS].apply(
        lambda x: QuantileRegStrs.EXOG_VAR if x.replace(
            QuantileRegStrs.QUADRATIC_VAR_SUFFIX, ''
            ) in independent_variables else QuantileRegStrs.CONTROL_VAR
        )
    regression_coeffs = regression_coeffs.sort_values(
        by=[QuantileRegStrs.VARIABLE_TYPE, QuantileRegStrs.INDEPENDENT_VARS, QuantileRegStrs.QUANTILE],
        ascending=[False, True, True]
        )
    regression_coeffs[QuantileRegStrs.QUANTILE] = regression_coeffs[QuantileRegStrs.QUANTILE].astype(float)
    regression_coeffs = regression_coeffs.set_index([QuantileRegStrs.REGRESSION_TYPE, 
                                             QuantileRegStrs.REGRESSION_DEGREE, 
                                             QuantileRegStrs.DEPENDENT_VAR, 
                                             QuantileRegStrs.VARIABLE_TYPE, 
                                             QuantileRegStrs.INDEPENDENT_VARS, 
                                             QuantileRegStrs.QUANTILE]
                                                    )
    regression_coeffs[QuantileRegStrs.ID] = generate_hash_from_dataframe(regression_coeffs, [
        QuantileRegStrs.REGRESSION_TYPE, 
        QuantileRegStrs.REGRESSION_DEGREE,
        QuantileRegStrs.DEPENDENT_VAR,
        QuantileRegStrs.INDEPENDENT_VARS
        ],
        length=12
        )
    regression_coeffs = regression_coeffs.set_index(QuantileRegStrs.ID, append=True)
    regression_coeffs = regression_coeffs.reorder_levels(
        [regression_coeffs.index.names[-1]] + regression_coeffs.index.names[:-1])
    return regression_coeffs

def get_quantile_regression_results_stats(results: Dict[int, RegressionResultsWrapper]) -> DataFrame:
    regression_stats = []
    for q, result in results.items():
        stats = result.summary().tables[0].as_html()
        stats = pd.read_html(stats, index_col=0)[0].reset_index()
        stats = pd.concat(
            [stats.iloc[:-1, :2], stats.iloc[:, 2:].rename(columns={2: 0, 3: 1})],
            axis=0, ignore_index=True)
        stats.columns = ['Stat', 'Value']
        stats['Quantile'] = q
        stats = stats.set_index(['Quantile', 'Stat'])
        regression_stats.append(stats)
    regression_stats = pd.concat(regression_stats, axis=0)
    return regression_stats

def get_quantile_regression_predictions_by_group(regression_data: DataFrame, regression_coeffs: DataFrame, group: str) -> Tuple[DataFrame, DataFrame]:
    group_data = regression_data.loc[
        idxslice(regression_data, level='Income Group', value=group, axis=0), :
        ] if group != 'All income' else regression_data
    dependent_var = regression_coeffs.index.get_level_values('Dep Var').unique()[0]
    quantiles = regression_coeffs.index.get_level_values('Quantile').unique()
    independent_vars = [var for var in regression_coeffs.index.get_level_values('Indep Vars').unique() if var != 'Intercept']
    x_values = DataFrame({var: np.linspace(group_data[var].min(), group_data[var].max(), 100) for var in independent_vars if ' ** 2' not in var})
    predictions = []
    significances = []
    columns = []
    for var in independent_vars:
        quadratic = ' ** 2' in var
        x_var_values = x_values[var.replace(' ** 2', '')]
        var_values = [var, 'Intercept'] if not quadratic else [var.replace(' ** 2', ''), var, 'Intercept']
        vars_idx = idxslice(regression_coeffs, level='Indep Vars', value=var_values, axis=0)
        var_coeffs = regression_coeffs.loc[vars_idx, :]
        for quantile in quantiles:
            quantile_idx = idxslice(var_coeffs, level='Quantile', value=quantile, axis=0)
            values = var_coeffs.loc[quantile_idx, group].drop_duplicates()
            values = values.values
            significance = ','.join([match.group() if match else '-' for match in [re.search(r"\*+$", val) for val in values[:-1]]])
            coeffs = [float(re.search(r"([-\d.]+)\(", val).group(1)) for val in values]
            prediction = coeffs[-1] + coeffs[0] * x_var_values
            if quadratic:
                prediction += coeffs[1] * x_var_values ** 2
            predictions.append(prediction)
            significances.append(significance)
            columns.append((dependent_var, var, quantile))
    predictions = DataFrame(predictions, index=MultiIndex.from_tuples(columns)).T
    significances = DataFrame(significances, index=MultiIndex.from_tuples(columns)).T
    return predictions, significances, x_values

def prettify_index_level(mapper: StringMapper, pattern: str, level: str, level_name: str, levels_to_remap: List[str]) -> str:
    if level_name in levels_to_remap:
        return level.map(lambda x: prettify_with_pattern(x, mapper, pattern))
    return level

def prettify_with_pattern(string: str, mapper: StringMapper, pattern: str) -> str:
    base_string, pattern_str, _ = string.partition(pattern)
    remapped_base = mapper.prettify_str(base_string)
    return f"{remapped_base}{pattern}" if pattern_str else remapped_base

def plot_income_levels_ecis_indicator_scatter(data, dependent_var, income_levels, eci_type, colors, income_colors, figsize=(9, 7)):

    eci_indicators = [c for c in data.columns if c.endswith(f' {eci_type}')]

    nrows = len(eci_indicators)
    ncols = len(income_levels)
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))

    for n, income_level in enumerate(income_levels):

        if income_level != 'All income':
            _data = data.loc[pd.IndexSlice[:,:,income_level,:,:], :].copy(deep=True)
        else:
            _data = data
        countries = _data.index.get_level_values('Country').unique().tolist()
        plot_countries_ecis_indicator_scatter(_data, 
                                            countries, 
                                            eci_type, 
                                            eci_indicators, 
                                            dependent_var, 
                                            colors=colors, 
                                            income_colors=income_colors, 
                                            marker_kwargs=None, 
                                            ncols=1, 
                                            figsize=(7,7), 
                                            arrow_style=None, 
                                            arrow_kwargs=None,
                                            n_steps=1,
                                            axes=axes.flat[n::ncols]
                                              )
    axes = adjust_axes_lims(axes, mode='rows', x=True, y=True)
    return axes

def get_regression_predictions(data: DataFrame, regression_coeffs: DataFrame, groups: List[str],
                               ) -> Tuple[DataFrame, DataFrame, DataFrame]:
    predictions, significances, x_values = [], [], []
    for group in groups:
        group_predictions, group_significances, group_x_values = get_quantile_regression_predictions_by_group(
            data,
            regression_coeffs,
            group)
        group_predictions.columns = pd.MultiIndex.from_tuples([(group,) + c for c in group_predictions.columns])
        group_significances.columns = pd.MultiIndex.from_tuples([(group,) + c for c in group_significances.columns])
        group_x_values.columns = pd.MultiIndex.from_tuples([(group, c) for c in group_x_values.columns])
        predictions.append(group_predictions)
        significances.append(group_significances)
        x_values.append(group_x_values)
    predictions = pd.concat(predictions, axis=1)
    significances = pd.concat(significances, axis=1)
    x_values = pd.concat(x_values, axis=1)
    return predictions, significances, x_values

def plot_regression_predictions_by_group(variables, predictions, significances, x_values, dependent_variable, groups, quantiles, quadratic, significance_plot_kwargs, axes=None):
    if axes is None:
        _, axes = plt.subplots()
    for n, group in enumerate(groups):
        current_axes = axes.flat[n::len(groups)]
        for ax, var in zip(current_axes, variables):
            for line in ax.lines:
                line.set_alpha(0.10)
            for quantile in quantiles:
                independent_var = f"{var} ** 2" if quadratic and f"{var} ** 2" in predictions.columns.get_level_values(-2).unique() else var
                x_vals = x_values[group][var]
                preds = predictions[group][(dependent_variable, independent_var, quantile)]
                significance = significances[group][(dependent_variable, independent_var, quantile)].values[0]
                ax.plot(x_vals, preds, **significance_plot_kwargs[significance])
                text = ax.text(x_vals.iloc[-1]*1.0025, preds.iloc[-1], f'{quantile}Q', horizontalalignment='left', zorder=99, fontsize=10, color='k')
            ax.figure.canvas.draw()
            bbox = text.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            bbox_transformed = bbox.transformed(ax.transData.inverted())
            right_x = bbox_transformed.x1
            ax.set_xlim(ax.get_xlim()[0], right_x)
            ax.autoscale(enable=True, axis='both', tight=False)
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
    return axes

def create_regression_file_paths(eci_type_folder, eci_type, regression_id):
    main_plot = eci_type_folder / "regression_data.png"
    regression_plot = eci_type_folder / f"{regression_id}_regression.png"
    return main_plot, regression_plot

def create_regression_plots(data: DataFrame, 
                            dependent_variable: str, 
                            independent_variables: List[str],
                            regression_coeffs: DataFrame,
                            regression_id: str,
                            folder: PathLike,
                            name_tag: str,
                            significance_plot_kwargs: Dict[str, Dict[str,Any]], 
                            indep_vars_colors: List[Color],
                            groups,
                            groups_colors,
                            recalculate
                            ):
        quantiles = regression_coeffs.index.get_level_values('Quantile').unique()
        quadratic = regression_coeffs.index.get_level_values('Reg Degree').unique()[0] == 'quadratic'
        main_plot = folder / "regression_data.png"
        regression_plot = folder / f"{regression_id}_regression.png"
        if not main_plot.exists() or not regression_plot.exists() or recalculate:
            predictions, significances, x_values = get_regression_predictions(data, regression_coeffs, groups)
            axes = plot_income_levels_ecis_indicator_scatter(data, dependent_variable, groups, eci_type, colors, groups_colors, figsize=(9, 7))
            if not main_plot.exists() or recalculate:
                axes.flat[0].figure.savefig(main_plot)
        
            if not regression_plot.exists() or recalculate:
                plot_regression_predictions_by_group(eci_indicators, predictions, significances, 
                                                        x_values, dependent_variable, groups, 
                                                        quantiles, quadratic, significance_plot_kwargs, 
                                                        axes=axes)
                axes.flat[0].figure.savefig(regression_plot)
                plt.close()

def plot_regressions_predictions(data: DataFrame, 
                                 dependent_variables: List[str], 
                                 independent_variables: Dict[str, List[str]], 
                                 regressions_folder: PathLike, 
                                 groups: List[str], 
                                 significance_plot_kwargs: Dict[str, Dict[str, Any]],
                                 indep_vars_colors: Optional[List[Color]]=None, 
                                 groups_colors: Optional[Dict[str, Color]]=None, 
                                 recalculate: Optional[bool]=False
                                 ):
    for dependent_variable in tqdm(dependent_variables, desc='Dependent Variables', position=0, leave=True):
        dep_var_folder = regressions_folder / dependent_variable.replace('/', '')
        if not dep_var_folder.exists(): 
            dep_var_folder.mkdir(exist_ok=True)
        for name_tag, independent_vars in tqdm(
            independent_variables.items(), desc='Eci Types', position=1, leave=False
            ):
            type_folder = dep_var_folder / name_tag
            if not type_folder.exists(): 
                type_folder.mkdir(exist_ok=True)
            regressions_coeffs_path = type_folder / f"{name_tag}_regressions.parquet"       
            if regressions_coeffs_path.exists():
                regressions_coeffs = pd.read_parquet(regressions_coeffs_path)
                for regression_id, regression_coeffs in tqdm(
                    regressions_coeffs.groupby('Id', axis=0), desc='Plots', position=2, leave=False
                    ):
                    create_regression_plots(data=data, 
                                            dependent_variable=dependent_variable, 
                                            independent_variables=independent_vars,
                                            regression_coeffs=regression_coeffs,
                                            regression_id=regression_id,
                                            groups=groups,
                                            folder=type_folder,
                                            name_tag=name_tag,
                                            significance_plot_kwargs=significance_plot_kwargs,
                                            indep_vars_colors=indep_vars_colors,
                                            groups_colors=groups_colors, 
                                            recalculate=recalculate
                                            )



                
                
                    
