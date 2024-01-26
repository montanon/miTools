import functools
import random
import re
import statistics
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

def is_ax_empty(ax):
    return not (ax.lines or ax.patches or ax.collections)

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

def plot_country_eci_indicator_scatter(country_data, x_col, y_col, color=None, income_colors=None, marker_kwargs=None, ax=None, 
                               arrows=True, year_labels=True, figsize=(9, 9), arrow_style=None, arrow_kwargs=None, n_steps=1):
    
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
    if marker_kwargs is None:
        marker_kwargs = {'marker': 'o', 'markeredgewidth': 2, 'markersize': 10, 'label_fontsize': 12}
    if arrow_style is None:
        arrow_style = ArrowStyle("Fancy", head_length=10, head_width=5, tail_width=.4)
    if arrow_kwargs is None:
        arrow_kwargs = dict(connectionstyle='arc3', color='grey', linewidth=1, linestyle=':', alpha=0.75)

    years = country_data.index.get_level_values('Year')[::n_steps]
    steps_index = country_data.index[::n_steps]
    income_levels = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    income_levels = [level for level in income_levels if level in country_data.loc[steps_index, :].index.get_level_values('Income Group')]
    for income_level in income_levels:
        ax.plot(country_data.loc[steps_index, :].loc[pd.IndexSlice[:,:,income_level,:,:], x_col].values, 
                country_data.loc[steps_index, :].loc[pd.IndexSlice[:,:,income_level,:,:], y_col].values, 
                marker=marker_kwargs['marker'], 
                markeredgecolor=color if color else 'k', 
                markeredgewidth=marker_kwargs['markeredgewidth'], 
                markerfacecolor=income_colors[income_level] if income_colors else 'white', 
                markersize=marker_kwargs['markersize'], 
                linestyle='',
                alpha=0.75
                )
        
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    x = country_data[x_col].values[::n_steps]
    y = country_data[y_col].values[::n_steps]
        
    if arrows:
        for i in range(len(x) - 1):
            try:
                arrow = FancyArrowPatch((x[i], y[i]), (x[i+1], y[i+1]),
                                        arrowstyle=arrow_style, 
                                        connectionstyle=arrow_kwargs['connectionstyle'], 
                                        color=arrow_kwargs['color'], 
                                        linewidth=arrow_kwargs['linewidth'], 
                                        linestyle=arrow_kwargs['linestyle'], 
                                        alpha=arrow_kwargs['alpha'])
                ax.add_patch(arrow)
            except Exception:
                pass
        #ax.set_xlim(x_lims)
        #ax.set_ylim(y_lims)    
    
    if year_labels:
        offset = 0.01 * (y_lims[1] - y_lims[0])  # Offset for annotation so it doesn't sit right on the point
        for xi, yi, year in zip(x, y, years):
            ax.annotate(str(year), (xi, yi - offset), textcoords="offset points", xytext=(0, -10), ha='center', 
                        fontsize=marker_kwargs['label_fontsize'])
            
    ax.set_ylabel(y_col)
    ax.set_xlabel(x_col)

    return ax

def plot_country_ecis_indicator_scatter(data, x_cols, y_col, colors, income_colors=None, marker_kwargs=None, ncols=3, 
                                figsize=(7,7), arrows=True, arrow_style=None, arrow_kwargs=None, year_labels=True, 
                                axes=None, n_steps=1):
    
    country = data.index.get_level_values('Country').unique()[0]
    nrows = (len(x_cols) + 1) // ncols
    x_type = x_cols[0].split(' ')[-1]
    if axes is None:
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    for ax, x_col in zip(axes.flat, x_cols):
        ax = plot_country_eci_indicator_scatter(data, 
                                        x_col, 
                                        y_col, 
                                        color=colors[x_col.replace(f' {x_type}', '')],
                                        income_colors=income_colors,
                                        marker_kwargs=marker_kwargs, 
                                        ax=ax,
                                        arrows=arrows,
                                        year_labels=year_labels,
                                        arrow_style=arrow_style,
                                        arrow_kwargs=arrow_kwargs,
                                        n_steps=n_steps
                                                )
    axes.flat[0].figure.suptitle(f"{country} {x_type}s vs {y_col} Evolution", 
                 fontsize=24, 
                 y=0.925, 
                 verticalalignment='bottom', 
                 horizontalalignment='center'
                                 )
    
    last_ax = axes.flat[-1]
    if is_axes_empty(last_ax) and income_colors is not None:  # Check if last_ax is empty
        last_ax.cla()
        last_ax.set_xticks([])  # Remove x-axis ticks
        last_ax.set_yticks([])  # Remove y-axis ticks
        last_ax.axis('off') 

        legend_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                        markersize=10, markeredgecolor='k', label=label) for label, color in income_colors.items()]
        last_ax.legend(handles=legend_handles, fontsize=16, loc='center left', ncols=1)

    return axes

def is_axes_empty(ax: Axes):
    return (len(ax.get_lines()) == 0 and
            len(ax.patches) == 0 and
            len(ax.texts) == 0 and
            ax.get_legend() is None and
            not ax.get_xlabel() and
            not ax.get_ylabel())

def plot_countries_ecis_indicator_scatter(data, countries, eci_type, x_cols, y_col, colors=None, income_colors=None, 
                                          marker_kwargs=None, ncols=3, figsize=(7,7), arrow_style=None, arrow_kwargs=None,
                                          n_steps=1, axes=None):
    
    nrows = (len(x_cols) + 1) // ncols
    if axes is None:
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))

    for country in countries:
        country_data = data.query('Country == @country')
        axes = plot_country_ecis_indicator_scatter(data=country_data, 
                                                   x_cols=x_cols, 
                                                   y_col=y_col, 
                                                   colors=colors, 
                                                   income_colors=income_colors, 
                                                   marker_kwargs=marker_kwargs, 
                                                   ncols=ncols, 
                                                   figsize=figsize,
                                                   arrows=False, 
                                                   arrow_style=arrow_style,
                                                   arrow_kwargs=arrow_kwargs,
                                                   year_labels=False,
                                                   axes=axes,
                                                   n_steps=n_steps
                                                   )
    axes.flat[0].figure.suptitle(f'Countries {eci_type} vs {y_col}', fontsize=22, y=0.9, 
                    verticalalignment='bottom', 
                    horizontalalignment='center')
    return axes

def prepare_regression_data(data: DataFrame, y_var: str, x_vars: List[str], 
                            str_mapper: StringMapper, control_vars: Optional[List[str]]=None,
                            ) -> Tuple[DataFrame, str, List[str], List[str]]:
    if control_vars is None:
        control_vars = []
    regression_data = data.loc[:, [y_var, *x_vars, *control_vars]].copy(deep=True)
    regression_data.columns = [str_mapper.uglify_str(var) for var in regression_data.columns]
    y_var = str_mapper.uglify_str(y_var)
    x_vars = [str_mapper.uglify_str(var) for var in x_vars]
    control_vars = [str_mapper.uglify_str(var) for var in control_vars]
    return regression_data, y_var, x_vars, control_vars
    
def get_quantile_regression_results(data: DataFrame, y_var: str, x_vars: List[str], 
                                    control_vars: Optional[List[str]]=None, 
                                    quadratic=False, quantiles: Optional[List[float]]=None, 
                                    max_iter: Optional[int]=2_500) -> Tuple[DataFrame, Dict]:
    if quantiles is None:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    formula_terms = x_vars.copy()
    if quadratic:
        formula_terms += [f"I({var} ** 2)" for var in formula_terms]
    if control_vars: 
        formula_terms += control_vars
    formula = f"{y_var} ~ " + " + ".join(formula_terms)
    results = {q: smf.quantreg(formula, data).fit(q=q, max_iter=max_iter) for q in quantiles}
    return results

def add_significance2(row: Series) -> Series:
    coeff = round(row['coeff'], 5)
    t_value = round(row['t-value'], 5)
    p_value = row['p-value']
    if p_value < 0.001:
        return f"{coeff}({t_value})***"
    elif p_value < 0.01:
        return f"{coeff}({t_value})**"
    elif p_value < 0.05:
        return f"{coeff}({t_value})*"
    else:
        return f"{coeff}({t_value})"
    
def get_quantile_regression_results_coeffs(results: Dict[int, RegressionResultsWrapper], x_vars: List[str]) -> DataFrame:
    _col_name_map = {
        'Unnamed: 0': 'Var',
        'coef': 'coeff',
        'std err': 'Std Err',
        't': 't-value',
        'P>|t|': 'p-value',
    }
    regression_df = []
    for q, result in results.items():
        coeffs = pd.concat(pd.read_html(results[q].summary().tables[1].as_html(), header=0))
        coeffs = coeffs.rename(columns={col: _col_name_map.get(col, col) for col in coeffs.columns})
        coeffs = coeffs.set_index('Var')
        coeffs['Value'] = coeffs[['coeff', 't-value', 'p-value']].apply(add_significance2, axis=1)
        coeffs = coeffs[['Value']]
        coeffs.columns = pd.MultiIndex.from_tuples([(str(q), c) for c in coeffs.columns])
        regression_df.append(coeffs)
    regression_df = pd.concat(regression_df, axis=1)
    regression_df = regression_df.reset_index().melt(id_vars='Var', var_name=['Quantile'], value_name='Value')
    regression_df.columns = ['Indep Vars', *regression_df.columns[1:]]
    regression_df['Indep Vars'] = (regression_df['Indep Vars']
                                   .replace(r'^I\((.*)\)$', r'\1', regex=True)
                                   .replace('Intercept', '_Intercept')
                                   )
    regression_df['Reg Type'] = type(results[q].model).__name__
    reg_degree = 'quadratic' if all(
        [f"{var} ** 2" in regression_df['Indep Vars'].values for var in x_vars]
        ) else 'linear'
    regression_df['Reg Degree'] = reg_degree
    regression_df['Dep Var'] = results[q].model.endog_names
    regression_df['Var Type'] = regression_df['Indep Vars'].apply(
        lambda x: 'Exog' if x.replace(' ** 2', '') in x_vars else 'Control'
        )
    regression_df = regression_df.sort_values(by=['Var Type', 'Indep Vars', 'Quantile'], ascending=[False, True, True])
    regression_df['Quantile'] = regression_df['Quantile'].astype(float)
    regression_df = regression_df.set_index(['Reg Type', 'Reg Degree', 'Dep Var', 'Var Type', 'Indep Vars', 'Quantile'])
    regression_df['Id'] = generate_hash_from_dataframe(regression_df, ['Reg Type', 'Dep Var', 'Indep Vars'])
    regression_df = regression_df.set_index('Id', append=True)
    regression_df = regression_df.reorder_levels([regression_df.index.names[-1]] + regression_df.index.names[:-1])
    regression_stats = results[q].summary().tables[0].as_html()
    regression_stats = pd.read_html(regression_stats, index_col=0)[0].reset_index()
    regression_stats = pd.concat(
        [regression_stats.iloc[:-1, :2], regression_stats.iloc[:, 2:].rename(columns={2: 0, 3: 1})],
          axis=0, ignore_index=True)
    regression_stats.columns = ['Stat', 'Value']
    regression_stats = regression_stats.set_index('Stat')

    return regression_df, regression_stats

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



                
                
                    
