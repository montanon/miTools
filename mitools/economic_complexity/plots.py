import functools
import random
import statistics
from string import ascii_uppercase, digits
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from numpy import ndarray
from pandas import DataFrame, Series

from ..utils import clean_str, stretch_string
from .objects import Product, ProductsBasket


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

def adjust_axes_lims(axes, x=True, y=True):
    if not (x and y):
        return axes
    if x:
        xlim_min, xlim_max = float('inf'), float('-inf')
    if y:
        ylim_min, ylim_max = float('inf'), float('-inf')
    for ax in axes.flat:
        if not is_ax_empty(ax):
            if x:
                x_min, x_max = ax.get_xlim()
                xlim_min, xlim_max = min(x_min, xlim_min), max(x_max, xlim_max)
            if y:
                y_min, y_max = ax.get_ylim()
                ylim_min, ylim_max = min(y_min, ylim_min), max(y_max, ylim_max)
    for ax in axes.flat:
        if x:
            ax.set_xlim(xlim_min, xlim_max)
        if y:
            ax.set_ylim(ylim_min, ylim_max)
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
    if last_ax.get_legend() is None and income_colors is not None:  # Check if last_ax is empty
        last_ax.cla()
        last_ax.set_xticks([])  # Remove x-axis ticks
        last_ax.set_yticks([])  # Remove y-axis ticks
        last_ax.axis('off') 

        legend_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                        markersize=10, markeredgecolor='k', label=label) for label, color in income_colors.items()]
        last_ax.legend(handles=legend_handles, fontsize=16, loc='center left', ncols=1)

    return axes

def plot_countries_ecis_indicator_scatter(data, countries, eci_type, x_cols, y_col, colors=None, income_colors=None, 
                                          marker_kwargs=None, ncols=3, figsize=(7,7), arrow_style=None, arrow_kwargs=None,
                                          n_steps=1):
    axes = None
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
                            control_vars: Optional[List[str]]=None
                            ) -> Tuple[DataFrame, str, List[str], List[str]]:
    if control_vars is None:
        control_vars = []
    _clean_str = functools.partial(clean_str, pattern='[ &$%(),-]+', sub_char='_')
    regression_data = data.loc[:, [y_var, *x_vars, *control_vars]].copy(deep=True)
    regression_data.columns = [_clean_str(var) for var in regression_data.columns]
    y_var = _clean_str(y_var)
    x_vars = [_clean_str(var) for var in x_vars]
    control_vars = [_clean_str(var) for var in control_vars]
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
    coeff = round(row['coeff'], 3)
    t_value = round(row['t-value'], 2)
    p_value = row['p-value']
    if p_value < 0.001:
        return f"{coeff}({t_value})***"
    elif p_value < 0.01:
        return f"{coeff}({t_value})**"
    elif p_value < 0.05:
        return f"{coeff}({t_value})*"
    else:
        return f"{coeff}({t_value})"
    
def get_quantile_regression_results_coeffs(results: Dict[int, DataFrame]) -> DataFrame:
    regression_df = []
    for q, result in results.items():
        coeffs = pd.concat([result.params, result.tvalues, result.pvalues], axis=1)
        coeffs.columns = ['coeff', 't-value', 'p-value']
        coeffs['Coefficient'] = coeffs[['coeff', 't-value', 'p-value']].apply(add_significance2, axis=1)
        coeffs = coeffs[['Coefficient']]
        coeffs.columns = pd.MultiIndex.from_tuples([(str(q), c) for c in coeffs.columns])
        regression_df.append(coeffs)
    regression_df = pd.concat(regression_df, axis=1)
    regression_df = regression_df.reset_index().melt(id_vars='index', var_name=['Quantile'], value_name='Coefficient')
    regression_df.columns = ['Indep Vars', *regression_df.columns[1:]]
    regression_df['Indep Vars'] = (regression_df['Indep Vars'].str
                                   .replace(r'^I\((.*)\)$', r'\1', regex=True)
                                   .replace('Intercept', '_Intercept')
                                   )
    regression_df['Type'] = 'quantreg'
    regression_df = regression_df.sort_values(by=['Indep Vars', 'Quantile'])
    regression_df = regression_df.set_index(['Type', 'Indep Vars', 'Quantile'])

    regression_stats = results[q].summary().tables[0].as_html()
    regression_stats = pd.read_html(regression_stats, index_col=0)[0].reset_index()
    regression_stats = pd.concat(
        [regression_stats.iloc[:-1, :2], regression_stats.iloc[:, 2:].rename(columns={2: 0, 3: 1})],
          axis=0, ignore_index=True)
    regression_stats.columns = ['Stat', 'Value']
    regression_stats = regression_stats.set_index('Stat')

    return regression_df, regression_stats

def get_quantile_regression_predictions(regression_data: DataFrame, regression_coeffs: DataFrame)-> Dict[str, ndarray]:
    independent_vars = regression_coeffs.index.get_level_values(0).unique()[1:]
    x_values = {var: np.linspace(
        regression_data[var].min(), regression_data[var].max(), 100
        ) for var in independent_vars}
    predictions = {}
    for var in independent_vars:
        predictions.setdefault(var, {})
        for n, q in enumerate(regression_coeffs.columns.get_level_values(0).unique()):
            beta = regression_coeffs.loc[var, (q, 'beta1')]
            intercept = regression_coeffs.loc['Intercept', (q, 'beta1')]
            pred = intercept + beta * x_values[var]
            if (q, 'beta2') in regression_coeffs.columns:
                beta2 = regression_coeffs.loc[var, (q, 'beta2')]
                pred += beta2 * (x_values[var]**2)
            predictions[var][q] = pred
    return predictions, x_values
