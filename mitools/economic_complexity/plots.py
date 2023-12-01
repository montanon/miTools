import random
import statistics
from string import ascii_uppercase, digits

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import stretch_string
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