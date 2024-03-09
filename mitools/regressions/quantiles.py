import re
import traceback
import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.axes import Axes
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from numpy.linalg import LinAlgError
from pandas import DataFrame, MultiIndex, Series
from statsmodels.regression.linear_model import RegressionResultsWrapper

#from tqdm import tqdm
from tqdm.notebook import tqdm

from ..economic_complexity import StringMapper
from ..pandas import idxslice
from ..regressions import generate_hash_from_dataframe
from ..utils import auto_adjust_columns_width
from ..visuals import (
    adjust_axes_labels,
    adjust_axes_lims,
    adjust_text_axes_limits,
    is_axes_empty,
)

warnings.simplefilter('ignore')
Color = Union[Tuple[int, int, int], str]

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
    STATS: str = 'Stats'
    INTERCEPT: str = 'Intercept'
    ANNOTATION: str = 'Q'
    PARQUET_SUFFIX: str = 'regressions'
    EXCEL_SUFFIX: str = 'regressions'
    MAIN_PLOT: str = 'regression_data'
    PLOTS_SUFFIX: str = 'regression'


def plot_regressions_predictions(data: DataFrame, 
                                 dependent_variables: List[str], 
                                 independent_variables: Dict[str, List[str]], 
                                 regressions_folder: PathLike, 
                                 groups: List[str], 
                                 all_groups: str,
                                 groups_col: Optional[str]='Income Group',
                                 entity_col: Optional[str]='Country',
                                 time_col: Optional[str]='Year',
                                 figsize: Optional[Tuple[float, float]]=(9,7),
                                 marker_kwargs: Optional[Dict[str, Any]]=None,
                                 annotation_kwargs: Optional[Dict[str, Any]]=None,
                                 text_x_offset: Optional[float]=0.0025,
                                 adjust_axes_lims_kwargs: Optional[Dict[str, Any]]=None,
                                 significance_plot_kwargs: Optional[Dict[str, Dict[str,Any]]]=None, 
                                 labels_fontsize: Optional[int]=16,
                                 indep_vars_colors: Optional[List[Color]]=None,
                                 groups_colors: Optional[Dict[str, Color]]=None,
                                 quantiles: Optional[List[float]]=None,
                                 recalculate: Optional[bool]=False,
                                 ):
    for dependent_variable in tqdm(dependent_variables, desc='Dependent Variables', position=0, leave=True):
        dep_var_name = dependent_variable.replace('/', '').replace(' ', '_')
        dep_var_folder = regressions_folder / dep_var_name
        if not dep_var_folder.exists(): 
            dep_var_folder.mkdir(exist_ok=True)
        for name_tag, independent_vars in tqdm(
            independent_variables.items(), desc='Independent Variables', position=1, leave=False
            ):
            name_tag_folder = dep_var_folder / name_tag
            if not name_tag_folder.exists(): 
                name_tag_folder.mkdir(exist_ok=True)
            regressions_coeffs_path = name_tag_folder / f"{name_tag}_{dep_var_name}_{QuantileRegStrs.PARQUET_SUFFIX}.parquet"       
            if regressions_coeffs_path.exists():
                regressions_coeffs = pd.read_parquet(regressions_coeffs_path)
                for regression_id, regression_coeffs in tqdm(
                    regressions_coeffs.groupby(QuantileRegStrs.ID, axis=0), desc='Plots', position=2, leave=False
                    ):
                    create_regression_plots(data=data, 
                                            regression_coeffs=regression_coeffs,
                                            regression_id=regression_id,
                                            dependent_variable=dependent_variable,
                                            independent_variables=independent_vars,
                                            name_tag=name_tag,
                                            groups=groups,
                                            all_groups=all_groups,
                                            folder=name_tag_folder,
                                            groups_col=groups_col,
                                            entity_col=entity_col,
                                            time_col=time_col,
                                            figsize=figsize,
                                            marker_kwargs=marker_kwargs,
                                            annotation_kwargs=annotation_kwargs,
                                            text_x_offset=text_x_offset,
                                            adjust_axes_lims_kwargs=adjust_axes_lims_kwargs,
                                            significance_plot_kwargs=significance_plot_kwargs,
                                            labels_fontsize=labels_fontsize,
                                            indep_vars_colors=indep_vars_colors,
                                            groups_colors=groups_colors,
                                            quantiles=quantiles,
                                            recalculate=recalculate,
                                            )
                    
def create_regression_plots(data: DataFrame, 
                            regression_coeffs: DataFrame,
                            regression_id: str,
                            dependent_variable: str, 
                            independent_variables: List[str],
                            name_tag: str,
                            groups: List[str],
                            all_groups: str,
                            folder: PathLike,
                            groups_col: Optional[str]='Income Group',
                            entity_col: Optional[str]='Country',
                            time_col: Optional[str]='Year',
                            figsize: Optional[Tuple[float, float]]=(9,7),
                            marker_kwargs: Optional[Dict[str, Any]]=None,
                            annotation_kwargs: Optional[Dict[str, Any]]=None,
                            text_x_offset: Optional[float]=0.0025,
                            adjust_axes_lims_kwargs: Optional[Dict[str, Any]]=None,
                            significance_plot_kwargs: Optional[Dict[str, Dict[str,Any]]]=None, 
                            labels_fontsize: Optional[int]=16,
                            indep_vars_colors: Optional[List[Color]]=None,
                            groups_colors: Optional[Dict[str, Color]]=None,
                            quantiles: Optional[List[float]]=None,
                            recalculate: Optional[bool]=False,
                            ):
    if not quantiles:
        quantiles = regression_coeffs.index.get_level_values(QuantileRegStrs.QUANTILE).unique()
    quadratic = (regression_coeffs.index
                 .get_level_values(QuantileRegStrs.REGRESSION_DEGREE).unique()[0] == QuantileRegStrs.QUADRATIC_REG
                 )
    main_plot = folder / f"{QuantileRegStrs.MAIN_PLOT}.png"
    regression_plot = folder / f"{regression_id}_{dependent_variable.replace('/', '').replace(' ', '_')}_{QuantileRegStrs.PLOTS_SUFFIX}.png"
    if not main_plot.exists() or not regression_plot.exists() or recalculate:
        predictions, significances, x_values = get_regression_predictions(data=data, 
                                                                          regression_coeffs=regression_coeffs,
                                                                          groups=groups,
                                                                          groups_col=groups_col,
                                                                          all_groups=all_groups
                                                                          )
        axes = plot_income_levels_ecis_indicator_scatter(data=data,
                                                         x_vars_cols=independent_variables,
                                                         y_var_col=dependent_variable,
                                                         name_tag=name_tag,
                                                         groups=groups,
                                                         all_groups=all_groups,
                                                         groups_col=groups_col,
                                                         entity_col=entity_col,
                                                         time_col=time_col,
                                                         colors=indep_vars_colors,
                                                         groups_colors=groups_colors,
                                                         figsize=figsize,
                                                         marker_kwargs=marker_kwargs,
                                                         adjust_axes_lims_kwargs=adjust_axes_lims_kwargs
                                                         )
        if not main_plot.exists() or recalculate:
            axes.flat[0].figure.savefig(main_plot)
        if not regression_plot.exists() or recalculate:
            plot_regression_predictions_by_group(independent_variables=independent_variables, 
                                                 dependent_variable=dependent_variable,
                                                 predictions=predictions,
                                                 x_values=x_values,
                                                 significances=significances,
                                                 groups=groups,
                                                 quantiles=quantiles,
                                                 quadratic=quadratic,
                                                 significance_plot_kwargs=significance_plot_kwargs,
                                                 annotation_kwargs=annotation_kwargs,
                                                 labels_fontsize=labels_fontsize,
                                                 text_x_offset=text_x_offset,
                                                 axes=axes
                                                 )
            axes.flat[0].figure.savefig(regression_plot)
            plt.close()

def plot_regression_predictions_by_group(independent_variables: List[str], 
                                         dependent_variable: str, 
                                         predictions: DataFrame, 
                                         x_values: DataFrame, 
                                         significances: DataFrame, 
                                         groups: List[str], 
                                         quantiles: List[float], 
                                         quadratic: bool, 
                                         significance_plot_kwargs: Optional[Dict[str, Dict[str, Any]]]=None,
                                         annotation_kwargs: Optional[Dict[str, Any]]=None,
                                         ncols: Optional[int]=3,
                                         figsize: Optional[Tuple[float, float]]=(7, 7),
                                         labels_fontsize: Optional[int]=16,
                                         text_x_offset: Optional[float]=0.0025,
                                         axes=None) -> Axes:
    if significance_plot_kwargs is None:
        significance_plot_kwargs = {}
    if axes is None:
        nrows = (len(independent_variables) + 1) // ncols
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    if annotation_kwargs is None:
        annotation_kwargs = {'fontsize': 10, 'color': 'k', 'horizontalalignment': 'left', 'zorder': 99}
    for n, group in enumerate(groups):
        current_axes = axes.flat[n::len(groups)]
        for ax, var in zip(current_axes, independent_variables):
            for line in ax.lines:
                line.set_alpha(0.10)
            for quantile in quantiles:
                independent_variable = (
                    f"{var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}" 
                    if quadratic and 
                    f"{var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}" in predictions.columns.get_level_values(-2).unique() 
                    else var
                    )
                x_vals = x_values[group][var]
                preds = predictions[group][(dependent_variable, independent_variable, quantile)]
                significance = significances[group][(dependent_variable, independent_variable, quantile)].values[0]
                ax.plot(x_vals, preds, **significance_plot_kwargs.get(significance, {}))
                text = ax.text(x_vals.iloc[-1]*(1 + text_x_offset), 
                               preds.iloc[-1], 
                               f'{quantile}{QuantileRegStrs.ANNOTATION}', 
                               **annotation_kwargs
                               )
            adjust_text_axes_limits(ax, text)
            adjust_axes_labels(ax, labels_fontsize)
    return axes

def create_regression_file_paths(eci_type_folder: PathLike, regression_id: str) -> Tuple[Path, Path]:
    eci_type_folder = Path(eci_type_folder)
    main_plot = eci_type_folder / f"{QuantileRegStrs.MAIN_PLOT}.png"
    regression_plot = eci_type_folder / f"{regression_id}_{QuantileRegStrs.PLOTS_SUFFIX}.png"
    return main_plot, regression_plot

def plot_group_data(ax, group_data, x_vals, significance_plot_kwargs, annotation_kwargs):
    for quantile, preds in group_data.items():
        ax.plot(x_vals, preds, **significance_plot_kwargs[quantile])
        text = ax.text(x_vals.iloc[-1]*1.0025, preds.iloc[-1], f'{quantile}{QuantileRegStrs.ANNOTATION}', **annotation_kwargs)
    return text

def get_regression_predictions(data: DataFrame, 
                               regression_coeffs: DataFrame, 
                               groups: List[str], 
                               groups_col: str, 
                               all_groups: str
                               ) -> Tuple[DataFrame, DataFrame, DataFrame]:
    predictions, significances, x_values = [], [], []
    for group in groups:
        group_predictions, group_significances, group_x_values = get_quantile_regression_predictions_by_group(
            regression_data=data,
            regression_coeffs=regression_coeffs,
            group=group,
            groups_col=groups_col,
            all_groups=all_groups
            )
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

def plot_income_levels_ecis_indicator_scatter(data: DataFrame, 
                                              x_vars_cols: List[str], 
                                              y_var_col: str,
                                              name_tag: str,
                                              groups: List[str],
                                              all_groups: str, 
                                              groups_col: Optional[str]='Income Group',
                                              entity_col: Optional[str]='Country', 
                                              time_col: Optional[str]='Year',
                                              colors: List[Color]=None, 
                                              groups_colors: Dict[str, Color]=None, 
                                              figsize: Optional[Tuple[float, float]]=(9,7),
                                              marker_kwargs: Optional[Dict[str, Any]]=None,
                                              adjust_axes_lims_kwargs: Optional[Dict[str, Any]]=None,
                                              ) -> Axes:
    if adjust_axes_lims_kwargs is None:
        adjust_axes_lims_kwargs = {'mode': 'rows', 'x': True, 'y': True}
    nrows = len(x_vars_cols)
    ncols = len(groups)
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    for n, group in enumerate(groups):
        if group != all_groups:
            group_data = data.loc[idxslice(data, level=groups_col, value=group, axis=0), :].copy(deep=True)
            groups = [group]
        else:
            group_data = data.copy(deep=True)
            groups = group_data.index.get_level_values(groups_col).unique().tolist()
        entities = data.index.get_level_values(entity_col).unique().tolist()
        plot_countries_ecis_indicator_scatter(data=group_data, 
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
                                            axes=axes.flat[n::ncols]
                                              )
    axes = adjust_axes_lims(axes, **adjust_axes_lims_kwargs)
    return axes

def prettify_index_level(mapper: StringMapper, pattern: str, level: str, level_name: str, 
                         levels_to_remap: List[str]) -> str:
    if level_name in levels_to_remap:
        return level.map(lambda x: prettify_with_pattern(x, mapper, pattern))
    return level

def prettify_with_pattern(string: str, mapper: StringMapper, pattern: str) -> str:
    base_string, pattern_str, _ = string.partition(pattern)
    remapped_base = mapper.prettify_str(base_string)
    return f"{remapped_base}{pattern}" if pattern_str else remapped_base

def get_quantile_regression_predictions_by_group(regression_data: DataFrame, 
                                                 regression_coeffs: DataFrame, 
                                                 group: str,
                                                 groups_col: str,
                                                 all_groups: str) -> Tuple[DataFrame, DataFrame]:
    group_data = get_group_data(regression_data, group, groups_col, all_groups)
    dependent_var = regression_coeffs.index.get_level_values(QuantileRegStrs.DEPENDENT_VAR).unique()[0]
    quantiles = regression_coeffs.index.get_level_values(QuantileRegStrs.QUANTILE).unique()
    independent_vars = [var for var in regression_coeffs.index.get_level_values(
        QuantileRegStrs.INDEPENDENT_VARS).unique() if var != QuantileRegStrs.INTERCEPT]
    x_values = prepare_x_values(group_data, independent_vars)

    predictions, significances, columns = [], [], []
    for var in independent_vars:
        quadratic = QuantileRegStrs.QUADRATIC_VAR_SUFFIX in var
        x_var_values = x_values[var.replace(QuantileRegStrs.QUADRATIC_VAR_SUFFIX, '')]
        var_values = [
            var, QuantileRegStrs.INTERCEPT] if not quadratic else [
                var.replace(QuantileRegStrs.QUADRATIC_VAR_SUFFIX, ''), var, QuantileRegStrs.INTERCEPT]
        vars_idx = idxslice(regression_coeffs, level=QuantileRegStrs.INDEPENDENT_VARS, value=var_values, axis=0)
        var_coeffs = regression_coeffs.loc[vars_idx, :]

        for quantile in quantiles:
            quantile_idx = idxslice(var_coeffs, level=QuantileRegStrs.QUANTILE, value=quantile, axis=0)
            values = var_coeffs.loc[quantile_idx, group].values
            significance = ','.join(
                [match.group() if match else '-' for match in [re.search(r"\*+$", val) for val in values[:-1]]]
                )
            coeffs = [float(re.search(r"([-\d.]+)\(", val).group(1)) for val in values]
            prediction = get_prediction(x_var_values, coeffs, quadratic)

            predictions.append(prediction)
            significances.append(significance)
            columns.append((dependent_var, var, quantile))

    predictions = DataFrame(predictions, index=MultiIndex.from_tuples(columns)).T
    significances = DataFrame(significances, index=MultiIndex.from_tuples(columns)).T
    return predictions, significances, x_values

def get_group_data(regression_data: DataFrame, group: str, group_col: str, all_groups: str) -> DataFrame:
    if group != all_groups:
        return regression_data.loc[regression_data.index.get_level_values(group_col) == group]
    return regression_data

def prepare_x_values(group_data: DataFrame, independent_vars: List[str]) -> DataFrame:
    return DataFrame({
        var: np.linspace(group_data[var].min(), group_data[var].max(), 100)
        for var in independent_vars if QuantileRegStrs.QUADRATIC_VAR_SUFFIX not in var
    })

def get_prediction(x_values: Series, coeffs: List[float], quadratic: bool) -> float:
    prediction = coeffs[-1] + coeffs[0] * x_values
    if quadratic:
        prediction += coeffs[1] * x_values ** 2
    return prediction

def get_quantile_regression_results_stats(results: Dict[int, RegressionResultsWrapper]) -> DataFrame:
    regression_stats = []
    for q, result in results.items():
        stats = result.summary().tables[0].as_html()
        stats = pd.read_html(stats, index_col=0)[0].reset_index()
        stats = pd.concat(
            [stats.iloc[:-1, :2], stats.iloc[:, 2:].rename(columns={2: 0, 3: 1})],
            axis=0, ignore_index=True)
        stats.columns = [QuantileRegStrs.STATS, QuantileRegStrs.VALUE]
        stats[QuantileRegStrs.Quantile] = q
        stats = stats.set_index([QuantileRegStrs.QUANTILE, QuantileRegStrs.STATS])
        regression_stats.append(stats)
    regression_stats = pd.concat(regression_stats, axis=0)
    return regression_stats

def get_quantile_regression_results_coeffs(results: Dict[int, RegressionResultsWrapper],
                                           independent_variables: List[str]
                                           ) -> DataFrame:
    regression_coeffs = pd.concat([process_quantile_regression_result(q, result) for q, result in results.items()], axis=1)
    regression_coeffs = (regression_coeffs.reset_index()
                         .melt(id_vars=QuantileRegStrs.UNNAMED, 
                               var_name=[QuantileRegStrs.QUANTILE], 
                               value_name=QuantileRegStrs.VALUE)
                         )
    regression_coeffs.columns = [QuantileRegStrs.INDEPENDENT_VARS, *regression_coeffs.columns[1:]]
    regression_coeffs[QuantileRegStrs.INDEPENDENT_VARS] = (regression_coeffs[QuantileRegStrs.INDEPENDENT_VARS]
                                   .replace(QuantileRegStrs.INDEPENDENT_VARS_PATTERN, r'\1', regex=True)
                                            )
    regression_coeffs[QuantileRegStrs.REGRESSION_TYPE] = type(list(results.values())[0].model).__name__
    reg_degree = QuantileRegStrs.QUADRATIC_REG if all(
        [f"{var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}" in regression_coeffs[
            QuantileRegStrs.INDEPENDENT_VARS].values for var in independent_variables]
        ) else QuantileRegStrs.LINEAR_REG
    regression_coeffs[QuantileRegStrs.REGRESSION_DEGREE] = reg_degree
    regression_coeffs[QuantileRegStrs.DEPENDENT_VAR] = list(results.values())[0].model.endog_names
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
        #QuantileRegStrs.DEPENDENT_VAR,
        QuantileRegStrs.INDEPENDENT_VARS
        ],
        length=12
        )
    regression_coeffs = regression_coeffs.set_index(QuantileRegStrs.ID, append=True)
    regression_coeffs = regression_coeffs.reorder_levels(
        [regression_coeffs.index.names[-1]] + regression_coeffs.index.names[:-1])
    return regression_coeffs

def process_quantile_regression_result(q: float, result: RegressionResultsWrapper) -> DataFrame:
    coeffs = pd.concat(pd.read_html(result.summary().tables[1].as_html(), header=0))
    coeffs = coeffs.set_index(QuantileRegStrs.UNNAMED)
    coeffs[QuantileRegStrs.VALUE] = coeffs[
            [QuantileRegStrs.COEF, QuantileRegStrs.T_VALUE, QuantileRegStrs.P_VALUE]
                            ].apply(quantile_regression_value, axis=1)
    coeffs = coeffs[[QuantileRegStrs.VALUE]]
    coeffs.columns = pd.MultiIndex.from_tuples([(str(q), c) for c in coeffs.columns])
    return coeffs

def quantile_regression_value(row: Series) -> Series:
    coeff = round(row[QuantileRegStrs.COEF], 5)
    t_value = round(row[QuantileRegStrs.T_VALUE], 5)
    p_value = row[QuantileRegStrs.P_VALUE]
    if p_value <= 0.001:
        return f"{coeff}({t_value})***"
    elif p_value <= 0.01:
        return f"{coeff}({t_value})**"
    elif p_value <= 0.05:
        return f"{coeff}({t_value})*"
    else:
        return f"{coeff}({t_value})"
    
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
        formula_terms += [f"I({var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX})" for var in formula_terms]
    if control_variables: 
        formula_terms += control_variables
    formula = f"{dependent_variable} ~ " + " + ".join(formula_terms)
    results = {q: smf.quantreg(formula, data).fit(q=q, max_iter=max_iter) for q in quantiles}
    return results

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
                                          figsize: Optional[Tuple[float, float]]=(7,7), 
                                          arrows: Optional[bool]=False,
                                          arrow_style: Optional[bool]=None, 
                                          arrow_kwargs: Optional[Dict[str, Any]]=None,
                                          set_arrows_ax_limits: Optional[bool]=False,
                                          marker_kwargs: Optional[Dict[str, Any]]=None, 
                                          axes: Optional[Axes]=None
                                          ) -> Axes:
    nrows = (len(x_vars_cols) + 1) // ncols
    if axes is None:
        _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    for entity in entities:
        country_data = data.query(f'{entity_col} == @entity')
        if country_data.empty: 
            continue
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
                                        figsize: Optional[Tuple[float, float]]=(9, 9), 
                                        arrows: Optional[bool]=True, 
                                        arrow_style: Optional[ArrowStyle]=None, 
                                        arrow_kwargs: Optional[Dict[str, Any]]=None,
                                        set_arrows_ax_limits: Optional[bool]=True, 
                                        marker_kwargs: Optional[Dict[str, Any]]=None, 
                                        axes: Optional[Axes]=None
                                        ) -> Axes:
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
                                       figsize: Optional[Tuple[float, float]]=(9, 9), 
                                       arrows: Optional[bool]=True, 
                                       arrow_style: Optional[ArrowStyle]=None, 
                                       arrow_kwargs: Optional[Dict[str, Any]]=None, 
                                       set_arrows_ax_limits: Optional[bool]=True, 
                                       marker_kwargs: Optional[Dict[str, Any]]=None,
                                       ax: Optional[Axes]=None
                                       ) -> Axes:
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

def create_quantile_regressions_results(data: DataFrame,
                                        dependent_variables: List[str],
                                        independent_variables: Dict[str, List[str]],
                                        groups: List[str],
                                        regressions_folder: PathLike,
                                        quadratics: List[bool],
                                        quantiles: Optional[List[float]]=None,
                                        str_mapper: Optional[StringMapper]=None,
                                        control_variables: Optional[List[List[str]]]=None,
                                        group_col: Optional[str]='Income Group',
                                        all_groups: Optional[str]='All income',
                                        max_iter: Optional[int]=2_500,
                                        recalculate: Optional[bool]=False,
                                        ):
    if control_variables is None:
        control_variables = [[]]
    for dependent_variable in tqdm(dependent_variables, desc='Dependent Variables', position=0, leave=False):
        dep_var_name = dependent_variable.replace('/', '').replace(' ', '_')
        dep_var_folder = regressions_folder / dep_var_name
        if not dep_var_folder.exists(): 
            dep_var_folder.mkdir(exist_ok=True)
        for name_tag, indep_variables in tqdm(
            independent_variables.items(), desc='Independent Variables', position=1, leave=False
            ):
            name_tag_folder = dep_var_folder / name_tag
            if not name_tag_folder.exists(): 
                name_tag_folder.mkdir(exist_ok=True)
            dep_var_name_excel = name_tag_folder / f"{name_tag}_{dep_var_name}_{QuantileRegStrs.EXCEL_SUFFIX}.xlsx"
            dep_var_name_parquet = name_tag_folder / f"{name_tag}_{dep_var_name}_{QuantileRegStrs.PARQUET_SUFFIX}.parquet"
            if not dep_var_name_excel.exists() or not dep_var_name_parquet.exists() or recalculate:
                name_tag_regressions = []
                for control_vars in tqdm(control_variables, desc='Control Variables', position=2, leave=False):
                    if dependent_variable not in control_vars:
                        try:
                            quadratic_regressions = []
                            for quadratic in quadratics:
                                group_regressions = []
                                for group in groups:
                                    if group != all_groups:
                                        group_data = data.loc[idxslice(data, level=group_col, value=group, axis=0), :].copy(deep=True)
                                    else:
                                        group_data = data.copy(deep=True)
                                    regression_data, dependent_var, independent_vars, c_vars = prepare_regression_data(
                                        data=group_data,
                                        dependent_variable=dependent_variable,
                                        independent_variables=indep_variables,
                                        control_variables=control_vars,
                                        str_mapper=str_mapper
                                        )
                                    with warnings.catch_warnings():
                                        regression_results = get_quantile_regression_results(
                                            data=regression_data, 
                                            dependent_variable=dependent_var, 
                                            independent_variables=independent_vars, 
                                            control_variables=c_vars, 
                                            quantiles=quantiles,
                                            quadratic=quadratic, 
                                            max_iter=max_iter
                                            )
                                    regression_coeffs = get_quantile_regression_results_coeffs(
                                        results=regression_results,
                                        independent_variables=independent_vars
                                        )
                                    regression_coeffs.columns = [group]
                                    # regression_stats = get_quantile_regression_results_stats(results=regression_results)
                                    group_regressions.append(regression_coeffs)
                                group_regressions = pd.concat(group_regressions, axis=1)
                                quadratic_regressions.append(group_regressions)
                            quadratic_regressions = pd.concat(quadratic_regressions, axis=0)
                            name_tag_regressions.append(quadratic_regressions)
                        except LinAlgError as e:
                            print(f"{e}\n{traceback.format_exc()}")
                            pass
                if name_tag_regressions:
                    name_tag_regressions = pd.concat(name_tag_regressions, axis=0)

                    levels_to_remap = [QuantileRegStrs.DEPENDENT_VAR, QuantileRegStrs.INDEPENDENT_VARS]
                    pretty_index = name_tag_regressions.index.set_levels([
                        prettify_index_level(str_mapper, 
                                             QuantileRegStrs.QUADRATIC_VAR_SUFFIX, 
                                             level, 
                                             level_id, 
                                             levels_to_remap
                                             ) for level, level_id in zip(name_tag_regressions.index.levels, 
                                                                          name_tag_regressions.index.names
                                                                          )
                    ],
                    level=name_tag_regressions.index.names
                    )
                    name_tag_regressions.index = pretty_index

                    name_tag_regressions.to_parquet(dep_var_name_parquet)

                    name_tag_regressions.to_excel(dep_var_name_excel)
                    book = openpyxl.load_workbook(dep_var_name_excel)
                    for sheet_name in book.sheetnames:
                        sheet = book[sheet_name]
                        auto_adjust_columns_width(sheet)
                    book.save(dep_var_name_excel)