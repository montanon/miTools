from concurrent.futures import ProcessPoolExecutor, as_completed
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame
from tqdm import tqdm

from mitools.economic_complexity.plots import plot_income_levels_ecis_indicator_scatter
from mitools.utils.helper_functions import stretch_string

from ..visuals import (
    adjust_ax_labels_fontsize,
    adjust_axes_lims,
    adjust_text_axes_limits,
)
from .quantiles import QuantileRegStrs, get_regression_predictions

Color = Union[Tuple[int, int, int], str]


def plot_regressions_predictions_parallel(
    data: DataFrame,
    dependent_variables: List[str],
    independent_variables: Dict[str, List[str]],
    regressions_folder: PathLike,
    groups: List[str],
    all_groups: str,
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
    figsize: Optional[Tuple[float, float]] = (9, 7),
    marker_kwargs: Optional[Dict[str, Any]] = None,
    annotation_kwargs: Optional[Dict[str, Any]] = None,
    text_x_offset: Optional[float] = 0.0025,
    adjust_axes_lims_kwargs: Optional[Dict[str, Any]] = None,
    significance_plot_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    labels_fontsize: Optional[int] = 16,
    indep_vars_colors: Optional[List[Color]] = None,
    groups_colors: Optional[Dict[str, Color]] = None,
    quantiles: Optional[List[float]] = None,
    recalculate: Optional[bool] = False,
    n_workers: Optional[int] = 8,
):
    tasks = []
    for dependent_variable in dependent_variables:
        dep_var_name = dependent_variable.replace("/", "").replace(" ", "_")
        dep_var_folder = regressions_folder / dep_var_name
        if not dep_var_folder.exists():
            dep_var_folder.mkdir(exist_ok=True)
        for name_tag, independent_vars in independent_variables.items():
            name_tag_folder = dep_var_folder / name_tag
            if not name_tag_folder.exists():
                name_tag_folder.mkdir(exist_ok=True)
            regressions_coeffs_path = (
                name_tag_folder
                / f"{name_tag}_{dep_var_name}_{QuantileRegStrs.PARQUET_SUFFIX}.parquet"
            )
            if regressions_coeffs_path.exists():
                regressions_coeffs = pd.read_parquet(regressions_coeffs_path)
                for regression_id, regression_coeffs in regressions_coeffs.groupby(
                    QuantileRegStrs.ID, axis=0
                ):
                    task = dict(
                        data=data.copy(deep=True),
                        regression_coeffs=regression_coeffs.copy(deep=True),
                        regression_id=regression_id,
                        dependent_variable=dependent_variable,
                        independent_variables=independent_vars.copy(),
                        name_tag=name_tag,
                        groups=groups.copy(),
                        all_groups=all_groups,
                        folder=name_tag_folder,
                        groups_col=groups_col,
                        entity_col=entity_col,
                        time_col=time_col,
                        figsize=figsize,
                        marker_kwargs=marker_kwargs.copy()
                        if marker_kwargs is not None
                        else None,
                        annotation_kwargs=annotation_kwargs.copy()
                        if annotation_kwargs is not None
                        else None,
                        text_x_offset=text_x_offset,
                        adjust_axes_lims_kwargs=adjust_axes_lims_kwargs.copy()
                        if adjust_axes_lims_kwargs is not None
                        else None,
                        significance_plot_kwargs=significance_plot_kwargs.copy()
                        if significance_plot_kwargs is not None
                        else None,
                        labels_fontsize=labels_fontsize,
                        indep_vars_colors=indep_vars_colors.copy()
                        if indep_vars_colors is not None
                        else None,
                        groups_colors=groups_colors.copy()
                        if groups_colors is not None
                        else None,
                        quantiles=quantiles.copy() if quantiles is not None else None,
                        recalculate=recalculate,
                    )
                    tasks.append(task)

                    if len(tasks) == n_workers:
                        with ProcessPoolExecutor(max_workers=n_workers) as executor:
                            futures = [
                                executor.submit(create_regression_plots, **task)
                                for task in tasks
                            ]
                            for future in tqdm(
                                as_completed(futures),
                                total=len(futures),
                                desc="Processing Plots",
                            ):
                                future.result()
                        tasks = []
    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(create_regression_plots, **task) for task in tasks
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing Plots"
            ):
                future.result()
        tasks = []


def plot_regressions_predictions(
    data: DataFrame,
    dependent_variables: List[str],
    independent_variables: Dict[str, List[str]],
    regressions_folder: PathLike,
    groups: List[str],
    all_groups: str,
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
    figsize: Optional[Tuple[float, float]] = (9, 7),
    marker_kwargs: Optional[Dict[str, Any]] = None,
    annotation_kwargs: Optional[Dict[str, Any]] = None,
    text_x_offset: Optional[float] = 0.0025,
    adjust_axes_lims_kwargs: Optional[Dict[str, Any]] = None,
    significance_plot_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    labels_fontsize: Optional[int] = 16,
    indep_vars_colors: Optional[List[Color]] = None,
    groups_colors: Optional[Dict[str, Color]] = None,
    quantiles: Optional[List[float]] = None,
    recalculate: Optional[bool] = False,
):
    for dependent_variable in tqdm(
        dependent_variables, desc="Dependent Variables", position=0, leave=True
    ):
        dep_var_name = dependent_variable.replace("/", "").replace(" ", "_")
        dep_var_folder = regressions_folder / dep_var_name
        if not dep_var_folder.exists():
            dep_var_folder.mkdir(exist_ok=True)
        for name_tag, independent_vars in tqdm(
            independent_variables.items(),
            desc="Independent Variables",
            position=1,
            leave=False,
        ):
            name_tag_folder = dep_var_folder / name_tag
            if not name_tag_folder.exists():
                name_tag_folder.mkdir(exist_ok=True)
            regressions_coeffs_path = (
                name_tag_folder
                / f"{name_tag}_{dep_var_name}_{QuantileRegStrs.PARQUET_SUFFIX}.parquet"
            )
            if regressions_coeffs_path.exists():
                regressions_coeffs = pd.read_parquet(regressions_coeffs_path)
                for regression_id, regression_coeffs in tqdm(
                    regressions_coeffs.groupby(QuantileRegStrs.ID, axis=0),
                    desc="Plots",
                    position=2,
                    leave=False,
                ):
                    create_regression_plots(
                        data=data,
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


def create_regression_plots(
    data: DataFrame,
    regression_coeffs: DataFrame,
    regression_id: str,
    dependent_variable: str,
    independent_variables: List[str],
    name_tag: str,
    groups: List[str],
    all_groups: str,
    folder: PathLike,
    groups_col: Optional[str] = "Income Group",
    entity_col: Optional[str] = "Country",
    time_col: Optional[str] = "Year",
    figsize: Optional[Tuple[float, float]] = (9, 7),
    marker_kwargs: Optional[Dict[str, Any]] = None,
    annotation_kwargs: Optional[Dict[str, Any]] = None,
    text_x_offset: Optional[float] = 0.0025,
    adjust_axes_lims_kwargs: Optional[Dict[str, Any]] = None,
    significance_plot_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    labels_fontsize: Optional[int] = 16,
    indep_vars_colors: Optional[List[Color]] = None,
    groups_colors: Optional[Dict[str, Color]] = None,
    quantiles: Optional[List[float]] = None,
    recalculate: Optional[bool] = False,
):
    if not quantiles:
        quantiles = regression_coeffs.index.get_level_values(
            QuantileRegStrs.QUANTILE
        ).unique()
    quadratic = (
        regression_coeffs.index.get_level_values(
            QuantileRegStrs.REGRESSION_DEGREE
        ).unique()[0]
        == QuantileRegStrs.QUADRATIC_REG
    )
    main_plot = folder / f"{QuantileRegStrs.MAIN_PLOT}.png"
    regression_plot = (
        folder
        / f"{regression_id}_{dependent_variable.replace('/', '').replace(' ', '_')}_{QuantileRegStrs.PLOTS_SUFFIX}.png"
    )
    if not main_plot.exists() or not regression_plot.exists() or recalculate:
        predictions, significances, x_values = get_regression_predictions(
            data=data,
            regression_coeffs=regression_coeffs,
            groups=groups,
            groups_col=groups_col,
            all_groups=all_groups,
        )
        axes = plot_income_levels_ecis_indicator_scatter(
            data=data,
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
            adjust_axes_lims_kwargs=adjust_axes_lims_kwargs,
        )
        if not main_plot.exists() or recalculate:
            axes.flat[0].figure.savefig(main_plot)
        if not regression_plot.exists() or recalculate:
            control_variables = (
                regression_coeffs.loc[
                    regression_coeffs.index.get_level_values(
                        QuantileRegStrs.VARIABLE_TYPE
                    )
                    == QuantileRegStrs.CONTROL_VAR
                ]
                .index.get_level_values(QuantileRegStrs.INDEPENDENT_VARS)
                .unique()
                .tolist()
            )
            plot_regression_predictions_by_group(
                independent_variables=independent_variables,
                dependent_variable=dependent_variable,
                control_variables=control_variables,
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
                axes=axes,
            )
            axes.flat[0].figure.savefig(regression_plot)
            plt.close()


def plot_regression_predictions_by_group(
    independent_variables: List[str],
    dependent_variable: str,
    predictions: DataFrame,
    x_values: DataFrame,
    significances: DataFrame,
    groups: List[str],
    quantiles: List[float],
    quadratic: bool,
    control_variables: Optional[List[str]] = None,
    significance_plot_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    annotation_kwargs: Optional[Dict[str, Any]] = None,
    ncols: Optional[int] = 3,
    figsize: Optional[Tuple[float, float]] = (7, 7),
    labels_fontsize: Optional[int] = 16,
    text_x_offset: Optional[float] = 0.0025,
    axes=None,
) -> Axes:
    if significance_plot_kwargs is None:
        significance_plot_kwargs = {}
    if axes is None:
        nrows = (len(independent_variables) + 1) // ncols
        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows)
        )
    if annotation_kwargs is None:
        annotation_kwargs = {
            "fontsize": 10,
            "color": "k",
            "horizontalalignment": "left",
            "zorder": 99,
        }
    for n, group in enumerate(groups):
        current_axes = axes.flat[n :: len(groups)]
        for ax, var in zip(current_axes, independent_variables):
            for line in ax.lines:
                line.set_alpha(0.10)
            for quantile in quantiles:
                independent_variable = (
                    f"{var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}"
                    if quadratic
                    and f"{var}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}"
                    in predictions.columns.get_level_values(-2).unique()
                    else var
                )
                x_vals = x_values[group][var]
                preds = predictions[group][
                    (dependent_variable, independent_variable, quantile)
                ]
                significance = significances[group][
                    (dependent_variable, independent_variable, quantile)
                ].values[0]
                ax.plot(x_vals, preds, **significance_plot_kwargs.get(significance, {}))
                text = ax.text(
                    x_vals.iloc[-1] * (1 + text_x_offset),
                    preds.iloc[-1],
                    f"{quantile}{QuantileRegStrs.ANNOTATION}",
                    **annotation_kwargs,
                )
            adjust_text_axes_limits(ax, text)
            adjust_ax_labels_fontsize(ax, labels_fontsize)
    if quadratic:
        independent_variables = [
            var
            for string in independent_variables
            for var in [string, f"{string}{QuantileRegStrs.QUADRATIC_VAR_SUFFIX}"]
        ]
    model_specification = f"{dependent_variable} ~ {' + '.join(independent_variables)}"
    model_specification += (
        f" + {' + '.join([var for var in control_variables if var != 'Intercept'])}"
        if control_variables
        else ""
    )
    ax.figure.text(
        0.5,
        0.045,
        stretch_string(model_specification, 140),
        ha="center",
        va="bottom",
        fontsize=22,
    )
    return axes


def plot_group_data(
    ax, group_data, x_vals, significance_plot_kwargs, annotation_kwargs
):
    for quantile, preds in group_data.items():
        ax.plot(x_vals, preds, **significance_plot_kwargs[quantile])
        text = ax.text(
            x_vals.iloc[-1] * 1.0025,
            preds.iloc[-1],
            f"{quantile}{QuantileRegStrs.ANNOTATION}",
            **annotation_kwargs,
        )
    return text
