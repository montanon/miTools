from os import PathLike
from typing import Dict, Iterable, List, Optional

from pandas import DataFrame, Series

from mitools.regressions.quantiles import QuantilesRegressionSpecs

from ..utils import remove_chars

SECTION_BREAK = "%$&" * 20
SUB_SECTION_BREAK = "#~^" * 20


def reset_stata(stata_data, stata_id_var, stata_level_var):
    cmds = []
    cmds.append("clear")
    cmds.append(f"quietly use {stata_data}")
    cmds.append(f"quietly levelsof {stata_level_var}, local(levels)")
    cmds.append(f"encode {stata_id_var}, generate(id)")
    cmds.append("xtset id Year")
    return cmds


def run_regression(regression: QuantilesRegressionSpecs, log_path):
    stata_variables = (
        regression.dependent_variable
        + " "
        + " ".join(regression.independent_variables)
        + " "
        + " ".join(regression.control_variables)
    )
    cmds = []
    cmds.append(f"quietly log using {log_path}, replace")
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_descriptive_statistics(stata_variables))
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_skewness_test(stata_variables))
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_correlations(stata_variables))
    cmds.append(f'di "{SECTION_BREAK}"')
    for var in regression.variables:
        cmds.extend(run_unit_root_test(var, "llc"))
        cmds.append(f'di "{SUB_SECTION_BREAK}"')
        cmds.extend(run_unit_root_test(var, "ips"))
        cmds.append(f'di "{SUB_SECTION_BREAK}"')
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_cointegration_test(stata_variables, "kao"))
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_panel_quantiles_model(stata_variables, quantiles=None))
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_vif_statistics())
    cmds.append(f'di "{SECTION_BREAK}"')
    cmds.extend(run_vce_corr_statistics())
    cmds.append(f'di "{SECTION_BREAK}"')
    return cmds


def run_descriptive_statistics(stata_variables):
    cmds = []
    cmds.append(f"tabstat {stata_variables}, stat (n mean median sd skewness kurtosis)")
    return cmds


def run_skewness_test(stata_variables):
    cmds = []
    cmds.append(f"sktest {stata_variables}")
    return cmds


def run_correlations(stata_variables):
    cmds = []
    cmds.append(f"pwcorr {stata_variables}, sig star(10)")
    return cmds


def run_unit_root_test(stata_variable, test):
    cmds = []
    cmds.append(f"xtunitroot {test} {stata_variable}, trend")
    return cmds


def run_cointegration_test(stata_variables, test):
    cmds = []
    cmds.append(f"xtcointtest {test} {stata_variables}")
    return cmds


def run_panel_threshold_model(stata_variables, threshold_stata_variable, thnum):
    cmds = []
    cmds.append(
        f"xtqreg {stata_variables}, rx({threshold_stata_variable}) qx({threshold_stata_variable}) thnum({thnum})"
    )
    return cmds


def run_panel_quantiles_model(stata_variables, quantiles=None):
    if quantiles is None:
        quantiles = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    cmds = []
    cmds.append(f"xtqreg {stata_variables}, q({', '.join(quantiles)})")
    return cmds


def run_vif_statistics():
    cmds = []
    cmds.append("vif, uncentered")
    return cmds


def run_vce_corr_statistics():
    cmds = []
    cmds.append("estat vce, corr")
    return cmds


def get_transformed_variable(data: DataFrame, var: str, transformation: str) -> str:
    if transformation in ["log", "square", "boxcox"]:
        if f"{var}_{transformation}" in data.columns:
            return f"{var}_{transformation}"
        else:
            raise ValueError(f"{var}_{transformation} not in data columns")
    if transformation is None:
        return var
    else:
        raise NotImplementedError(f"{transformation} not implemented")


def indicator_to_stata(name: str) -> str:
    new_name = remove_chars(name, "&-,")
    splits = new_name.split(" ")
    new_name = "".join(
        [
            string[:3] if n != len(splits) - 1 else string
            for n, string in enumerate(splits)
            if string not in ["", " "]
        ]
    )
    return new_name


def map_variables_to_stata(
    variables_list: List[Dict[str, List[str]]],
    variables_map: Series,
    log: Optional[bool] = False,
):
    mapped_variables_list = []
    for variables in variables_list:
        for key, values in variables.items():
            log_key = f"{key}_log"
            key = log_key if log and log_key in variables_map.index else key
            key_map = variables_map.loc[key, :][0]
            values_map = []
            for val in values:
                log_val = f"{val}_log"
                val = log_val if log and log_val in variables_map.index else val
                values_map.append(variables_map.loc[val, :][0])
            mapped_variables_list.append({key_map: values_map})
    return mapped_variables_list


def xtreg_fe(dep_var: str, indep_vars: List[str]):
    return (
        f"xtreg {dep_var} "
        + " ".join([f"{indep_var}" for indep_var in indep_vars])
        + ", robust fe"
    )


def xtreg_fe_te(dep_var: str, indep_vars: List[str]):
    return (
        f"xtreg {dep_var} "
        + " ".join([f"{indep_var}" for indep_var in indep_vars])
        + " i.Year, robust fe"
    )


def xtdcc2(dep_var: str, indep_vars: List[str], lag: int) -> str:
    RUN_CS_ARDL = "capture noisily "
    RUN_CS_ARDL += f"xtdcce2 {dep_var}, lr(L.{dep_var} " + " ".join(
        [f"{indep_var} L.{indep_var}" for indep_var in indep_vars]
    )
    RUN_CS_ARDL += f") lr_options(ardl) cr({dep_var} " + " ".join(
        [f"{indep_var}" for indep_var in indep_vars]
    )
    RUN_CS_ARDL += f") cr_lags({lag})"
    return RUN_CS_ARDL


def run_panel_data_xtreg_commands(
    regressions_vars: Iterable[Dict[str, List[str]]],
    groupings: List[str],
    sub_groupings: Dict[str, List[str]],
    seci_cols: Dict[str, List[str]],
    regressions_id: str,
    logs_folder: PathLike,
    split_string: str,
    tag: str,
    model: str,
    db_path: PathLike,
) -> List:
    assert model in [
        "fe",
        "fe-te",
        "ardl",
    ], 'Model must be one of "fe", "fe-te" or "ardl"'
    commands_to_run = ["clear"]
    for group in groupings:
        log_file = f"{logs_folder}/{tag}_{model}_{group}_{regressions_id}.log"
        commands_to_run.append(f'quietly log using "{log_file}", replace')
        commands_to_run.append(f'quietly use "{db_path}", clear')
        commands_to_run.append("quietly levelsof CurrIncome, local(levels)")
        if group != "All countries":
            commands_to_run.append(f'quietly keep if CurrIncome == "{group}"')
        commands_to_run.append("encode Country, generate(id)")
        commands_to_run.append("xtset id Year")
        for regression_vars in regressions_vars:
            for dep_var, indep_vars in regression_vars.items():
                commands_to_run.append(split_string)
                if dep_var in sub_groupings.keys():
                    reg_indep_vars = [
                        c for c in seci_cols if c in sub_groupings[dep_var]
                    ]
                    reg_indep_vars.extend(indep_vars)
                else:
                    reg_indep_vars = [*seci_cols, *indep_vars]
                if model == "fe":
                    xtreg_command = xtreg_fe(dep_var, reg_indep_vars)
                elif model == "fe-te":
                    xtreg_command = xtreg_fe_te(dep_var, reg_indep_vars)
                elif model == "ardl":
                    xtreg_command = xtdcc2(dep_var, reg_indep_vars, lag=1)
                commands_to_run.append(xtreg_command)
                commands_to_run.append(split_string)
        commands_to_run.append("quietly log close")
    return commands_to_run
