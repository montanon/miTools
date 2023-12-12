import hashlib
import os
import re
import time
from os import PathLike
from typing import Dict, List

import numpy as np
import pandas as pd
from fuzzywuzzy import process

from ..utils import *
from .regressions_data import CSARDLResults, OLSResults, RegressionData, XTRegResults

#SPLIT_PATTERN = '={2,}\n'
#MODEL_PATTERN = f'({SPLIT_PATTERN}(\n)+){{1}}'
NUMBER_PATTERN = '-?\d*\.?\d+(?:[Ee][\+\-]?\d+)?'#'-?\d*\.*\d*([Ee][\+\-]\d+)?'
#REGRESSION_PATTERN = f'({SPLIT_PATTERN}){{2}}(.*?)({SPLIT_PATTERN}){{1}}(.*?)({SPLIT_PATTERN}){{2}}'

SPLIT_PATTERN = '''===============================================================================\n> ==========\n'''
REGRESSION_PATTERN = f'({SPLIT_PATTERN}){{2}}(.*?)({SPLIT_PATTERN}){{2}}'#(.*?)({SPLIT_PATTERN}){{1}}'

OLS_VAR_NAMES = ['Coefficient', 'Std. err.', 't', 'P>|t|', '95% Conf. Low', '95% Conf. High']
XTREG_VAR_NAMES = ['Coefficient', 'std. err.', 't', 'P>|t|', '95% Conf. Low', '95% Conf. High']

    
def process_logs_folder(folder: PathLike):
    logs_paths = [f"{folder}/{f}" for f in os.listdir(f'{folder}') if f.endswith('.log')]
    ols_df, csardl_df = process_logs(logs_paths)
    return ols_df, csardl_df

def threaded_process_logs(logs_paths, model_types, batch_size: Optional[int]=1, n_threads: Optional[int]=1):
    if n_threads > 1:
        parallel_function = parallel(n_threads, batch_size)(process_logs)
        return parallel_function(logs_paths, model_types)
    return process_logs(logs_paths, model_types)

def process_logs(logs_paths, split_str: Optional[str]=None):
    dataframes = []
    for log_path in logs_paths:
        print(log_path)
        regressions_data = process_log(log_path, split_str)
        group = regressions_data.index.get_level_values('Group').unique()[0]
        dataframes.append(regressions_data)
    dataframes = pd.concat(remove_dataframe_duplicates(dataframes))
    return dataframes

def process_log(log_path: PathLike, split_str: Optional[str]=None):
    log = load_log(log_path)
    tag, model_type, group, regression_id = get_log_data_from_path(log_path)
    regression_strs = get_splits_from_regressions_str(log, split_str)
    regression_data = []
    for n, regression_str in enumerate(regression_strs):
        regression = process_regression_str(regression_str, model_type)
        regression['Id'] = generate_hash_from_dataframe(regression)
        regression_data.append(regression)
    regression_data = pd.concat(remove_dataframe_duplicates(regression_data))
    regression_data['Group'] = group
    regression_data['Type'] = model_type
    regression_data = regression_data.set_index(['Group', 'Id', 'Type'], append=True)
    return regression_data

def load_log(log_path): return read_text_file(log_path)

def get_log_data_from_path(log_path: PathLike):
    tag, model_type, group, regression_id = os.path.basename(log_path.replace('.log', '')).split('_')
    return tag, model_type, group, regression_id

def process_regression_str(regression_str: str, model_type: str) -> Dict[str,List]:
    regression_data = get_model_data_from_log(regression_str, model_type)
    regression_data = regression_data.to_pretty_df()
    return regression_data

def get_splits_from_regressions_str(regression_str, split_pattern=None):
    if split_pattern is None:
        split_pattern = SPLIT_PATTERN
    splits = re.split(f'({split_pattern}){{1}}', regression_str)
    splits = [split for split in splits[1:] if split not in ['\n', '', split_pattern]]
    return splits

def get_regression_strs_from_log(log: str, regression_pattern=None):
    if regression_pattern is None:
        regression_pattern = SPLIT_PATTERN
    regression_strs = []
    while len(log) > 300:
        match = re.search(regression_pattern, log, re.DOTALL)
        regression_strs.append(match[0])
        log = log[match.end():]
    return regression_strs

def get_coefficients_from_table_rows(coefficient_rows: List[str], var_names: List[str]) -> Dict:
    coefficients = {}
    for row in coefficient_rows:
        variable = re.match(' *[A-Za-z0-9\_\~.]+ *(?=\|)', row)
        end_of_variable = variable.end()
        variable = variable.group(0).strip()
        if not '(omitted)' in row[end_of_variable:]:
            coeffs = re.findall(NUMBER_PATTERN, row[end_of_variable:])
            _coeffs = coeffs
            coeffs = [float(c) if c != '.' else 0.0 for c in coeffs]
        else:
            coeffs = [0, 9999, 0, 1.0, 0, 0]
        coeffs = {v: c for v, c in zip(var_names, coeffs)}
        coefficients[variable] = coeffs
    return coefficients

def get_model_data_from_log(regression_str, model_type):
    if model_type == 'csardl':
        return get_csardl_data_from_log(regression_str)
    elif model_type == 'xtreg':
        return get_xtreg_data_from_log(regression_str)
    elif model_type == 'ols':
        return get_ols_data_from_log(regression_str)
    else:
        raise NotImplementedError

def get_xtreg_data_from_log(xtreg_str: str):
    n_obs = get_numbers_from_str(re.search(rf'Number of obs += +\n* *-?\d*\,*\d*\.*\d+\n', xtreg_str).group(0))[-1]
    n_groups = get_numbers_from_str(re.search(rf'Number of groups += +\n*(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    try:
        n_obs_per_group = get_numbers_from_str(re.search(r'Obs per group \(T\) += +-?\d*\.*\d+\n', xtreg_str).group(0))[-1]
    except AttributeError:
        n_obs_per_group = get_numbers_from_str(re.search(rf'avg *= +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    F_stats = get_numbers_from_str(re.search(rf'F\(-?\d+, -?\d+\) += +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    Prob_F = get_numbers_from_str(re.search(rf'Prob > F += +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    R_sq = get_numbers_from_str(re.search(rf'Overall += +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    R_sq_within = get_numbers_from_str(re.search(rf'Within += +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    R_sq_between = get_numbers_from_str(re.search(rf'Between += +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1]
    corr = get_numbers_from_str(re.search(rf'corr\((.*?)= +(({NUMBER_PATTERN})|(.))+\n', xtreg_str).group(0))[-1] 

    if xtreg_str.startswith('note:'):
        xtreg_str = '\n'.join(xtreg_str.split('\n')[1:])

    coefficients_table = xtreg_str.split('\n\n')[3]
    dep_variable = re.search('(Indicat(?:~\d+X|or\w+X))|(ECI)', coefficients_table).group(0).strip()

    coefficient_rows = coefficients_table.split('\n')[5:-6]
    coefficients = get_coefficients_from_table_rows(coefficient_rows, XTREG_VAR_NAMES)
    indep_variables = list(coefficients.keys())

    std_errs = None
    t_values = None
    p_values = None
    significances = None
    conf_interval = None

    model_params = None
    model_specification = None

    return XTRegResults(
        n_obs=n_obs,
        n_groups=n_groups,
        n_obs_per_group=n_obs_per_group,
        F_stats=F_stats,
        Prob_F=Prob_F,
        R_sq=R_sq,
        R_sq_within=R_sq_within,
        R_sq_between=R_sq_between,
        corr=corr,
        dep_variable=dep_variable,
        indep_variables=indep_variables,
        coefficients=coefficients,
        std_errs=std_errs,
        t_values=t_values,
        p_values=p_values,
        significances=significances,
        conf_interval=conf_interval,
        model_params=model_params,
        model_specification=model_specification
    )
    

def get_ols_data_from_log(ols_str: str):
    
    n_obs = get_numbers_from_str(re.search(rf'Number of obs += +\n* *-?\d*\,*\d*\.*\d+\n', ols_str).group(0))[-1]
    F_stats = get_numbers_from_str(re.search(r'F\(\d+, \d+\) += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    Prob_F = get_numbers_from_str(re.search(r'Prob > F += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    R_sq = get_numbers_from_str(re.search(r'R-squared += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    AdjR_sq = get_numbers_from_str(re.search(r'Adj R-squared += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    Root_MSE = get_numbers_from_str(re.search(r'Root MSE += *\n*-?\d*\.*\d*([eE][\+\-]\d+)?', ols_str).group(0))[-1]
    coefficients_table = ols_str.split('\n\n')[1]

    dep_variable = re.search('Indicat(?:~\d+X|or\w+X)', coefficients_table).group(0).strip()

    coefficient_rows = coefficients_table.split('\n')[3:-2]
    coefficients = get_coefficients_from_table_rows(coefficient_rows, OLS_VAR_NAMES)
    
    indep_variables = list(coefficients.keys())\

    model_params = {}
 
    std_errs = None
    t_values = None
    p_values = None
    significances = None
    conf_interval = None
    model_specification = None    

    return OLSResults(
        model_params=model_params,
        n_obs=n_obs,
        F_stats=F_stats,
        Prob_F=Prob_F,
        R_sq=R_sq,
        AdjR_sq=AdjR_sq,
        Root_MSE=Root_MSE,
        dep_variable=dep_variable,
        indep_variables=indep_variables,
        coefficients=coefficients,
        std_errs=std_errs,
        t_values=t_values,
        p_values=p_values,
        significances=significances,
        conf_interval=conf_interval,
        model_specification=model_specification
    )

def get_csardl_data_from_log(csardl_str):

    var_names = ['Coef.', 'Std. Err.', 'z', 'P>|z|', '95% Conf. Low', '95% Conf. High']

    if 'No observations left' in csardl_str or 'conformability error' in csardl_str: 
        return CSARDLResults(
            model_params={},
            model_specification='No observations left or conformability error',
            n_obs=0,
            n_groups=0,
            n_obs_per_group=0,
            F_stats=0,
            Prob_F=0,
            R_sq=0,
            R_sq_MG=0,
            Root_MSE=0,
            CD_stats=0,
            p_value=0,
            dep_variable=0,
            indep_variables=0,
            short_run_coefficients={'Empty': {k: 99 for k in var_names}},
            adj_term_coefficients={'Empty': {k: 99 for k in var_names}},
            long_run_coefficients={'Empty': {k: 99 for k in var_names}},

            short_run_std_errs=None,
            short_run_z_values=None,
            short_run_p_values=None,
            short_run_significances=None,
            short_run_conf_intervals=None,

    
            long_run_std_errs=None,
            long_run_z_values=None,
            long_run_p_values=None,
            long_run_significances=None,
            long_run_conf_intervals=None,

            adj_term_std_errs=None,
            adj_term_z_values=None,
            adj_term_p_values=None,
            adj_term_significances=None,
            adj_term_conf_intervals=None,
    )
    
    n_obs = get_numbers_from_str(re.search(rf'Number of obs += +\n*(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    n_groups = get_numbers_from_str(re.search(rf'Number of groups += +\n*(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    try:
        n_obs_per_group = get_numbers_from_str(re.search(r'Obs per group \(T\) += +-?\d*\.*\d+\n', csardl_str).group(0))[-1]
    except AttributeError:
        n_obs_per_group = get_numbers_from_str(re.search(rf'avg *= +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    F_stats = get_numbers_from_str(re.search(rf'F\(-?\d+, -?\d+\) += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    Prob_F = get_numbers_from_str(re.search(rf'Prob > F += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    R_sq = get_numbers_from_str(re.search(rf'R-squared += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    R_sq_MG = get_numbers_from_str(re.search(rf'R-squared \(MG\) += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    Root_MSE = get_numbers_from_str(re.search(rf'Root MSE += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    try:
        CD_stats = get_numbers_from_str(re.search(rf'CD Statistic += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    except AttributeError as e:
        CD_stats = 0
    try:
        p_value = get_numbers_from_str(re.search(rf'p-value += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    except AttributeError as e:
        p_value = 0

    coefficients_table = re.split('(?<!-)(?=-{2,}\n +[A-Za-z0-9]+ *\|)', csardl_str)[1]
    dep_variable = re.search(' +[A-Za-z0-9]+', coefficients_table).group(0).strip()

    coefficients_table_lines = coefficients_table.split('\n')

    short_run_line = find_str_line_number_in_text(coefficients_table, 'Short Run Est.')
    adj_term_line = find_str_line_number_in_text(coefficients_table, 'Adjust. Term')
    long_run_line = find_str_line_number_in_text(coefficients_table, 'Long Run Est.')
    end_table_line = find_str_line_number_in_text(coefficients_table, 'Mean Group Variables:')

    short_run_coefficients = coefficients_table_lines[short_run_line+3:adj_term_line-1]
    adj_term_coefficients = coefficients_table_lines[adj_term_line+3:long_run_line-1]
    long_run_coefficients = coefficients_table_lines[long_run_line+3:end_table_line-1]
    
    short_run_coefficients = get_coefficients_from_table_rows(short_run_coefficients, var_names)
    adj_term_coefficients = get_coefficients_from_table_rows(adj_term_coefficients, var_names)
    long_run_coefficients = get_coefficients_from_table_rows(long_run_coefficients, var_names)
    indep_variables = list(short_run_coefficients.keys())

    model_params = {}
    try:
        model_params['lag'] = int(csardl_str.split('cr_lags(')[1][0])
    except Exception:
        model_params['lag'] = 0

    try:
        model_specification = re.search('capture noisily: ', csardl_str, re.DOTALL).group(0)
        model_specification = model_specification.split('\n(Dynamic)')[0]
    except Exception:
        model_specification = {}

    short_run_std_errs=None
    short_run_z_values=None
    short_run_p_values=None
    short_run_significances=None
    short_run_conf_intervals=None

    long_run_std_errs=None
    long_run_z_values=None
    long_run_p_values=None
    long_run_significances=None
    long_run_conf_intervals=None

    adj_term_std_errs=None
    adj_term_z_values=None
    adj_term_p_values=None
    adj_term_significances=None
    adj_term_conf_intervals=None
    
    return CSARDLResults(
        model_params=model_params,

        n_obs=n_obs,
        n_groups=n_groups,
        n_obs_per_group=n_obs_per_group,
        F_stats=F_stats,
        Prob_F=Prob_F,
        R_sq=R_sq,
        R_sq_MG=R_sq_MG,
        Root_MSE=Root_MSE,
        CD_stats=CD_stats,
        p_value=p_value,
        
        dep_variable=dep_variable,
        indep_variables=indep_variables,
        
        short_run_coefficients=short_run_coefficients,
        short_run_std_errs=short_run_std_errs,
        short_run_z_values=short_run_z_values,
        short_run_p_values=short_run_p_values,
        short_run_significances=short_run_significances,
        short_run_conf_intervals=short_run_conf_intervals,

        long_run_coefficients=long_run_coefficients,
        long_run_std_errs=long_run_std_errs,
        long_run_z_values=long_run_z_values,
        long_run_p_values=long_run_p_values,
        long_run_significances=long_run_significances,
        long_run_conf_intervals=long_run_conf_intervals,

        adj_term_coefficients=adj_term_coefficients,
        adj_term_std_errs=adj_term_std_errs,
        adj_term_z_values=adj_term_z_values,
        adj_term_p_values=adj_term_p_values,
        adj_term_significances=adj_term_significances,
        adj_term_conf_intervals=adj_term_conf_intervals,
        model_specification=model_specification,
    )

def generate_significance_color_styles(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            val = df.iloc[r, c]
            if isinstance(val, str) and 'ECI' in df.index[r][-1] or 'SCP' in df.index[r][-1] or 'SCI' in df.index[r][-1] or 'PSEV' in df.index[r][-1]:
                pos_value = float(val.split(' ')[0]) >= 0.0
                if '***' in val:
                    val_style = 'background-color: limegreen; font-weight: bold;' if pos_value else 'background-color: red; font-weight: bold;'
                elif '**' in val:
                    val_style = 'background-color: springgreen; font-weight: bold;' if pos_value else 'background-color: orangered; font-weight: bold;'
                elif '*' in val:
                    val_style = 'background-color: aquamarine; font-weight: bold;' if pos_value else 'background-color: salmon; font-weight: bold;'
                else:
                    val_style = ''
            else:
                val_style = ''
            styles.iloc[r, c] = val_style
    return styles
    
def df_selection(df, indicators, columns, col_filters, index_filters): 
    _df = df.unstack(columns).loc[pd.IndexSlice[:,:,indicators,:]]
    for col, values in col_filters.items():
        if values:
            mask = (_df.columns.get_level_values(col).isin(values))
            _df = _df.loc[:, mask]
    for idx, values in index_filters.items():
        if values:
            mask = (_df.index.get_level_values(idx).isin(values))
            _df = _df.loc[mask, :]
    return _df

def get_ids_with_containing_string(df, text_column, substring, id_col):
    filtered_df = df[df.index.get_level_values(text_column).str.contains(substring)]
    unique_ids = filtered_df.index.get_level_values(id_col).unique()
    return unique_ids

def df_view(df, indicators, columns, col_filters, index_filters, eci_col, col_name, row_name):
    _df = df_selection(df, indicators, columns, col_filters, index_filters)
    eci_ids = get_ids_with_containing_string(_df, 'Variable', eci_col, 'Id')
    _df = _df.loc[_df.index.get_level_values('Id').isin(eci_ids), :]
    _df = sort_df(_df, col_name, 1, income_sorting_key)
    _df = sort_df(_df, [row_name, 'Variable'], 0)
    sorted_ids = _df.reset_index().groupby('Id')['Variable'].count().sort_values(axis=0, ascending=True).index
    _df = _df.loc[sorted_ids, :]
    _df = style_csardl_results(_df, row_name, col_name)
    return _df

def sort_df(df, column, axis, key=None):
    return df.sort_values(by=column, axis=axis, key=key)

def income_sorting_key(val):
    string_value = {
        'All Variations': 99,
        'All countries': 5,
        'High income': 4,
        'Upper middle income': 3,
        'Lower middle income': 2,
        'Low income': 1
    }
    return pd.Index([string_value[v] for v in val], name='Income')

def indep_var_sorting_key(val):
    string_value = {
        'ECI': 0,
        'Agriculture ECI': 1,
        'Fishing ECI': 2,
        'Food & Beverages ECI': 3,
        'Machinery ECI': 4,
        'Metal Products ECI': 5,
        'Mining & Quarrying ECI': 6,
        'Other Manufacturing ECI': 7, 
        'Petroleum, Chemicals & Non-Metals ECI': 8, 
        'Textiles & Wearing Apparel ECI': 9,
        'Transport Equipment ECI': 10,
        'Wood & Paper ECI': 11,
        'SECI': 12
    }
    return pd.Index([string_value[v] if v in string_value else 99 for v in val], name='Indep Var')

def generate_styling(df, row_level, col_level):
    row_level = df.index.names.index(row_level)
    col_level = df.columns.names.index(col_level)
    
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            
            if r > 0 and df.index[r][row_level] != df.index[r - 1][row_level]:
                styles.iloc[r, c] += 'border-top: 2px solid black; '

            if c > 0 and df.columns[c][col_level] != df.columns[c - 1][col_level]:
                styles.iloc[r, c] += 'border-left: 2px solid black; '

    return styles

def style_multiindex_borders(df, row_level_name, col_level_name):
    
    row_level = df.index.names.index(row_level_name)
    col_level = df.columns.names.index(col_level_name)
    
    row_styles = [{"selector": f"tbody tr th.level{i}", "props": [
        ("border-top", "2px solid black"),
        ("border-right", "2px solid black"),
        ("border-left", "2px solid black"),
        ("border-bottom", "2px solid black")
    ]} for i in range(row_level-1, df.index.nlevels+1)]
    col_styles = [{"selector": f"thead th.level{i}", "props": [
        ("border-top", "2px solid black"),
        ("border-right", "2px solid black"),
        ("border-left", "2px solid black"),
        ("border-bottom", "2px solid black"),
        ('text-align', 'left')
    ]} for i in range(col_level-1, df.columns.nlevels+1)]
    
    return row_styles + col_styles

def style_csardl_results(selection, row_level_name, col_level_name):
    color_style = generate_significance_color_styles(selection)
    cell_border_style = generate_styling(selection, row_level_name, col_level_name)
    indexes_style = style_multiindex_borders(selection, row_level_name, col_level_name)
    return selection.style.apply(lambda x: color_style + cell_border_style, axis=None).set_table_styles(indexes_style)

def save_dfs_to_excel(dataframes, sheet_names, path):
    if len(dataframes) != len(sheet_names):
        raise ValueError("The number of dataframes and sheet names must be the same.")
    with pd.ExcelWriter(path) as writer:
        for df, sheet_name in zip(dataframes, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name)

def mask_results(dataframe, indicator_names):
    dataframe = dataframe.reset_index()
    dataframe['Variable'] = dataframe['Variable'].apply(lambda x: ind_var_name_replace(x, indicator_names))
    dataframe['Dep Var'] = dataframe['Dep Var'].apply(lambda x: ind_var_name_replace(x, indicator_names))
    dataframe['Indep Var'] = dataframe['Indep Var'].map(indicator_names.to_dict()['Original Name'])
    dataframe['Variable'] = dataframe['Variable'].apply(lambda x: ind_var_name_replace(x, indicator_names))
    if 'Lag' in dataframe.columns:
        dataframe['Lag'] = dataframe['Lag'].astype(str) + '-year Lag'
    dataframe = dataframe.set_index([c for c in dataframe.columns if c != 'Result'])
    return dataframe

def ind_var_name_replace(string, indicator_names):
    ind_mapping = {
        'PetChe~SSECI': 'PetCheNonSSECI',
        'TexWea~SSECI': 'TexWeaAppSSECI',
        'PetChe~nSECI': 'PetCheNonSECI',
        'TexWea~pSECI': 'TexWeaAppSECI',
        'PetChe~NSECI': 'PetCheNonNSECI',
        'TexWea~NSECI': 'TexWeaAppNSECI',
        'PetCheN~NSCI': 'PetCheNonNSCI',
        'TexWeaA~NSCI': 'TexWeaAppNSCI',
        'PetCheN~SSCI': 'PetCheNonSSCI',
        'TexWeaA~SSCI': 'TexWeaAppSSCI',
        'PetCheN~SSCP': 'PetCheNonSSCP',
        'TexWeaA~SSCP': 'TexWeaAppSSCP',
        'PetCheNonP~V': 'PetCheNonPSEV',
        'TexWeaAppP~V': 'TexWeaAppPSEV',
        'PetChe~SASCI': 'PetCheNonSASCI',
        'TexWea~SASCI': 'TexWeaAppSASCI',
        'PetChe~NASCI': 'PetCheNonNASCI',
        'TexWea~NASCI': 'TexWeaAppNASCI'

    }
    if string in ind_mapping:
        indicator_name = indicator_names.to_dict()['Original Name'][ind_mapping[string]]
        return indicator_name
    search_indicator = re.search('Indic[ator]*~?\d{1,}X', string)
    search_eci = re.search('[A-Za-z& \-,]*(ASCI|SECI|ECI|SCI|SCP|PSEV)', string)
    if search_indicator:
        indicator = search_indicator.group(0)
        indicator = re.sub('Indic[ato]*~+r*', 'Indicator', indicator)
        indicator_name = indicator_names.to_dict()['Original Name'][indicator]
        string = re.sub('Indic[a-z\~]+\d{1,}X', indicator_name, string)
    elif search_eci:
        indicator = search_eci.group(0)
        indicators_dict = indicator_names.to_dict()['Original Name']
        if indicator not in indicators_dict.values() and indicator in indicators_dict.keys():
            indicator_name = indicators_dict[indicator]
        elif indicator in indicators_dict.values():
            indicator_name = indicator
        else:
            print(indicator, search_eci, string)
            raise Exception
        string = re.sub('(?<=[._])?[A-Za-z& ]*(ASCI|SECI|ECI|SCI|SCP|PSEV)', indicator_name, string)
        if string.find('~') > -1:
            print(string)
    return string

def has_duplicated_indices(df: DataFrame):
    return df.index.duplicated().any()

def print_duplicated_indices(df):
    duplicated = df.index[df.index.duplicated()].unique()
    for idx in duplicated:
        print(idx)

def generate_hash_from_dataframe(df):
    dep_var = df.index.get_level_values('Dep Var').unique()[0]
    indep_vars = ' '.join(df.index.get_level_values('Indep Var').unique())
    variables = ' '.join(df.index.get_level_values('Variable').unique())
    hasher = hashlib.md5()
    hasher.update(rf'{dep_var} {indep_vars} {variables}'.encode('utf-8'))
    return hasher.hexdigest()[:6]
