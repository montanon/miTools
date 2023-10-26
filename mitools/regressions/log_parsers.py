import os
import re
import pandas as pd
import numpy as np
import time

from colorama import init, Fore
init(autoreset=True)

from ..utils import *
from .regressions_data import OLSResults, CSARDLResults
from typing import List, Dict, Match
from icecream import ic

SPLIT_PATTERN = '========================================================================================='
MODEL_PATTERN = f'({SPLIT_PATTERN}(\n)+){{1}}'
NUMBER_PATTERN = '-?\d*\.*\d+([Ee][\+\-]\d+)?'

OLS_VAR_NAMES = ['Coefficient', 'Std. err.', 't', 'P>|t|', '95% Conf. Low', '95% Conf. High']


def get_ols_data_from_log(ols_str: str):
    
    n_obs = get_numbers_from_str(re.search(rf'Number of obs += +\n* *-?\d*\,*\d*\.*\d+\n', ols_str).group(0))[-1]
    F_stats = get_numbers_from_str(re.search(r'F\(\d+, \d+\) += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    Prob_F = get_numbers_from_str(re.search(r'Prob > F += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    R_sq = get_numbers_from_str(re.search(r'R-squared += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    AdjR_sq = get_numbers_from_str(re.search(r'Adj R-squared += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    Root_MSE = get_numbers_from_str(re.search(r'Root MSE += *\n*-?\d*\.*\d*([eE][\+\-]\d+)?', ols_str).group(0))[-1]
    coefficients_table = ols_str.split('\n\n')[1]
    dep_variables = re.search(' +[A-Za-z0-9]+ +', coefficients_table).group(0).strip()

    coefficient_rows = coefficients_table.split('\n')[3:-2]
    coefficients = get_coefficients_from_table_rows(coefficient_rows, OLS_VAR_NAMES)
    
    indep_indep_variablesvars = list(coefficients.keys())

    model_stats = {}

    Root_MSE = None
    dep_variable = None
    indep_variables = None
    coefficients = None    
    std_errs = None
    t_values = None
    p_values = None
    significances = None
    conf_interval = None
    model_params = None
    model_specification = None    

    return OLSResults(
        model_stats=model_stats,
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
        model_params=model_params,
        model_specification=model_specification,
    )

def get_coefficients_from_table_rows(coefficient_rows: List[str], var_names: List[str]) -> Dict:
    coefficients = {}
    for row in coefficient_rows:
        variable = re.match(' *[A-Za-z0-9\_\~.]+ *(?=\|)', row)
        end_of_variable = variable.end()
        variable = variable.group(0).strip()
        coeffs = re.findall('-?\d*\.?\d+', row[end_of_variable:])
        coeffs = [float(c) for c in coeffs]
        coeffs = {v: c for v, c in zip(var_names, coeffs)}
        coefficients[variable] = coeffs
    return coefficients

def get_csardl_data_from_log(csardl_str):

    var_names = ['Coef.', 'Std. Err.', 'z', 'P>|z|', '95% Conf. Low', '95% Conf. High']

    if 'No observations left' in csardl_str or 'conformability error' in csardl_str: 
        return CSARDLResults(
            model_stats=None,
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
            command=0,
            lag=99,
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

    model_specification = re.search(r'Command: .*', csardl_str).group(0)
    model_stats = {}
    model_stats['lag'] = int(model_specification[-2:-1])

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
        model_stats=model_stats,

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

def dict_to_df(model_dict: Dict):
    base_data = {
        'n_obs': model_dict['n_obs'],
        'n_groups': model_dict['n_groups'],
        'obs_p_group': model_dict['obs_p_group'],
        'F_stats': model_dict['F_stats'],
        'Prob_F': model_dict['Prob_F'],
        'R_sq': model_dict['R_sq'],
        'R_sqMG': model_dict['R_sqMG'],
        'RootMSE': model_dict['RootMSE'],
        'CD_stats': model_dict['CD_stats'],
        'p_val': model_dict['p_val'],
        'lag': model_dict['lag'],
        'command': model_dict['command']
    }
    rows = []
    for var, stats in model_dict['short_run_coeffs'].items():
        row = {'variable': var, 'type': 'short_run'}
        row.update(stats)
        rows.append(row)
    for var, stats in model_dict['adj_term_coeffs'].items():
        row = {'variable': var, 'type': 'adj_term'}
        row.update(stats)
        rows.append(row)
    for var, stats in model_dict['long_run_coeffs'].items():
        row = {'variable': var, 'type': 'long_run'}
        row.update(stats)
        rows.append(row)
    df = pd.DataFrame(rows)
    for key, value in base_data.items():
        df[key] = value
        
    df.index = pd.MultiIndex.from_product([list([model_dict['dep_var']]), [model_dict['lag']], df['type'].values])
    relevant_cols = ['variable', 'Coef.', 'P>|z|']
    df = df[relevant_cols]
    df.index.names = ['Dep Var', 'Lag', 'Time Span']
    df = df.set_index('variable', append=True)
    if model_dict['indep_vars'] == 0:
        model_dict['indep_vars'] = [str(model_dict['indep_vars'])]
    df['Indep Var'] = [v for v in model_dict['indep_vars'] if v.find('.') == -1][0]
    df = df.set_index('Indep Var', append=True)
    return df

def add_significance(row):
    p_value = float(row.split(' ')[1].replace('(','').replace(')',''))  
    if p_value < 0.01:
        return row + "***"
    elif p_value < 0.05:
        return row + "**"
    elif p_value < 0.1:
        return row + "*"
    else:
        return row

def generate_significance_color_styles(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            val = df.iloc[r, c]
            if isinstance(val, str) and not df.index[r][0] in df.index[r][-1]:
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

def read_regressions_log(log):
    
    dataframes = []
    
    count = 0
    while log:       
        if count <= 51:
            regression_str = extract_regression_from_log(log)
            ols_str, csardl_str = get_models_from_regression(regression_str)

            ols_data = get_ols_data_from_log(ols_str)
            try:
                csardl_data = get_csardl_data_from_log(csardl_str)
            except Exception as e:
                print(csardl_str)
                raise Exception(str(e))
            csardl_df = regression_data_to_df(csardl_data)
            
            dataframes.append(csardl_df)
            csardl_df = csardl_df.style.apply(lambda _: generate_significance_color_styles, axis=None)

            log = log[len(regression_str)-270:]
        else:
            log = False

        count += 1
    return pd.concat(dataframes)

def extract_regression_from_log(log):

    start_of_reg = "(=+\n){2,3}"
    midd_of_reg = "(=+\n){1}"
    end_of_reg = "(=+\n){2,3}"

    start_match = re.search(start_of_reg, log, re.DOTALL)
    after_start_str = log[start_match.end():]
    middle_match = re.search(midd_of_reg, after_start_str, re.DOTALL)
    after_middle_str = after_start_str[middle_match.end():]
    end_match = re.search(end_of_reg, after_middle_str, re.DOTALL)
    
    match = log[start_match.start():start_match.end()+middle_match.end()+end_match.end()]

    return match

def get_models_from_regression(regression_str):
    no_borders = re.sub('(====+\n){2,3}', '', regression_str)
    split = re.split('====+\n{1,}', no_borders)
    ols_str, csardl_str = split
    return ols_str, csardl_str

def regression_data_to_df(regression_data):
    regression_data = dict_to_df(regression_data)
    regression_data['Result'] = regression_data['Coef.'].round(2).astype(str) + ' (' + regression_data['P>|z|'].astype(str) + ')'
    regression_data['Result'] = regression_data['Result'].apply(lambda x: add_significance(x))
    regression_data = regression_data[['Result']]
    return regression_data

def df_selection(df, indicators, columns, col_filters, index_filters):
    _df = df.unstack(columns).loc[pd.IndexSlice[indicators,:]]
    for col, values in col_filters.items():
        if values:
            mask = (_df.columns.get_level_values(col).isin(values))
            _df = _df.loc[:, mask]
    for idx, values in index_filters.items():
        if values:
            mask = (_df.index.get_level_values(idx).isin(values))
            _df = _df.loc[mask, :]
    return _df

def df_view(df, indicators, columns, col_filters, index_filters, col_name, row_name):
    _df = df_selection(df, indicators, columns, col_filters, index_filters)
    _df = sort_df(_df, row_name, 0, indep_var_sorting_key)
    _df = sort_df(_df, col_name, 1, income_sorting_key)
    _df = style_csardl_results(_df, row_name, col_name)
    return _df

def sort_df(df, column, axis, key):
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
    return pd.Index([string_value[v] for v in val], name='Indep Var')

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
