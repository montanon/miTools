import os
import re
import pandas as pd
import numpy as np
import time

from colorama import init, Fore
init(autoreset=True)

from ..utils import *
from typing import List, Dict, Match
from icecream import ic

SPLIT_PATTERN = '========================================================================================='
MODEL_PATTERN = f'({SPLIT_PATTERN}(\n)+){{1}}'
NUMBER_PATTERN = '-?\d*\.*\d+([Ee][\+\-]\d+)?'

OLS_VAR_NAMES = ['Coefficient', 'Std. err.', 't', 'P>|t|', '95% Conf. Low', '95% Conf. High']


def get_ols_data_from_log(ols_str: str):
    
    n_obs = get_numbers_from_str(re.search(r'Number of obs += +\n* *-?\d*\,*\d*\.*\d+\n', ols_str).group(0))[-1]
    F_stats = get_numbers_from_str(re.search(r'F\(\d+, \d+\) += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    Prob_F = get_numbers_from_str(re.search(r'Prob > F += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    R_sq = get_numbers_from_str(re.search(r'R-squared += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    AdjR_sq = get_numbers_from_str(re.search(r'Adj R-squared += +-?\d*\.*\d+\n', ols_str).group(0))[-1]
    RootMSE = get_numbers_from_str(re.search(r'Root MSE += *\n*-?\d*\.*\d*([eE][\+\-]\d+)?', ols_str).group(0))[-1]
    coefficients_table = ols_str.split('\n\n')[1]
    dep_var = re.search(' +[A-Za-z0-9]+ +', coefficients_table).group(0).strip()

    coefficient_rows = coefficients_table.split('\n')[3:-2]
    coefficients = get_coefficients_from_table_rows(coefficient_rows, OLS_VAR_NAMES)
    
    indep_vars = list(coefficients.keys())

    return dict(
        n_obs=n_obs,
        F_stats=F_stats,
        Prob_F=Prob_F,
        R_sq=R_sq,
        AdjR_sq=AdjR_sq,
        RootMSE=RootMSE,
        dep_var=dep_var,
        indep_vars=indep_vars,
        coeffs=coefficients
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
        return dict(
        n_obs=0,
        n_groups=0,
        obs_p_group=0,
        F_stats=0,
        Prob_F=0,
        R_sq=0,
        R_sqMG=0,
        RootMSE=0,
        CD_stats=0,
        p_val=0,
        dep_var=0,
        indep_vars=0,
        command=0,
        lag=99,
        short_run_coeffs={'Empty': {k: 99 for k in var_names}},
        adj_term_coeffs={'Empty': {k: 99 for k in var_names}},
        long_run_coeffs={'Empty': {k: 99 for k in var_names}}
    )
    
    n_obs = get_numbers_from_str(re.search(rf'Number of obs += +\n*(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    n_groups = get_numbers_from_str(re.search(rf'Number of groups += +\n*(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    try:
        obs_p_group = get_numbers_from_str(re.search(r'Obs per group \(T\) += +-?\d*\.*\d+\n', csardl_str).group(0))[-1]
    except AttributeError:
        obs_p_group = get_numbers_from_str(re.search(rf'avg *= +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    F_stats = get_numbers_from_str(re.search(rf'F\(-?\d+, -?\d+\) += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    Prob_F = get_numbers_from_str(re.search(rf'Prob > F += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    R_sq = get_numbers_from_str(re.search(rf'R-squared += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    R_sqMG = get_numbers_from_str(re.search(rf'R-squared \(MG\) += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    RootMSE = get_numbers_from_str(re.search(rf'Root MSE += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    try:
        CD_stats = get_numbers_from_str(re.search(rf'CD Statistic += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    except AttributeError as e:
        CD_stats = 0
    try:
        p_val = get_numbers_from_str(re.search(rf'p-value += +(({NUMBER_PATTERN})|(.))+\n', csardl_str).group(0))[-1]
    except AttributeError as e:
        p_val = 0

    coefficients_table = re.split('(?<!-)(?=-{2,}\n +[A-Za-z0-9]+ *\|)', csardl_str)[1]
    dep_var = re.search(' +[A-Za-z0-9]+', coefficients_table).group(0).strip()

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
    indep_vars = list(short_run_coefficients.keys())

    command = re.search(r'Command: .*', csardl_str).group(0)
    lag = int(command[-2:-1])
    
    return dict(
        n_obs=n_obs,
        n_groups=n_groups,
        obs_p_group=obs_p_group,
        F_stats=F_stats,
        Prob_F=Prob_F,
        R_sq=R_sq,
        R_sqMG=R_sqMG,
        RootMSE=RootMSE,
        CD_stats=CD_stats,
        p_val=p_val,
        dep_var=dep_var,
        indep_vars=indep_vars,
        command=command,
        lag=lag,
        short_run_coeffs=short_run_coefficients,
        adj_term_coeffs=adj_term_coefficients,
        long_run_coeffs=long_run_coefficients
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
    
def color_by_significance(var):
    styles = []
    for val in var:
        try:
            if not isinstance(val, float) and var.name[0] not in var.name[-1]:
                if '***' in val:
                    val_style = 'background-color: limegreen'
                elif '**' in val:
                    val_style = 'background-color: springgreen'
                elif '*' in val:
                    val_style = 'background-color: aquamarine'
                else:
                    val_style = ''
            else:
                val_style = ''
        except Exception as e:
            print(str(e))
            print(np.isnan(val))
            print(val)
            print(type(val))
            print(var)
        styles.append(val_style)
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
            csardl_df = csardl_df.style.apply(color_by_significance, axis=1)

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
