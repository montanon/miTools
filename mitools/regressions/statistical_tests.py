import re
from typing import Dict, Optional, Tuple

import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame, Series
from scipy.stats import anderson, shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller


def shapiro_test(data: Series, alpha: float=0.05) -> Tuple[float, float]:
    stat, p = shapiro(data)
    return {'statistic': stat, 'p-value': p}

def shapiro_test_dataframe_groups(data: Dict[str, DataFrame], criteria: Optional[float]=0.05):
    shapiro_tests = []
    for group, group_data in data.items():
        statistics = group_data.apply(shapiro_test, axis=0, result_type='expand')
        statistics = statistics.T
        statistics['hypothesis'] = statistics['p-value'].apply(lambda x: 'Reject' if x < criteria else 'Accept')
        statistics = statistics.T
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        shapiro_tests.append(statistics)
    shapiro_tests = pd.concat(shapiro_tests, axis=0)
    return shapiro_tests

def anderson_test(data: Series, criteria: Optional[float]=0.01) -> Dict[str, float]:
    normal_critical_values = [0.15, 0.1, 0.05, 0.025, 0.01]
    result = anderson(data, dist='norm')
    return {'statistic': result.statistic, 'critical_value': result.critical_values[normal_critical_values.index(criteria)]}

def anderson_test_dataframe_groups(data: Dict[str, DataFrame], criteria: Optional[float]=0.01):
    anderson_tests = []
    for group, group_data in data.items():
        statistics = group_data.apply(anderson_test, args=(criteria,), axis=0, result_type='expand')
        statistics = statistics.T
        statistics['hypothesis'] = statistics.apply(lambda x: 'Reject' if x['statistic'] > x['critical_value'] else 'Accept', axis=1)
        statistics = statistics.T  
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        anderson_tests.append(statistics)
    anderson_tests = pd.concat(anderson_tests, axis=0)
    return anderson_tests

def adf_test(data: Series) -> Dict[str, float]:
    result = adfuller(data, autolag='AIC') 
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    return {'statistic': adf_statistic, 'p-value': p_value, 'critical_value_5%': critical_values['5%']}

def adf_test_dataframe_groups(data: Dict[str, DataFrame]):
    adf_tests = []
    for group, group_data in data.items():
        statistics = group_data.apply(adf_test, axis=0, result_type='expand')
        statistics = statistics.T
        statistics['hypothesis'] = statistics.apply(lambda row: 'Reject' if row['statistic'] < row['critical_value_5%'] else 'Accept', axis=1)
        statistics = statistics.T
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        adf_tests.append(statistics)
    adf_tests = pd.concat(adf_tests, axis=0)
    return adf_tests

def calculate_vif(data, threshold=5):
    data = add_constant(data)
    vif_data = pd.DataFrame()
    vif_data["variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif_data['hypothesis'] = vif_data['VIF'].apply(lambda x: 'Accept' if x <= threshold else 'Reject')
    vif_data = vif_data.set_index('variable')
    return vif_data

def calculate_vif_dataframe_groups(data: Dict[str, DataFrame], threshold: Optional[float]=5):
    vif_scores = []
    for group, group_data in data.items():
        statistics = calculate_vif(group_data, threshold=threshold)
        statistics = statistics.loc[statistics.index != 'const'].T
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        vif_scores.append(statistics)
    vif_scores = pd.concat(vif_scores, axis=0)
    return vif_scores

def calculate_dw(data, dependent_var):
    X = data.drop(dependent_var, axis=1)
    y = data[dependent_var]
    X = sm.add_constant(X)  
    model = sm.OLS(y, X).fit()
    dw_stat = durbin_watson(model.resid)
    if dw_stat < 1.5:
        hypothesis = 'Reject (Positive autocorrelation)'
    elif dw_stat > 2.5:
        hypothesis = 'Reject (Negative autocorrelation)'
    else:
        hypothesis = 'Accept (Little to no autocorrelation)'
    return pd.DataFrame({'DW Statistic': dw_stat, 'Hypothesis': hypothesis}, index=[dependent_var])

def calculate_dw_dataframe_groups(data: Dict[str, DataFrame], dependent_var: str):
    dw_tests = []
    for group, group_data in data.items():
        statistics = calculate_dw(group_data, dependent_var)
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        dw_tests.append(statistics)
    dw_tests = pd.concat(dw_tests, axis=0)
    return dw_tests

def calculate_bp(data, dependent_var):
    X = data.drop(dependent_var, axis=1)
    y = data[dependent_var]
    X = sm.add_constant(X) 
    model = sm.OLS(y, X).fit()
    test_stat, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
    if p_value < 0.05:
        hypothesis = 'Reject (Signs of heteroscedasticity)'
    else:
        hypothesis = 'Accept (No apparent heteroscedasticity)'
    return pd.DataFrame({'BP Statistic': test_stat, 'p-value': p_value, 'Hypothesis': hypothesis}, index=[dependent_var])

def calculate_bp_dataframe_groups(data: Dict[str, DataFrame], dependent_var: str):
    bp_tests = []
    for group, group_data in data.items():
        statistics = calculate_bp(group_data, dependent_var)
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        bp_tests.append(statistics)
    bp_tests = pd.concat(bp_tests, axis=0)
    return bp_tests

def calculate_white_test(data, dependent_var):
    X = data.drop(dependent_var, axis=1)
    y = data[dependent_var]
    X = sm.add_constant(X) 
    model = sm.OLS(y, X).fit()
    test_stat, p_value, _, _ = het_white(model.resid, model.model.exog)
    if p_value < 0.05:
        hypothesis = 'Reject (Signs of heteroscedasticity)'
    else:
        hypothesis = 'Accept (No apparent heteroscedasticity)'
    return pd.DataFrame({'White Statistic': test_stat, 'p-value': p_value, 'Hypothesis': hypothesis}, index=[dependent_var])

def calculate_white_test_dataframe_groups(data: Dict[str, DataFrame], dependent_var: str):
    white_tests = []
    for group, group_data in data.items():
        statistics = calculate_white_test(group_data, dependent_var)
        statistics.index.name = 'statistics'
        statistics['Group'] = group
        statistics = statistics.reset_index().set_index(['Group', 'statistics'])
        white_tests.append(statistics)
    white_tests = pd.concat(white_tests, axis=0)
    return white_tests

def regex_symbol_replacement(match):
    return rf'\{match.group(0)}'

def dataframe_to_latex(dataframe: DataFrame):
    symbols_pattern = r"([\ \_\-\&\%\$\#])"
    table = (dataframe.rename(columns=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x) if isinstance(x, str) else str(round(x, 1)),
                            index=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x) if isinstance(x, str) else str(round(x, 1)))
                .to_latex(multirow=True, multicolumn=True, multicolumn_format='c'))
    table = "\\begin{adjustbox}{width=\\textwidth,center}\n" + f"{table}" + "\end{adjustbox}\n"
    return table
