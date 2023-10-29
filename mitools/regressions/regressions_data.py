from dataclasses import dataclass
from abc import ABC
from typing import Union, Dict, Tuple, List, Any
from ..utils import replace_prefix, add_significance
from pandas import DataFrame, MultiIndex
from abc import ABC, abstractmethod


class RegressionData(ABC):
    pass

def format_result(row):
    return f"{row.round(2)} ({row.astype(str)})"

@dataclass(frozen=True)
class XTRegResults(RegressionData):

    n_obs: int
    n_groups: int
    n_obs_per_group: int
    F_stats: Dict[Tuple[int, int],float]
    Prob_F: int
    R_sq: float
    R_sq_within: float
    R_sq_between: float
    corr: float

    dep_variable: str
    indep_variables: List[str]

    coefficients: Dict[str,float]
    std_errs: Dict[str,float]
    t_values: Dict[str,float]
    p_values: Dict[str,float]
    significances: Dict[str,str]
    conf_interval: Dict[str,Tuple[float,float]]

    model_params: Dict[str,Any]
    model_specification: str

    coefficient_col: str = 'Coefficient'
    significance_col: str = 'P>|t|'

    def __post__init__(self):
        assert all(v in self.coefficients.keys() for v in self.indep_variables)
        assert all(v in self.std_errs.keys() for v in self.indep_variables)
        assert all(v in self.t_values.keys() for v in self.indep_variables)
        assert all(v in self.p_values.keys() for v in self.indep_variables)
        assert all(v in self.significances.keys() for v in self.indep_variables)
        assert all(v in self.conf_interval.keys() for v in self.indep_variables)

    def to_df(self):
        
        base_data = {
        'n_obs':self.n_obs,
        'n_groups': self.n_groups,
        'n_obs_per_group': self.n_obs_per_group,
        'F_stats': self.F_stats,
        'Prob_F': self.Prob_F,
        'R_sq': self.R_sq,
        'R_sq_within': self.R_sq_within,
        'R_sq_between': self.R_sq_between,
        'model_specification': self.model_specification
        }

        rows = []
        for var, stats in self.coefficients.items():
            row = {'Variable': var}
            row.update(stats)
            rows.append(row)
        df = DataFrame(rows)
        for key, value in base_data.items():
            df[key] = value
        relevant_cols = ['Variable', self.coefficient_col, self.significance_col]
        try:
            df = df[relevant_cols]
        except Exception:
            df[relevant_cols] = ''
        df['Dep Var'] = self.dep_variable
        df = df.set_index('Dep Var')
        df = df.set_index('Variable', append=True)
        if self.indep_variables == 0:
            self.indep_variables = [str(self.indep_variables)] # YIKES
        df['Indep Var'] = [v for v in self.indep_variables if v.find('.') == -1][0]
        df = df.set_index('Indep Var', append=True)
        return df
    
    def to_pretty_df(self):
        regression_data = self.to_df()
        try:
            regression_data['Result'] = (regression_data[self.coefficient_col]
                                        .round(2)
                                        .astype(str) + ' (' + regression_data[self.significance_col]
                                        .astype(str) + ')')
            regression_data['Result'] = regression_data['Result'].apply(add_significance)
        except Exception:
            regression_data['Result'] = ''
        regression_data = regression_data[['Result']]
        return regression_data


@dataclass(frozen=False)
class OLSResults(RegressionData):

    n_obs: int
    F_stats: Dict[Tuple[int, int],float]
    Prob_F: int
    R_sq: float
    AdjR_sq: float
    Root_MSE: float

    dep_variable: str
    indep_variables: List[str]

    coefficients: Dict[str,float]
    std_errs: Dict[str,float]
    t_values: Dict[str,float]
    p_values: Dict[str,float]
    significances: Dict[str,str]
    conf_interval: Dict[str,Tuple[float,float]]
    model_params: Dict[str,Any]
    model_specification: str

    coefficient_col: str = 'Coefficient'
    significance_col: str = 'P>|t|'

    def __post__init__(self):
        assert all(v in self.coefficients.keys() for v in self.indep_variables)
        assert all(v in self.std_errs.keys() for v in self.indep_variables)
        assert all(v in self.t_values.keys() for v in self.indep_variables)
        assert all(v in self.p_values.keys() for v in self.indep_variables)
        assert all(v in self.significances.keys() for v in self.indep_variables)
        assert all(v in self.conf_interval.keys() for v in self.indep_variables)

    def to_df(self):

        base_data = {
        'n_obs':self.n_obs,
        'F_stats': self.F_stats,
        'Prob_F': self.Prob_F,
        'R_sq': self.R_sq,
        'AdjR_sq': self.AdjR_sq,
        'Root_MSE': self.Root_MSE,
        'model_specification': self.model_specification
        }
        rows = []
        for var, stats in self.coefficients.items():
            row = {'Variable': var}
            row.update(stats)
            rows.append(row)
        df = DataFrame(rows)
        for key, value in base_data.items():
            df[key] = value
        relevant_cols = ['Variable', 'Coefficient', 'P>|t|']
        df = df[relevant_cols]
        df['Dep Var'] = self.dep_variable
        df = df.set_index('Dep Var')
        df = df.set_index('Variable', append=True)
        if self.indep_variables == 0:
            self.indep_variables = [str(self.indep_variables)] # YIKES
        df['Indep Var'] = [v for v in self.indep_variables if v.find('.') == -1][0]
        df = df.set_index('Indep Var', append=True)
        return df
    
    def to_pretty_df(self):
        regression_data = self.to_df()
        regression_data['Result'] = (regression_data[self.coefficient_col]
                                    .round(2)
                                    .astype(str) + ' (' + regression_data[self.significance_col]
                                    .astype(str) + ')')
        regression_data['Result'] = regression_data['Result'].apply(add_significance)
        regression_data = regression_data[['Result']]
        return regression_data


@dataclass(frozen=False)
class CSARDLResults(RegressionData):

    model_params: Dict

    n_obs: int
    n_groups: int
    n_obs_per_group: int
    F_stats: Dict[Tuple[int, int],float]
    Prob_F: int
    R_sq: float
    R_sq_MG: float
    Root_MSE: float
    CD_stats: float
    p_value: float

    dep_variable: str
    indep_variables: List[str]
    
    short_run_coefficients: Dict[str,float]
    short_run_std_errs: Dict[str,float]
    short_run_z_values: Dict[str,float]
    short_run_p_values: Dict[str,float]
    short_run_significances: Dict[str,str]
    short_run_conf_intervals: Dict[str,Tuple[float,float]]

    long_run_coefficients: Dict[str,float]
    long_run_std_errs: Dict[str,float]
    long_run_z_values: Dict[str,float]
    long_run_p_values: Dict[str,float]
    long_run_significances: Dict[str,str]
    long_run_conf_intervals: Dict[str,Tuple[float,float]]

    adj_term_coefficients: Dict[str,float]
    adj_term_std_errs: Dict[str,float]
    adj_term_z_values: Dict[str,float]
    adj_term_p_values: Dict[str,float]
    adj_term_significances: Dict[str,str]
    adj_term_conf_intervals: Dict[str,Tuple[float,float]]

    model_specification: str

    coefficient_col: str = 'Coef.'
    significance_col: str = 'P>|z|'

    def __post__init__(self):
        assert all(replace_prefix(v, 'L.', '') in self.short_run_coefficients.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_std_errs.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_z_values.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_p_values.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_significances.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_conf_intervals.keys() for v in self.indep_variables)

        assert all(replace_prefix(v, 'lr_', '') in self.long_run_coefficients.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_std_errs.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_z_values.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_p_values.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.long_run_significances.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_conf_intervals.keys() for v in self.indep_variables)

        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_coefficients.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_std_errs.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_z_values.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_p_values.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.adj_term_significances.keys() for v in self.indep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_conf_intervals.keys() for v in self.indep_variables)

    def to_df(self):
        base_data = {
        'n_obs':self.n_obs,
        'n_groups': self.n_groups,
        'n_obs_per_group': self.n_obs_per_group,
        'F_stats': self.F_stats,
        'Prob_F': self.Prob_F,
        'R_sq': self.R_sq,
        'R_sq_MG': self.R_sq_MG,
        'Root_MSE': self.Root_MSE,
        'CD_stats': self.CD_stats,
        'p_value': self.p_value,
        'lag': self.model_params.get('lag', None),
        'model_specification': self.model_specification
        }
        rows = []
        for var, stats in self.short_run_coefficients.items():
            row = {'Variable': var, 'type': 'short_run'}
            row.update(stats)
            rows.append(row)
        for var, stats in self.adj_term_coefficients.items():
            row = {'Variable': var, 'type': 'adj_term'}
            row.update(stats)
            rows.append(row)
        for var, stats in self.long_run_coefficients.items():
            row = {'Variable': var, 'type': 'long_run'}
            row.update(stats)
            rows.append(row)
        df = DataFrame(rows)
        for key, value in base_data.items():
            df[key] = value
            
        df.index = MultiIndex.from_product([list([self.dep_variable]), [self.model_params.get('lag', None)], df['type'].values])
        relevant_cols = ['Variable', 'Coef.', 'P>|z|']
        
        try:
            df = df[relevant_cols]
        except Exception:
            df['Variable'] = ''
            df['Coef.'] = 0.0
            df['P>|z|'] = 1.0

        df.index.names = ['Dep Var', 'Lag', 'Time Span']
        df = df.set_index('Variable', append=True)
        if self.indep_variables != 0:
            df['Indep Var'] = [v for v in self.indep_variables if v.find('.') == -1][0]
        else:
            df['Indep Var'] = 'None' # YIKES
        df = df.set_index('Indep Var', append=True)
        return df

    def to_pretty_df(self):
        regression_data = self.to_df()
        regression_data['Result'] = (regression_data[self.coefficient_col]
                                    .round(2)
                                    .astype(str) + ' (' + regression_data[self.significance_col]
                                    .astype(str) + ')')
        regression_data['Result'] = regression_data['Result'].apply(add_significance)
        regression_data = regression_data[['Result']]
        return regression_data
