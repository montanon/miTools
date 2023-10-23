from dataclasses import dataclass
from abc import ABC
from typing import Union, Dict, Tuple, List, Any
from ..utils import replace_prefix


@dataclass(froze=True)
class OLSResults:

    model_stats: Dict

    n_obs: int
    F_stats: Dict[Tuple[int, int],float]
    Prob_F: int
    R_sq: float
    Adj_R_sq: float
    Root_MSE: float

    indep_variable: str
    dep_variables: List[str]

    coefficients: Dict[str,float]
    std_errs: Dict[str,float]
    t_values: Dict[str,float]
    p_values: Dict[str,float]
    significances: Dict[str,str]
    conf_interval: Dict[str,Tuple[float,float]]
    model_params: Dict[str,Any]
    model_specification: str

    def __post__init__(self):
        assert all(v in self.coefficients.keys() for v in self.dep_variables)
        assert all(v in self.std_errs.keys() for v in self.dep_variables)
        assert all(v in self.t_values.keys() for v in self.dep_variables)
        assert all(v in self.p_values.keys() for v in self.dep_variables)
        assert all(v in self.significances.keys() for v in self.dep_variables)
        assert all(v in self.conf_interval.keys() for v in self.dep_variables)


@dataclass(frozen=True)
class CSARDLResults:

    model_stats: Dict

    n_obs: int
    n_groups: int
    n_obs_per_group: int
    F_stats: Dict[Tuple[int, int],float]
    Prob_F: int
    R_sq: float
    R_sq_MG: float
    Root_MSE: float
    CD_stat: float
    p_value: float

    indep_variable: str
    dep_variables: List[str]
    
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

    def __post__init__(self):
        assert all(replace_prefix(v, 'L.', '') in self.short_run_coefficients.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_std_errs.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_z_values.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_p_values.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_significances.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.short_run_conf_intervals.keys() for v in self.dep_variables)

        assert all(replace_prefix(v, 'lr_', '') in self.long_run_coefficients.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_std_errs.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_z_values.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_p_values.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.long_run_significances.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.long_run_conf_intervals.keys() for v in self.dep_variables)

        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_coefficients.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_std_errs.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_z_values.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_p_values.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'L.', '') in self.adj_term_significances.keys() for v in self.dep_variables)
        assert all(replace_prefix(v, 'lr_', '') in self.adj_term_conf_intervals.keys() for v in self.dep_variables)

