from typing import Dict, List, Literal, Optional

import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import anderson, shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from mitools.exceptions import ArgumentValueError
from mitools.regressions import OLSModel


def shapiro_test(data: Series) -> Dict[float, float]:
    stat, p = shapiro(data)
    return {"statistic": stat, "p-value": p}


def shapiro_tests(data: DataFrame) -> DataFrame:
    shapiro_tests = []
    for column in data.columns:
        test_result = shapiro_test(data[column])
        test_result["variable"] = column
        shapiro_tests.append(test_result)
    return DataFrame(shapiro_tests).set_index("variable")


def anderson_test(
    data: Series,
    criteria: Optional[float] = 0.01,
    dist: Optional[
        Literal[
            "norm", "expon", "logistic", "gumbel", "gumbel_l", "gumbel_r", "extreme1"
        ]
    ] = "dist",
) -> Dict[str, float]:
    normal_critical_values = [0.15, 0.1, 0.05, 0.025, 0.01]
    result = anderson(data, dist=dist)
    return {
        "statistic": result.statistic,
        "critical_value": result.critical_values[
            normal_critical_values.index(criteria)
        ],
    }


def anderson_tests(
    data: DataFrame, criteria: float = 0.01, dist: str = "norm"
) -> DataFrame:
    anderson_tests = []
    for column in data.columns:
        test_result = anderson_test(data[column], criteria=criteria, dist=dist)
        test_result["variable"] = column
        anderson_tests.append(test_result)
    return DataFrame(anderson_tests).set_index("variable")


def adf_test(
    data: Series,
    critical_value: Optional[Literal[1, 5, 10]] = 5,
    regression: Optional[Literal["c", "ct", "ctt", "nc"]] = "c",
) -> Dict[str, float]:
    result = adfuller(data, autolag="AIC", regression=regression)
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    return {
        "statistic": adf_statistic,
        "p-value": p_value,
        f"critical_value_{critical_value}%": critical_values[f"{critical_value}%"],
    }


def adf_tests(
    data: DataFrame,
    critical_value: Literal[1, 5, 10] = 5,
    regression: Literal["c", "ct", "ctt", "nc"] = "c",
) -> DataFrame:
    adf_tests = []
    for column in data.columns:
        test_result = adf_test(
            data[column], critical_value=critical_value, regression=regression
        )
        test_result["variable"] = column
        adf_tests.append(test_result)
    return DataFrame(adf_tests).set_index("variable")


def calculate_vif(
    data: DataFrame,
    dependent_variable: str,
    independent_variables: Optional[List[str]] = None,
    threshold: float = 5,
) -> DataFrame:
    if independent_variables is not None:
        included_variables = independent_variables
    else:
        included_variables = [c for c in data.columns if c != dependent_variable]
    data = data[included_variables].copy(deep=True)
    data = add_constant(data)
    vif_data = DataFrame()
    vif_data["variable"] = data.columns
    vif_data["VIF"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    vif_data["hypothesis"] = vif_data["VIF"].apply(
        lambda x: "Accept" if x <= threshold else "Reject"
    )
    vif_data = vif_data.set_index("variable")
    return vif_data


def durbin_watson_test(
    data: DataFrame,
    dependent_variable: str,
    independent_variables: Optional[List[str]] = None,
) -> DataFrame:
    if independent_variables is not None:
        included_variables = independent_variables
    else:
        included_variables = [c for c in data.columns if c != dependent_variable]
    model = OLSModel(
        data=data,
        dependent_variable=dependent_variable,
        independent_variables=included_variables,
    )
    results = model.fit()
    dw_stat = durbin_watson(results.resid)
    if dw_stat < 1.5:
        hypothesis = "Reject (Positive autocorrelation)"
    elif dw_stat > 2.5:
        hypothesis = "Reject (Negative autocorrelation)"
    else:
        hypothesis = "Accept (Little to no autocorrelation)"
    return DataFrame(
        {"DW Statistic": dw_stat, "Hypothesis": hypothesis}, index=[dependent_variable]
    )


def breusch_pagan_test(
    data: DataFrame,
    dependent_variable: str,
    independent_variables: Optional[List[str]] = None,
) -> DataFrame:
    if independent_variables is not None:
        included_variables = independent_variables
    else:
        included_variables = [c for c in data.columns if c != dependent_variable]
    model = OLSModel(
        data=data,
        dependent_variable=dependent_variable,
        independent_variables=included_variables,
    )
    results = model.fit()
    test_stat, p_value, _, _ = het_breuschpagan(results.resid, results.model.exog)
    if p_value < 0.01:
        hypothesis = "Reject (Signs of heteroscedasticity)"
    else:
        hypothesis = "Accept (No apparent heteroscedasticity)"
    return DataFrame(
        {"BP Statistic": test_stat, "p-value": p_value, "Hypothesis": hypothesis},
        index=[dependent_variable],
    )


def white_test(
    data: DataFrame,
    dependent_variable: str,
    independent_variables: Optional[List[str]] = None,
) -> DataFrame:
    if independent_variables is not None:
        included_variables = independent_variables
    else:
        included_variables = [c for c in data.columns if c != dependent_variable]
    model = OLSModel(
        data=data,
        dependent_variable=dependent_variable,
        independent_variables=included_variables,
    )
    results = model.fit()
    test_stat, p_value, _, _ = het_white(results.resid, results.model.exog)
    if p_value < 0.05:
        hypothesis = "Reject (Signs of heteroscedasticity)"
    else:
        hypothesis = "Accept (No apparent heteroscedasticity)"
    return pd.DataFrame(
        {"White Statistic": test_stat, "p-value": p_value, "Hypothesis": hypothesis},
        index=[dependent_variable],
    )


class StatisticalTests:
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
    ):
        self.data = data
        self.dependent_variable = dependent_variable
        if independent_variables is None:
            self.independent_variables = [
                c for c in data.columns if c != dependent_variable
            ]
        else:
            self.independent_variables = independent_variables
        self.control_variables = control_variables or []

        if dependent_variable and dependent_variable not in data.columns:
            raise ArgumentValueError(
                f"Dependent variable '{dependent_variable}' not found in data."
            )
        missing_ivars = [v for v in self.independent_variables if v not in data.columns]
        if missing_ivars:
            raise ArgumentValueError(
                f"Independent variables {missing_ivars} not found in data."
            )
        missing_cvars = [v for v in self.control_variables if v not in data.columns]
        if missing_cvars:
            raise ArgumentValueError(
                f"Control variables {missing_cvars} not found in data."
            )

    def shapiro_test(self) -> DataFrame:
        return shapiro_tests(data=self.data)

    def anderson_test(self, criteria: float = 0.01, dist: str = "norm") -> DataFrame:
        return anderson_tests(data=self.data, criteria=criteria, dist=dist)

    def adf_test(
        self,
        critical_value: Literal[1, 5, 10] = 5,
        regression: Literal["c", "ct", "ctt", "nc"] = "c",
    ) -> DataFrame:
        return adf_tests(
            data=self.data, critical_value=critical_value, regression=regression
        )

    def calculate_vif(self, threshold: float = 5.0) -> DataFrame:
        return calculate_vif(
            data=self.data,
            dependent_variable=self.dependent_variable,
            independent_variables=self.independent_variables,
            threshold=threshold,
        )

    def durbin_watson_test(self) -> DataFrame:
        return durbin_watson_test(
            data=self.data,
            dependent_variable=self.dependent_variable,
            independent_variables=self.independent_variables,
        )

    def breusch_pagan_test(self) -> DataFrame:
        return breusch_pagan_test(
            data=self.data,
            dependent_variable=self.dependent_variable,
            independent_variables=self.independent_variables,
        )

    def white_test(self) -> DataFrame:
        return white_test(
            data=self.data,
            dependent_variable=self.dependent_variable,
            independent_variables=self.independent_variables,
        )

    def __repr__(self) -> str:
        variables = []
        if self.dependent_variable:
            variables.append(f"dependent_variable='{self.dependent_variable}'")
        if self.independent_variables:
            variables.append(f"independent_variables={self.independent_variables}")
        if self.control_variables:
            variables.append(f"control_variables={self.control_variables}")

        return f"StatisticalTests(data.shape={self.data.shape}, {', '.join(variables)})"

    def __str__(self) -> str:
        parts = ["Statistical Tests Configuration:"]
        parts.append(f"Data Shape: {self.data.shape}")

        if self.dependent_variable:
            parts.append(f"Dependent Variable: {self.dependent_variable}")

        if self.independent_variables:
            parts.append("Independent Variables:")
            parts.extend(f"  - {var}" for var in self.independent_variables)

        if self.control_variables:
            parts.append("Control Variables:")
            parts.extend(f"  - {var}" for var in self.control_variables)

        return "\n".join(parts)
