from typing import List, Optional, Union

import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas import DataFrame
from statsmodels.regression.rolling import RollingOLS
from statsmodels.robust.norms import HuberT

from mitools.exceptions import ArgumentValueError
from mitools.regressions.base_models import BaseRegressionModel


class OLSModel(BaseRegressionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "OLS"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = smf.ols(formula=self.formula, data=self.data, *args, **kwargs)
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = sm.OLS(endog, exog, *self.args, **self.kwargs)
        self.model_name = "OLS"
        self.results = model.fit(*args, **kwargs)
        self.fitted = True
        return self.results


class QuantileRegressionModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
        quantiles: Union[float, List[float]] = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
            *args,
            **kwargs,
        )
        if isinstance(quantiles, float):
            self.quantiles = [quantiles]
        else:
            self.quantiles = sorted(quantiles)
        self.model_name = "QuantReg"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = smf.quantreg(
                formula=self.formula, data=self.data, *self.args, **self.kwargs
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            self.model = sm.QuantReg(endog, exog, *self.args, **self.kwargs)
        self.results = {}
        for q in self.quantiles:
            self.results[q] = self.model.fit(q=q, *args, **kwargs)
        self.fitted = True
        return (
            self.results if len(self.quantiles) > 1 else self.results[self.quantiles[0]]
        )

    def predict(
        self,
        new_data: Optional[DataFrame] = None,
        quantiles: Optional[Union[float, List[float]]] = None,
    ):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        if quantiles is not None:
            quantiles = [quantiles] if isinstance(quantiles, float) else quantiles
            quantiles = sorted(quantiles)
            if any(quantile not in self.quantiles for quantile in quantiles):
                raise ArgumentValueError(
                    f"Quantiles {[quantile for quantile in quantiles if quantile not in self.quantiles]} not in 'self.quantiles'={self.quantiles}"
                )
            predictions = {q: self.results[q].predict(new_data) for q in self.quantiles}
            return predictions if len(self.quantiles) > 1 else predictions[quantiles[0]]
        else:
            return {q: self.results[q].predict(new_data) for q in self.quantiles}

    def summary(self, quantiles: Optional[Union[float, List[float]]] = None):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        if quantiles is not None:
            quantiles = [quantiles] if isinstance(quantiles, float) else quantiles
            quantiles = sorted(quantiles)
            if any(quantile not in self.quantiles for quantile in quantiles):
                raise ArgumentValueError(
                    f"Quantiles {[quantile for quantile in quantiles if quantile not in self.quantiles]} not in 'self.quantiles'={self.quantiles}"
                )
            summaries = {q: self.results[q].summary() for q in quantiles}
            return summaries if len(quantiles) > 1 else summaries[quantiles[0]]
        else:
            summaries = {q: self.results[q].summary() for q in self.quantiles}
            print(len(self.quantiles))
            return (
                summaries if len(self.quantiles) > 1 else summaries[self.quantiles[0]]
            )


class RLMModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
        M: Optional[sm.robust.norms.RobustNorm] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
            *args,
            **kwargs,
        )
        self.model_name = "RLM"
        self.M = M or HuberT()

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = smf.rlm(
                formula=self.formula,
                data=self.data,
                M=self.M,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = sm.RLM(endog, exog, M=self.M, *self.args, **self.kwargs)
        self.results = model.fit(*args, **kwargs)
        self.fitted = True
        return self.results


class RollingOLSModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
        window: int = 30,
        min_nobs: Optional[int] = None,
        expanding: bool = False,
        missing: str = "drop",
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=None,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
            *args,
            **kwargs,
        )
        if self.dependent_variable is None:
            raise ArgumentValueError(
                "You must provide a dependent_variable for RollingOLSModel."
            )
        self.model_name = "RollingOLS"
        self.window = window
        self.min_nobs = min_nobs
        self.expanding = expanding
        self.missing = missing

    def fit(self, add_constant: bool = True, *args, **kwargs):
        endog = self.data[self.dependent_variable]
        exog_vars = self.independent_variables + self.control_variables
        exog = self.data[exog_vars]
        if add_constant:
            exog = sm.add_constant(exog, has_constant="add")
        model = RollingOLS(
            endog,
            exog,
            window=self.window,
            min_nobs=self.min_nobs,
            expanding=self.expanding,
            missing=self.missing,
            *self.args,
            **self.kwargs,
        )
        self.results = model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    def predict(self, new_data: Optional[DataFrame] = None):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet.")
        if new_data is None:
            return self.results.predict()
        else:
            missing_vars = [
                col
                for col in self.independent_variables + self.control_variables
                if col not in new_data.columns
            ]
            if missing_vars:
                raise ArgumentValueError(
                    f"new_data is missing required variables: {missing_vars}"
                )
            exog = new_data[self.independent_variables + self.control_variables].copy()
            if (
                "const" in self.results.model.exog.columns
                and "const" not in exog.columns
            ):
                exog = sm.add_constant(exog, has_constant="add")
            return self.results.predict(exog)
