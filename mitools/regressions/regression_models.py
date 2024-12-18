from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from pandas import DataFrame

from mitools.exceptions import ArgumentStructureError, ArgumentValueError


class BaseRegressionModel(ABC):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
    ):
        if (dependent_variable is None) and formula is None:
            raise ArgumentValueError(
                "Either dependent_variable and independent_variables, only dependent_variable, or explicit formula must be provided"
            )
        elif (
            dependent_variable is not None
            and independent_variables is not None
            and formula is not None
        ):
            raise ArgumentValueError(
                "Only one of dependent_variable and independent_variables, or explicit formula must be provided"
            )
        self.data = data
        self.formula = formula
        self.dependent_variable = dependent_variable or ""
        if (
            self.formula is None
            and self.dependent_variable is not None
            and independent_variables is None
        ):
            self.independent_variables = [
                c for c in data.columns if c != self.dependent_variable
            ]
        else:
            self.independent_variables = independent_variables or []
        self.independent_variables.sort()
        self.control_variables = control_variables or []
        self.control_variables.sort()
        if self.formula is None:
            self.variables = (
                [self.dependent_variable]
                + self.independent_variables
                + self.control_variables
            )
        else:
            self.variables = None
        self.fitted = False
        self.model_name = None
        self.results = None

    @abstractmethod
    def fit(self):
        pass

    def predict(self, new_data: Optional[DataFrame] = None):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        if hasattr(self.results, "predict"):
            return self.results.predict(new_data)  # Validate for non-formula
        else:
            raise ArgumentStructureError(
                "Results object does not have a predict method"
            )

    def summary(self):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        if hasattr(self.results, "summary"):
            return self.results.summary()
        else:
            raise ArgumentStructureError(
                "Results object does not have a summary method"
            )

    @classmethod
    def from_arrays(
        cls,
        y: np.ndarray,
        X: np.ndarray,
        controls: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        if y.ndim != 1:
            raise ArgumentValueError("y must be 1-dimensional")
        if X.ndim != 2:
            raise ArgumentValueError("x must be 2-dimensional")
        if controls is not None and controls.ndim != 2:
            raise ArgumentValueError("controls must be 2-dimensional")
        n_samples = len(y)
        if len(X) != n_samples:
            raise ArgumentValueError("x and y must have same number of samples")
        if controls is not None and len(controls) != n_samples:
            raise ArgumentValueError("controls must have same number of samples as y")
        independent_vars = [f"x{i+1}" for i in range(X.shape[1])]
        control_vars = []
        if controls is not None:
            control_vars = [f"c{i+1}" for i in range(controls.shape[1])]
        data_dict = {"y": y}
        for i, var in enumerate(independent_vars):
            data_dict[var] = X[:, i]
        if controls is not None:
            for i, var in enumerate(control_vars):
                data_dict[var] = controls[:, i]

        data = DataFrame(data_dict)

        return cls(
            data=data,
            dependent_variable="y",
            independent_variables=independent_vars,
            control_variables=control_vars if controls is not None else None,
            *args,
            **kwargs,
        )


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
            model = sm.OLS(endog, exog, *args, **kwargs)
        self.model_name = "OLS"
        self.results = model.fit()
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
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
        )
        if isinstance(quantiles, float):
            self.quantiles = [quantiles]
        else:
            self.quantiles = sorted(quantiles)
        self.model_name = "QuantReg"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = smf.quantreg(formula=self.formula, data=self.data)
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = sm.QuantReg(endog, exog)
        self.results = {}
        for q in self.quantiles:
            self.results[q] = model.fit(q=q, *args, **kwargs)
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


class PooledOLSModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
        )
        self.model_name = "PooledOLS"

    def fit(
        self, add_constant: bool = True, cov_type: str = "unadjusted", *args, **kwargs
    ):
        if self.formula is not None:
            model = PooledOLS.from_formula(
                formula=self.formula,
                data=self.data,
                *args,
                **kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = PooledOLS(
                dependent=endog,
                exog=exog,
                *args,
                **kwargs,
            )
        self.results = model.fit(cov_type=cov_type)
        self.fitted = True
        return self.results

    def validate_data(self):
        if self.data.index.nlevels != 2:
            raise ArgumentValueError(
                "Data must have two levels in the index, referring to the corresponding entities and time periods, in that order."
            )


class RandomEffectsModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
        )
        self.model_name = "RandomEffects"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = RandomEffects.from_formula(
                formula=self.formula,
                data=self.data,
                *args,
                **kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = RandomEffects(
                dependent=endog,
                exog=exog,
                *args,
                **kwargs,
            )
        self.results = model.fit()
        self.fitted = True
        return self.results

    def validate_data(self):
        if self.data.index.nlevels != 2:
            raise ArgumentValueError(
                "Data must have two levels in the index, referring to the corresponding entities and time periods, in that order."
            )


class BetweenOLSModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
        )
        self.model_name = "BetweenOLS"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = BetweenOLS.from_formula(
                formula=self.formula,
                data=self.data,
                *args,
                **kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = BetweenOLS(
                dependent=endog,
                exog=exog,
                *args,
                **kwargs,
            )
        self.results = model.fit()
        self.fitted = True
        return self.results

    def validate_data(self):
        if self.data.index.nlevels != 2:
            raise ArgumentValueError(
                "Data must have two levels in the index, referring to the corresponding entities and time periods, in that order."
            )


class PanelOLSModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
        entity_effects: bool = False,
        time_effects: bool = False,
    ):
        super().__init__(
            data=data,
            formula=formula,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=control_variables,
        )
        self.model_name = "PanelOLS"
        self.entity_effects = entity_effects
        self.time_effects = time_effects

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            model = PanelOLS.from_formula(
                formula=self.formula,
                data=self.data,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                *args,
                **kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            model = PanelOLS(
                dependent=endog,
                exog=exog,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                *args,
                **kwargs,
            )
        self.results = model.fit()
        self.fitted = True
        return self.results

    def validate_data(self):
        if self.data.index.nlevels != 2:
            raise ArgumentValueError(
                "Data must have two levels in the index, referring to the corresponding entities and time periods, in that order."
            )
