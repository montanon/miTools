from typing import List, Optional

import statsmodels.api as sm
from linearmodels import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from pandas import DataFrame

from mitools.regressions.base_models import BasePanelRegressionModel


class PanelOLSModel(BasePanelRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
        entity_effects: bool = False,
        time_effects: bool = False,
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
        self.model_name = "PanelOLS"
        self.entity_effects = entity_effects
        self.time_effects = time_effects

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            self.model = PanelOLS.from_formula(
                formula=self.formula,
                data=self.data,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            self.model = PanelOLS(
                dependent=endog,
                exog=exog,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                *self.args,
                **self.kwargs,
            )
        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results


class PooledOLSModel(BasePanelRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
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
        self.model_name = "PooledOLS"

    def fit(
        self,
        add_constant: bool = True,
        *args,
        **kwargs,
    ):
        if self.formula is not None:
            self.model = PooledOLS.from_formula(
                formula=self.formula,
                data=self.data,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            self.model = PooledOLS(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = self.model.fit(
            *args,
            **kwargs,
        )
        self.fitted = True
        return self.results


class RandomEffectsModel(BasePanelRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
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
        self.model_name = "RandomEffects"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            self.model = RandomEffects.from_formula(
                formula=self.formula,
                data=self.data,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            self.model = RandomEffects(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results


class BetweenOLSModel(BasePanelRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
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
        self.model_name = "BetweenOLS"

    def fit(self, add_constant: bool = True, *args, **kwargs):
        if self.formula is not None:
            self.model = BetweenOLS.from_formula(
                formula=self.formula,
                data=self.data,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            if add_constant:
                exog = sm.add_constant(exog)
            self.model = BetweenOLS(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results


class FirstDifferenceOLSModel(BasePanelRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        formula: Optional[str] = None,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        control_variables: Optional[List[str]] = None,
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
        self.model_name = "FirstDifferenceOLS"

    def fit(self, *args, **kwargs):
        if self.formula is not None:
            self.model = FirstDifferenceOLS.from_formula(
                formula=self.formula,
                data=self.data,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            self.model = FirstDifferenceOLS(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results
