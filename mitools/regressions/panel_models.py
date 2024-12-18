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

from mitools.exceptions import ArgumentValueError
from mitools.regressions.linear_models import BaseRegressionModel


class BasePanelRegressionModel(BaseRegressionModel):
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
        self.validate_data(data)
        super().__init__(
            data,
            formula,
            dependent_variable,
            independent_variables,
            control_variables,
            *args,
            **kwargs,
        )
        self.model_name = "PanelOLS"

    def validate_data(self, data):
        if data.index.nlevels != 2:
            raise ArgumentValueError(
                "Data must have two levels in the index, referring to the corresponding entities and time periods, in that order."
            )


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
            model = PanelOLS.from_formula(
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
            model = PanelOLS(
                dependent=endog,
                exog=exog,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                *self.args,
                **self.kwargs,
            )
        self.results = model.fit(*args, **kwargs)
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
            model = PooledOLS.from_formula(
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
            model = PooledOLS(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = model.fit(
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
            model = RandomEffects.from_formula(
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
            model = RandomEffects(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = model.fit(*args, **kwargs)
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
            model = BetweenOLS.from_formula(
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
            model = BetweenOLS(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = model.fit(*args, **kwargs)
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
            model = FirstDifferenceOLS.from_formula(
                formula=self.formula,
                data=self.data,
                *self.args,
                **self.kwargs,
            )
        else:
            endog = self.data[self.dependent_variable]
            exog_vars = self.independent_variables + self.control_variables
            exog = self.data[exog_vars]
            model = FirstDifferenceOLS(
                dependent=endog,
                exog=exog,
                *self.args,
                **self.kwargs,
            )
        self.results = model.fit(*args, **kwargs)
        self.fitted = True
        return self.results
