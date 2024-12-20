from typing import List, Literal, Optional

from pandas import DataFrame
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from mitools.exceptions import ArgumentValueError
from mitools.regressions.base_models import BaseRegressionModel


class MarkovRegressionModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: Optional[str] = None,
        independent_variables: Optional[List[str]] = None,
        k_regimes: int = 2,
        trend: Literal["n", "c", "t", "ct"] = "c",
        switching_variance: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=None,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=None,
            *args,
            **kwargs,
        )

        if self.formula is not None:
            raise ArgumentValueError(
                "MarkovRegression does not support a formula interface."
            )

        if self.dependent_variable is None:
            raise ArgumentValueError(
                "You must provide a dependent_variable for MarkovRegressionModel."
            )
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_variance = switching_variance
        self.model_name = "MarkovRegression"

    def fit(self, *args, **kwargs):
        endog = self.data[self.dependent_variable].values
        exog = None
        if self.independent_variables:
            exog = self.data[self.independent_variables].values

        self.model = MarkovRegression(
            endog,
            k_regimes=self.k_regimes,
            trend=self.trend,
            exog=exog,
            switching_variance=self.switching_variance,
            *self.args,
            **self.kwargs,
        )

        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    def predict(self, start=None, end=None, exog=None, **kwargs):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        return self.results.predict(start=start, end=end, exog=exog, **kwargs)
