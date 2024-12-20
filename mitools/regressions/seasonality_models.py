from typing import Optional, Sequence, Union

import numpy as np
from pandas import DataFrame
from statsmodels.tsa.seasonal import MSTL

from mitools.exceptions import ArgumentValueError


class MSTLModel:
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: Optional[str] = None,
        periods: Optional[Union[int, Sequence[int]]] = None,
        windows: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ):
        self.data = data
        self.dependent_variable = dependent_variable
        self.periods = periods
        self.windows = windows
        self.kwargs = kwargs
        self.model_name = "MSTL"
        self.fitted = False

    def fit(self, *args, **kwargs):
        y = self.data[self.dependent_variable].values
        self.model = MSTL(y, periods=self.periods, **self.kwargs)
        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    def predict(self, steps: int = 0):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        trend = self.results.trend
        seasonal = self.results.seasonal
        if seasonal.ndim == 1:
            seasonal_sum = seasonal
        else:
            seasonal_sum = np.sum(seasonal, axis=1)
        fitted_values = trend + seasonal_sum
        return fitted_values

    def summary(self):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        comp_info = (
            f"MSTL Decomposition\n"
            f"------------------\n"
            f"Observations: {len(self.results.trend)}\n"
            f"Number of seasonal components: {self.results.seasonal.shape[1] if self.results.seasonal.ndim > 1 else 1}\n"
            f"Trend shape: {self.results.trend.shape}\n"
            f"Seasonal shape: {self.results.seasonal.shape}\n"
            f"Remainder shape: {self.results.resid.shape}\n"
        )
        return comp_info
