from typing import List, Optional, Union

from linearmodels.asset_pricing import TradedFactorModel as TFModel
from pandas import DataFrame

from mitools.exceptions import ArgumentValueError
from mitools.regressions.linear_models import BaseRegressionModel


class TradedFactorModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        portfolios: Optional[Union[str, List[str]]] = None,
        factors: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=None,
            dependent_variable=portfolios,
            independent_variables=factors,
            control_variables=None,
            *args,
            **kwargs,
        )
        self.model_name = "TradedFactorModel"

    def fit(self, *args, **kwargs):
        if isinstance(self.dependent_variable, str):
            portfolios = self.data[[self.dependent_variable]]
        else:
            portfolios = self.data[self.dependent_variable]
        factors = self.data[self.independent_variables]
        model = TFModel(portfolios=portfolios, factors=factors)
        self.results = model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    def predict(self, new_data: Optional[DataFrame] = None):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        if new_data is None:
            new_data = self.data[self.independent_variables]
        else:
            if not all(var in new_data.columns for var in self.independent_variables):
                missing = [
                    var
                    for var in self.independent_variables
                    if var not in new_data.columns
                ]
                raise ArgumentValueError(
                    f"new_data is missing required factor columns: {missing}"
                )
            new_data = new_data[self.independent_variables]
        return self.results.predict(new_data)
