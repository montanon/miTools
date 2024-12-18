from abc import ABC, abstractmethod
from typing import List, Optional

import statsmodels.api as sm
import statsmodels.formula.api as smf
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
        if (
            dependent_variable is None or independent_variables is None
        ) and formula is None:
            raise ArgumentValueError(
                "Either dependent_variable and independent_variables, or explicit formula must be provided"
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
        self.independent_variables = independent_variables or []
        self.independent_variables.sort()
        self.control_variables = control_variables or []
        self.control_variables.sort()
        self.dependent_variable = dependent_variable or ""
        if self.formula is None:
            self.variables = (
                [self.dependent_variable]
                + self.independent_variables
                + self.control_variables
            )
        else:
            self.variables = None
        self.fitted = False
        self.model = None
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
