from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
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
        *args,
        **kwargs,
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
        self.args = list(args)
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, *args, **kwargs):
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
        self.model_name = None

    def validate_data(self, data):
        if data.index.nlevels != 2:
            raise ArgumentValueError(
                "Data must have two levels in the index, referring to the corresponding entities and time periods, in that order."
            )
