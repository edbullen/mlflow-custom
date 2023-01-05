import mlflow
import pandas as pd


class HybridFunction(mlflow.pyfunc.PythonModel):
    """ This Class wraps a custom function in MLflow, so that it can be parameterised, tracked and deployed with MLflow

    Attributes:
        x0: boundary on x-axis between function A and function B
        y0: y-axis value for function A output and base-value for function B
        gradient: slope / gradient for increase in y given x (result) for function B

    """

    def __init__(self, x0, y0, gradient):
        self.x0 = x0
        self.y0 = y0
        self.gradient = gradient

    def _custom_function(self, model_input :pd.DataFrame):
        """ Hybrid function combines two different functions depending on the value of x
            x < x0 -> y is a constant value y0
            x >= x0 -> y is uniform gradient function,  y = gradient*x + c (c is the intercept)

        Args:
           model_input: :pandas.DataFrame:

        Returns:
            model_output: :pandas.DataFrame:
        """

        y_intercept = self.y0 - self.gradient * self.x0

        model_output = model_input.apply(lambda x: self.y0 if x < self.x0 else self.gradient*x + y_intercept)

        return model_output

    def predict(self, context, model_input :pd.DataFrame):
        """ MLflow entry-point to apply the function to a data payload
        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference. https://mlflow.org/docs/latest/_modules/mlflow/pyfunc/model.html
            model_input: :pandas.Dataframe: series of x-values to apply the _custom_function() to

        Returns:
            model_output: :pandas.Dataframe:
        """

        return model_input.apply(self._custom_function)
