# Example for Packaging Python code as a Custom Model in MLflow

MLflow is typically used for deploying machine learning models developed using frameworks such as Scikit-learn or Tensorflow.  
  
MLflow can also be used to package up custom functions written in Python and deployed in the same MLflow framework.
  
This makes it convenient to deploy and track parameterised versions of custom functions in MLflow on a Databricks platform and execute these via the model.predict() interface against data in a Databricks cluster.  

The approach demonstrated here is to use the `mlflow.pyfunc` `PythonModel` class to wrap a custom Python class called `HybridFunction` with a method `_custom_function()` containing the custom logic and accessed via the MLflow `predict()` method.


## Example Hybrid Function

A hypothetical model-function is used to illustrate the use-case.

a Hybrid Function or [Piecewise Function](https://en.wikipedia.org/wiki/Piecewise) is implemented that combines two different functions depending on the value of x (the domain).  

+ When x < x0, y is a constant value y0
+ When x >= x0, y is uniform gradient function, y = gradient*x + c  (c is the intercept)

![hybrid function](./doc/HybridFunction.png "Hybrid Function")


This simple concept can be extended to encompass complex business rules or "models" that are not catered for by the usual machine learning frameworks.  
  
The intention is to show how bespoke business logic can be coded independently in Python and then packaged and deployed for use against business data in the MLflow framwork.
## Code


## Databricks Notebooks

