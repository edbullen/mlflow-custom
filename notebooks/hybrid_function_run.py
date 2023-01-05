# Databricks notebook source
# MAGIC %md
# MAGIC # Run Custom Function using MLflow
# MAGIC Notebook to call an instance of the `HybridFunction` class that has been deployed in MLflow.  
# MAGIC Multiple instances of the Hybrid Function can be stored in MLflow with different function parameters.  
# MAGIC MLflow can be used to choose a particular function configuration and execute it against a data-pipeline.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load the model

# COMMAND ----------


import mlflow

# Instead of specifying the model version or run-id, make this code generic to pick up whatever model has been promoted to Production
registered_model = "models:/custom_model/Production"

# Code copied from the MLflow Artifact sample code
loaded_model = mlflow.pyfunc.load_model(registered_model)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Prediction

# COMMAND ----------

import pandas as pd

# data needs to be in the correct format to match the model signature
data = pd.DataFrame([range(10)])

# COMMAND ----------

loaded_model.predict(data)

# COMMAND ----------


