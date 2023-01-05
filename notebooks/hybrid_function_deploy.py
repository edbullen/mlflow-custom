# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Custom Function to MLflow
# MAGIC Notebook to deploy an instance of the `HybridFunction` class in MLflow with function attributes set to parameterise the custom function.

# COMMAND ----------

import mlflow
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Class with Custom Function

# COMMAND ----------

# import from ./functions/HybridFunction.py - reference from the root of this Repo
from functions.HybridFunction import HybridFunction

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Experiment

# COMMAND ----------

# instantiate the custom model with parameters that define the model characteristics (see repo README)
model = HybridFunction(x0=5, y0=2, gradient=1)

# COMMAND ----------

# log the model as an MLflow experiment
mlflow.set_experiment("/Users/ed.bullen@databricks.com/hybrid_model")
response = mlflow.pyfunc.log_model("custom_model", python_model=model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Model-Run in Experiment

# COMMAND ----------


# Set any tags we want in the Run to help identify it
mlflow.set_tag("project", "hybrid_model")

# Set some metrics that provide information about the run - i.e. the parameters associated with the model-instance
mlflow.log_metric('x0', 5)
mlflow.log_metric('y0', 2)
mlflow.log_metric('gradient', 1)

mlflow.end_run() 


# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Model Registry
# MAGIC 
# MAGIC Search for a model to register   
# MAGIC https://mlflow.org/docs/latest/search-runs.html#python   `mlflow.search_runs(...`   
# MAGIC or just select this one (specifying the `model_uri`.

# COMMAND ----------

# the response variable from earlier mlflow.pyfunc.log_model() call has information we can use
print(f"Artifact Path: \t {response.artifact_path}")
print(f"Run ID: \t {response.run_id}")
print(f"Model URI: \t {response.model_uri}")

# COMMAND ----------

# register model - after this will appear in the Databricks UI "Models" view
registry_response = mlflow.register_model(response.model_uri, "custom_model")

# COMMAND ----------

# update some meta-data details about the model like the description
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.update_model_version(
    version=registry_response.version,
    name = registry_response.name,
    description="This is a custom model combining two different functions depending on the value of x applied to the fn y=f(x)"
)


# COMMAND ----------

print(f"Model Version Number: \t {registry_response.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote Model to Production

# COMMAND ----------

client.transition_model_version_stage("custom_model", registry_response.version, stage = "Production", archive_existing_versions=True)


# COMMAND ----------


