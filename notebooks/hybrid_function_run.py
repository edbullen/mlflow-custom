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
registered_model = "models:/hybridfunction/Production"

# Code copied from the MLflow Artifact sample code
loaded_model = mlflow.pyfunc.load_model(registered_model, suppress_warnings=True)


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

# MAGIC %md 
# MAGIC ## Simulate using model for batch-scoring against Delta-Lake data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pyspark DF Example

# COMMAND ----------

pyspark_df = spark.read.option("inferSchema", True)\
    .option("header", True)\
    .option("delimiter", ";")\
    .csv("/databricks-datasets/wine-quality/winequality-red.csv")

# COMMAND ----------

display(pyspark_df.head(5))

# COMMAND ----------

custom_model_udf = mlflow.pyfunc.spark_udf(spark, "models:/hybridfunction/Production")

# COMMAND ----------

from pyspark.sql import functions as F
pyspark_df.withColumn('prediction', 
                      F.round(custom_model_udf(F.col("fixed acidity")),2))\
           .select("fixed acidity", "prediction").show(100, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL Query Example

# COMMAND ----------

pyspark_df.createOrReplaceTempView("wine_data_tmp")

# COMMAND ----------

# register the model as a sql function that can be accessed in SQL
spark.udf.register("custom_model_udf", custom_model_udf)

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT `fixed acidity`, custom_model_udf(`fixed acidity`) as prediction
# MAGIC FROM wine_data_tmp;
