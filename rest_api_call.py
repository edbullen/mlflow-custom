import pandas as pd
import os, json
import requests

"""
Example - call MLflow Serving API to score data payload against model served out of MLflow

"""

# replace this URL with the URL of the MLflow model serving endpoint.
MLFLOW_URL = 'https://7177936340822399.9.gcp.databricks.com/model/hybridmodel/Production/invocations'


def get_token():
    """ Set the Databricks PAT token as a local environment variable"""
    return os.environ.get("DATABRICKS_TOKEN")


def call_model(api_url, token, dataset: pd.DataFrame):
    """ Data passed in as a Pandas DataFrame  passed to the MLflow REST API for predictions"""

    headers = {'Authorization': f'Bearer {token}'}

    data_core = dataset.to_dict(orient='records')
    data_json = {"dataframe_records": data_core}

    response = requests.request(method='POST', headers=headers, url=api_url, json=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


if __name__ == '__main__':
    token = get_token()

    payload_df = pd.DataFrame([range(-2, 13)])

    # Call the MLFlow Model API
    model_results = call_model(MLFLOW_URL, token, payload_df)
    print(model_results)
