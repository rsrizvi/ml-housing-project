import pandas as pd
import requests

from rest_api import MinimalInput


ENDPOINT = "http://localhost:8000/predict"
BONUS_ENDPOINT = "http://localhost:8000/predict_minimal"


data = pd.read_csv("data/future_unseen_examples.csv", dtype={'zipcode': str})

for idx, row in data.iterrows():
    example = row.to_dict()
    response = requests.post(ENDPOINT, json=example)
    response.raise_for_status()

    out = response.json()
    print(f"Example {idx + 1}:", out)

    example_minimal = {k: example[k] for k in MinimalInput.model_fields.keys()}
    response = requests.post(BONUS_ENDPOINT, json=example_minimal)
    response.raise_for_status()

    out = response.json()
    print(f"Example {idx + 1} minimal:", out, "\n")
    
