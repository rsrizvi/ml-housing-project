import pandas as pd
import requests


ENDPOINT = "http://localhost:8000/predict"


data = pd.read_csv("data/future_unseen_examples.csv", dtype={'zipcode': str})

for idx, row in data.iterrows():
    example = row.to_dict()
    response = requests.post(ENDPOINT, json=example)
    response.raise_for_status()

    out = response.json()
    print(f"Example {idx + 1}:", out)
