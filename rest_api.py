import json
import logging
import os
import pathlib
import pickle
from datetime import datetime
from functools import lru_cache
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


MODEL_PATH = pathlib.Path("model/model.pkl")
FEATURES_PATH = pathlib.Path("model/model_features.json")
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
# Assume the CI/CD will set the container tag based on semantic versioning
CONTAINER_TAG = os.getenv("CONTAINER_TAG", "latest")

logger = logging.getLogger(__name__)

app = FastAPI()

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)


class HomeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


class MinimalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: str


# Abstractable "feature store" for demographics (low-effort: load once, cache lookups)
@lru_cache(maxsize=1)
def get_demographics():
    return pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str}).set_index("zipcode")


@app.post("/predict")
def predict(home: HomeInput) -> Dict:
    return _predict(home.dict())


# Bonus endpoint
@app.post("/predict_minimal")
def predict_minimal(home: MinimalInput) -> Dict:
    return _predict(home.dict())


def _predict(input_data: Dict) -> Dict:
    try:
        input_df = pd.DataFrame([input_data])
        
        demo_df = get_demographics()
        zipcode = input_data["zipcode"]

        if zipcode in demo_df.index:
            imputed = False
            zip_data = demo_df.loc[zipcode]
        else:
            # Impute (mean from all demographics data)
            imputed = True
            zip_data = demo_df.mean()

        for col, val in zip_data.items():
            input_df[col] = val
        
        features_df = input_df[model_features]
        prediction = model.predict(features_df)[0]
        
        return {
            "predicted_price": float(prediction),
            "metadata": {
                "input": input_data,
                "timestamp": datetime.now().isoformat(),
                "container_tag": CONTAINER_TAG,
                "imputed_demographics": imputed
            }
        }

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    workers = min(os.cpu_count() or 1, 4)
    uvicorn.run("rest_api:app", host="0.0.0.0", port=8000, workers=workers)

