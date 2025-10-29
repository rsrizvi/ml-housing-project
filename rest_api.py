import json
import logging
import os
import pathlib
import pickle
from datetime import datetime
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = pathlib.Path("model/model.pkl")
FEATURES_PATH = pathlib.Path("model/model_features.json")
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"

logger = logging.getLogger(__name__)

app = FastAPI()

with open("model/model.pkl", 'rb') as f:
    # TODO use a service for this
    model = pickle.load(f)
with open(FEATURES_PATH, 'r') as f:
    # TODO use a feature store for this
    model_features = json.load(f)
demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})


# TODO use a feature store for this
class HomeInput(BaseModel):
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


@app.post("/predict")
def predict(home: HomeInput) -> Dict:
    try:
        input_df = pd.DataFrame([home.dict()])

        # TODO use a feature store for this
        merged_df = input_df.merge(demographics, how="left", on="zipcode")

        # Impute missing demographics
        if merged_df.isnull().values.any():
            # TODO improve this
            raise RuntimeError("Missing demographics")

        features_df = merged_df[model_features]

        # TODO use a service for this
        prediction = model.predict(features_df)[0]

        return {
            "predicted_price": float(prediction),
            "metadata": {
                "input": home.dict(),
                "timestamp": datetime.now().isoformat(),
                "model_version": "v1"  # TODO fix this
            }
        }

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error")


# TODO bonus endpoint


if __name__ == "__main__":
    # TODO add a reverse proxy layer with auth + load balancing
    import uvicorn
    workers = min(os.cpu_count() or 1, 4)
    uvicorn.run("rest_api:app", host="0.0.0.0", port=8000, workers=workers)

