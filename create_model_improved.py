import json
import pathlib
import pickle

import pandas as pd
import shap
import xgboost as xgb
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from create_model import DEMOGRAPHICS_PATH, OUTPUT_DIR, RANDOM_SEED, SALES_COLUMN_SELECTION, SALES_PATH, load_data


def main():
	X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
	x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
	        X, y, random_state=RANDOM_SEED)

	xgb_model = xgb.XGBRegressor(
		random_state=RANDOM_SEED,
		objective="reg:squarederror",
		eta=0.025,                   
        max_depth=8,                
        tree_method="hist",
	)
	model = pipeline.make_pipeline(preprocessing.RobustScaler(),
	                               xgb_model).fit(
	                                   x_train, y_train)

	output_dir = pathlib.Path(OUTPUT_DIR)
	output_dir.mkdir(exist_ok=True)

	pickle.dump(model, open(output_dir / "model.pkl", "wb"))
	json.dump(list(x_train.columns),
	          open(output_dir / "model_features.json", "w"))

	# SHAP metrics
	trained_xgb = model.named_steps["xgbregressor"]
	explainer = shap.TreeExplainer(trained_xgb)
	shap_values = explainer.shap_values(model.named_steps["robustscaler"].transform(_x_test))
	shap_df = pd.DataFrame(shap_values, columns=_x_test.columns)

	shap_summary = shap_df.abs().mean().sort_values(ascending=False)
	shap_summary.to_csv("shap_feature_importance.csv")


if __name__ == "__main__":
	main()
