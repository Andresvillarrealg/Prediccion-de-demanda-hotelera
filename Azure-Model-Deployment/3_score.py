import os
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

def init():
    global model, scaler, feature_cols, features_to_scale, col_idx
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir is None:
        raise Exception("No se encontró la variable de entorno AZUREML_MODEL_DIR")
    outputs_folder = os.path.join(model_dir, "outputs")
    model_path = os.path.join(outputs_folder, "lgb_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el archivo de modelo: {model_path}")
    model = joblib.load(model_path)
    scaler_params_path = os.path.join(outputs_folder, "scaler_params.json")
    if os.path.exists(scaler_params_path):
        with open(scaler_params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        scaler = MinMaxScaler(feature_range=tuple(params["feature_range"]))
        scaler.data_min_   = np.array(params["data_min"])
        scaler.data_max_   = np.array(params["data_max"])
        scaler.data_range_ = np.array(params["data_range"])
        scaler.scale_      = np.array(params["scale"])
        scaler.min_        = np.array(params["min"])
        scaler.n_features_in_ = int(params["n_features_in"])
    else:
        scaler = None
    cols_path = os.path.join(outputs_folder, "feature_cols.json")
    if os.path.exists(cols_path):
        with open(cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        raise FileNotFoundError(f"No existe el archivo feature_cols.json: {cols_path}")
    features_to_scale = [
        "personas", "lag_1", "lag_7", "lag_14", "rolling_mean_7",
        "dow_sin", "dow_cos", "month_sin", "month_cos"
    ]
    if "personas" not in features_to_scale:
        raise Exception("'personas' no está en la lista features_to_scale")
    col_idx = features_to_scale.index("personas")

def _inverse_scale(y_scaled):
    if scaler is None:
        return y_scaled
    tmp = np.zeros((len(y_scaled), len(features_to_scale)))
    tmp[:, col_idx] = y_scaled
    return scaler.inverse_transform(tmp)[:, col_idx]

def run(raw_body):
    try:
        payload = json.loads(raw_body)
        df = pd.DataFrame(payload)
        X = df[feature_cols]
        preds_scaled = model.predict(X)
        preds_real = _inverse_scale(preds_scaled)
        return json.dumps({"predictions": preds_real.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})