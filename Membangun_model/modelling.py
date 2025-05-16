import os
import urllib.request
import mlflow.sklearn
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Download dataset dari GitHub
os.makedirs("dataset_preprocessing", exist_ok=True)

base_url = "https://raw.githubusercontent.com/MuhDila/Eksperimen_SML_Muhammad-Dila/master/preprocessing/dataset_preprocessing/"
files = ["x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy"]

for file in files:
    url = base_url + file
    save_path = os.path.join("dataset_preprocessing", file)
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(url, save_path)

# Load dataset
X_train = np.load('dataset_preprocessing/x_train.npy')
y_train = np.load('dataset_preprocessing/y_train.npy')
X_val = np.load('dataset_preprocessing/x_val.npy')
y_val = np.load('dataset_preprocessing/y_val.npy')

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Jalankan: mlflow ui
mlflow.set_experiment("Modelling_Muhammad-Dila")

# Training dan Logging
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mlflow.log_param("dataset", "dataset_preprocessing")
    mlflow.log_param("X_train_shape", X_train.shape)
    mlflow.log_param("X_val_shape", X_val.shape)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Simpan dan log model
    joblib.dump(model, "model_rf.pkl", compress=3)
    mlflow.log_artifact("model_rf.pkl", artifact_path="model")

print("Training selesai.")
