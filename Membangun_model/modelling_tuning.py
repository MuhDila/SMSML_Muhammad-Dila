import os
import urllib.request
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import matplotlib.pyplot as plt

# Inisialisasi DagsHub
dagshub.init(repo_owner='MuhDila', repo_name='Eksperimen_SML_Muhammad-Dila', mlflow=True)

# Download dataset
os.makedirs("dataset_preprocessing", exist_ok=True)
base_url = "https://raw.githubusercontent.com/MuhDila/Eksperimen_SML_Muhammad-Dila/master/preprocessing/dataset_preprocessing/"
files = ["x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy"]
for file in files:
    save_path = os.path.join("dataset_preprocessing", file)
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(base_url + file, save_path)

# Load dataset
X_train = np.load('dataset_preprocessing/x_train.npy')
y_train = np.load('dataset_preprocessing/y_train.npy')
X_val = np.load('dataset_preprocessing/x_val.npy')
y_val = np.load('dataset_preprocessing/y_val.npy')

# Setup MLflow experiment
mlflow.set_experiment("Modelling_Tuning_Advanced")

n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, 15]

for n in n_estimators_list:
    for depth in max_depth_list:
        with mlflow.start_run():
            model = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Log param & metric
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)

            # Log model
            model_name = f"model_rf_{n}_{depth}"
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            # Save dan log metric info JSON
            metric_info = {"mse": mse, "r2_score": r2}
            with open("metric_info.json", "w") as f:
                json.dump(metric_info, f)
            mlflow.log_artifact("metric_info.json")

            # Buat dan log visualisasi residu
            residuals = y_val - y_pred
            plt.figure(figsize=(6,4))
            plt.hist(residuals, bins=20)
            plt.title(f"Residuals Histogram (n={n}, depth={depth})")
            plt.savefig("residuals_hist.png")
            plt.close()
            mlflow.log_artifact("residuals_hist.png")

print("Advanced hyperparameter tuning dan DagsHub logging selesai.")
