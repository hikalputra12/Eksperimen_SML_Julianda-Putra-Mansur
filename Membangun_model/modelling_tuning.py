import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV

def main():
    import sys
    import io
    # Force stdout and stderr to UTF-8 to prevent UnicodeEncodeError in MLflow on Windows
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    # Inisialisasi DagsHub tracking remote
    dagshub.init(repo_owner='hikalputra12', repo_name='Eksperimen_SML_Julianda-Putra-Mansur', mlflow=True)
    dagshub_uri = mlflow.get_tracking_uri()
    tracking_uris = [dagshub_uri, "http://127.0.0.1:5000"]

    # Mengambil dataset bersih secara dinamis
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "Eksperimen_SML_Julianda-Putra-Mansur", "preprocessing", "loan_approval_preprocessing", "dataset_clean.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data bersih tidak ditemukan di {data_path}. Jalankan kriteria 1 terlebih dahulu!")
        
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simpan status eksekusi agar tidak mencetak teks berulang secara membingungkan
    print("="*50)
    print("[+] Memulai Hyperparameter Tuning dan logging ke DagsHub serta Localhost...")
    print("="*50)

    for uri in tracking_uris:
        # Hubungkan ke tracking server tujuan
        mlflow.set_tracking_uri(uri)
        
        # Inisialisasi Eksperimen MLflow Tuning
        mlflow.set_experiment("Loan_Approval_Tuning")
        
        # Mengaktifkan automatic logging (dibatasi max_tuning_runs=0 dan tanpa log_post_training_metrics agar tidak error HTTP 500 di localhost)
        mlflow.sklearn.autolog(max_tuning_runs=0, log_post_training_metrics=False)
        
        with mlflow.start_run(run_name="RandomForest_BayesianSearch"):
            # Ruang pencarian Hyperparameter untuk Bayesian Optimization
            search_spaces = {
                'n_estimators': (50, 150),
                'max_depth': (5, 15),
                'min_samples_split': (2, 10)
            }
            
            rf = RandomForestClassifier(random_state=42)
            # Menggunakan BayesSearchCV untuk optimasi Bayesian
            opt = BayesSearchCV(
                estimator=rf,
                search_spaces=search_spaces,
                n_iter=10, # jumlah iterasi pencarian
                cv=3,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1
            )
            opt.fit(X_train, y_train)
            
            best_model = opt.best_estimator_
            best_params = opt.best_params_
            
            # Logging parameter terbaik hasil pencarian secara manual
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
                
            # Evaluasi model terbaik hasil tuning
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Logging metrik hasil tuning secara manual
            mlflow.log_metric("best_accuracy", acc)
            mlflow.log_metric("best_precision", prec)
            mlflow.log_metric("best_recall", rec)
            mlflow.log_metric("best_f1_score", f1)
            
            # Logging model tersetel terbaik sebagai artefak resmi (dalam folder 'model' sesuai instruksi gambar)
            mlflow.sklearn.log_model(best_model, "model")
            
            # Membuat dan mengunggah estimator.html (representasi visual model)
            from sklearn.utils import estimator_html_repr
            estimator_path = "estimator.html"
            with open(estimator_path, "w", encoding="utf-8") as f:
                f.write(estimator_html_repr(best_model))
            mlflow.log_artifact(estimator_path)
            if os.path.exists(estimator_path):
                os.remove(estimator_path)

            # Membuat dan mengunggah metric_info.json (informasi metrik evaluasi)
            import json
            metrics_data = {
                "best_accuracy": acc,
                "best_precision": prec,
                "best_recall": rec,
                "best_f1_score": f1
            }
            metric_info_path = "metric_info.json"
            with open(metric_info_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=4)
            mlflow.log_artifact(metric_info_path)
            if os.path.exists(metric_info_path):
                os.remove(metric_info_path)

            # Membuat dan mengunggah training_confusion_matrix.png
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
            plt.title('Confusion Matrix')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = "training_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            if os.path.exists(cm_path):
                os.remove(cm_path)
                
            # Membuat dan mengunggah Feature Importance Plot (Tuned) sebagai artefak tambahan
            import numpy as np
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = X.columns
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances - Tuned Model")
            plt.bar(range(X.shape[1]), importances[indices], align="center")
            plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            fi_path = "feature_importance_tuned.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)
            if os.path.exists(fi_path):
                os.remove(fi_path)
            
            print(f"[+] Hyperparameter Tuning Selesai & Di-log ke {uri}!")
            print(f"    Best Params  : {dict(best_params)}")
            print(f"    Best Accuracy: {acc:.4f}")
            print("-" * 50)

if __name__ == "__main__":
    main()