"""BugScope – Model Training Pipeline for Software Defect Prediction."""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "software_metrics.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape}")

    cat_cols = ["language", "module_type"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["defective", "defect_count", "severity"]]
    feature_cols = cat_cols + num_cols

    y = df["defective"].values
    le_dict = {}
    X_df = df[feature_cols].copy()
    for c in cat_cols:
        le = LabelEncoder()
        X_df[c] = le.fit_transform(X_df[c].astype(str))
        le_dict[c] = le

    scaler = StandardScaler()
    X_df[num_cols] = scaler.fit_transform(X_df[num_cols])
    X = X_df.values

    selector = SelectKBest(mutual_info_classif, k=min(18, X.shape[1]))
    X_sel = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
    print("Selected:", selected_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=14, random_state=42,
            n_jobs=-1, class_weight="balanced_subsample"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=5, random_state=42, learning_rate=0.08
        ),
    }

    results = {}
    best_model, best_score, best_name = None, 0, ""
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0
        results[name] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "roc_auc": float(auc),
            "report": classification_report(y_test, y_pred, target_names=["Clean", "Defective"], output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        print(f"{name}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
        if f1 > best_score:
            best_score, best_model, best_name = f1, model, name

    importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(selected_features, best_model.feature_importances_.tolist()))

    joblib.dump(best_model, MODEL_DIR / "defect_model.joblib")
    joblib.dump(le_dict, MODEL_DIR / "encoders.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(selector, MODEL_DIR / "selector.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.joblib")
    joblib.dump(selected_features, MODEL_DIR / "selected_features.joblib")
    joblib.dump(num_cols, MODEL_DIR / "num_cols.joblib")
    joblib.dump(cat_cols, MODEL_DIR / "cat_cols.joblib")

    meta = {
        "best_model": best_name,
        "results": results,
        "selected_features": selected_features,
        "feature_importances": importances,
        "n_samples": int(len(df)),
        "n_features_original": len(feature_cols),
        "n_features_selected": len(selected_features),
        "defect_rate": float(y.mean()),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nBest: {best_name} (F1={best_score:.4f})")
    print("Artifacts saved.")

if __name__ == "__main__":
    main()
