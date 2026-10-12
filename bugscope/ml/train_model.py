#!/usr/bin/env python3
"""BugScope — train defect prediction models and write evaluation artifacts."""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
EVAL = os.path.join(BASE, "evaluation")
DB = os.path.join(os.path.dirname(BASE), "database")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(EVAL, exist_ok=True)
os.makedirs(DB, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "software_metrics.csv"))
print(f"Records: {len(df)} | Defect rate: {df['defective'].mean():.3f}")

cat_cols = ["language", "module_type"]
num_cols = [c for c in df.columns if c not in cat_cols + ["defective", "defect_count", "severity"]]
feature_cols = cat_cols + num_cols

y = df["defective"].values
encoders = {}
X_df = df[feature_cols].copy()
for c in cat_cols:
    le = LabelEncoder()
    X_df[c] = le.fit_transform(X_df[c].astype(str))
    encoders[c] = le

scaler = StandardScaler()
X_df[num_cols] = scaler.fit_transform(X_df[num_cols])
X = X_df.values

selector = SelectKBest(mutual_info_classif, k=min(18, X.shape[1]))
X_sel = selector.fit_transform(X, y)
selected = [f for f, m in zip(feature_cols, selector.get_support()) if m]

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=14, random_state=42,
        n_jobs=-1, class_weight="balanced_subsample"
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=120, max_depth=5, random_state=42, learning_rate=0.08
    ),
}

results = {}
best_name, best_score, best_model = "", 0, None
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"{name}: F1={results[name]['f1']:.3f} AUC={results[name]['roc_auc']:.3f}")
    if results[name]["f1"] > best_score:
        best_score = results[name]["f1"]
        best_name, best_model = name, model

importances = {}
if hasattr(best_model, "feature_importances_"):
    importances = dict(zip(selected, [float(v) for v in best_model.feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

joblib.dump(best_model, os.path.join(MODELS, "defect_model.joblib"))
joblib.dump(encoders, os.path.join(MODELS, "encoders.joblib"))
joblib.dump(scaler, os.path.join(MODELS, "scaler.joblib"))
joblib.dump(selector, os.path.join(MODELS, "selector.joblib"))
joblib.dump(feature_cols, os.path.join(MODELS, "feature_cols.joblib"))
joblib.dump(selected, os.path.join(MODELS, "selected_features.joblib"))
joblib.dump(num_cols, os.path.join(MODELS, "num_cols.joblib"))
joblib.dump(cat_cols, os.path.join(MODELS, "cat_cols.joblib"))

meta = {
    "best_model": best_name,
    "results": results,
    "feature_importance": importances,
    "selected_features": selected,
    "n_samples": int(len(df)),
    "defect_rate": float(y.mean()),
}
with open(os.path.join(EVAL, "metrics.json"), "w") as f:
    json.dump(meta, f, indent=2)

eda = {
    "total_records": int(len(df)),
    "defective_modules": int(y.sum()),
    "clean_modules": int(len(y) - y.sum()),
    "mean_loc": float(df["loc"].mean()),
    "mean_complexity": float(df["cyclomatic_complexity"].mean()),
    "mean_churn": float(df["code_churn"].mean()),
    "languages": df["language"].value_counts().to_dict(),
    "module_types": df["module_type"].value_counts().to_dict(),
}
with open(os.path.join(EVAL, "eda_stats.json"), "w") as f:
    json.dump(eda, f, indent=2)

df.to_csv(os.path.join(DB, "software_metrics.csv"), index=False)
print(f"Best: {best_name} | Artifacts → {MODELS}")
