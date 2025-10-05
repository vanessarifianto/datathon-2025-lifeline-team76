import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -----------------------------
# Config
# -----------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"
FEATURES = [
    'LB','AC','FM','UC','ASTV','mSTV','ALTV','mLTV',
    'DL','DS','DP','Width','Min','Max','Nmax','Nzeros',
    'Mode','Mean','Median','Variance','Tendency'
]

# -----------------------------
# Data loader (auto-finds label)
# -----------------------------
def load_ctg(url=URL):
    xl = pd.ExcelFile(url)
    target_candidates = ["NSP", "CLASS", "CLASS\n"]
    for sheet in xl.sheet_names:
        df = pd.read_excel(url, sheet_name=sheet)
        df.columns = df.columns.astype(str).str.strip()
        for t in target_candidates:
            if t in df.columns:
                return df, t
    raise ValueError(f"Target not found in sheets: {xl.sheet_names}")

# -----------------------------
# Build pipeline
# -----------------------------
def make_pipe(model):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", model),
    ])

def main():
    parser = argparse.ArgumentParser(description="Train CTG RF-SMOTE pipeline and save weights.")
    parser.add_argument("--excel", type=str, default=None, help="Optional local path to CTG.xls (else downloads).")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    args = parser.parse_args()

    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path(args.outputs_dir); outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.excel:
        df, target_col = load_ctg(args.excel)
    else:
        df, target_col = load_ctg()

    # Select features/label
    feats = [c for c in FEATURES if c in df.columns]
    data = df[feats + [target_col]].dropna().copy()
    y = data[target_col].astype(int)
    if y.min() == 1:  # map 1,2,3 -> 0,1,2
        y = y - 1
    X = data[feats].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Baselines (optional logging)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, mdl in {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RF300": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }.items():
        scores = cross_validate(
            make_pipe(mdl), X, y, cv=cv,
            scoring={"macro_f1": "f1_macro", "bal_acc": "balanced_accuracy"},
            n_jobs=-1
        )
        print(f"{name} | Macro-F1={scores['test_macro_f1'].mean():.3f} | BalAcc={scores['test_bal_acc'].mean():.3f}")

    # GridSearchCV for RF
    rf_pipe = make_pipe(RandomForestClassifier(class_weight="balanced", random_state=42))
    param_grid = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 6, 10],
        "clf__min_samples_split": [2, 5],
    }
    grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print("\nBest RF params:", grid.best_params_)
    print("Best Macro-F1 (CV):", grid.best_score_)

    best_model = grid.best_estimator_

    # Hold-out evaluation
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    bal = balanced_accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro")
    print("\nHold-out evaluation (best RF):")
    print("Balanced Acc:", bal)
    print("Macro F1    :", mf1)
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("RandomForest (Best) — Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    (outputs_dir / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outputs_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Feature importances CSV
    rf_fitted = best_model.named_steps["clf"]
    importances = pd.Series(rf_fitted.feature_importances_, index=feats).sort_values(ascending=False)
    importances.to_csv(outputs_dir / "feature_importances.csv")

    # Metrics JSON
    with open(outputs_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"balanced_accuracy": float(bal), "macro_f1": float(mf1)}, f, indent=2)

    # Save weights
    import joblib
    weights_path = models_dir / "rf_smote_pipeline.joblib"
    joblib.dump(best_model, weights_path)
    print(f"\nSaved model to: {weights_path.resolve()}")
    print(f"Artifacts in: {outputs_dir.resolve()}")

if __name__ == "__main__":
    main()
