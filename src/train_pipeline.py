import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
import joblib

DATA_PATH = os.path.join("data", "raw", "employee_attrition.csv")
REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
MODEL_DIR = "reports", "models"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "models"), exist_ok=True)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()

def main():
    print("=" * 70)
    print("WORKFORCE ANALYTICS & EMPLOYEE ATTRITION PREDICTION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    # ---------------- DATA QUALITY ----------------
    quality = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target": "Attrition",
        "target_distribution": df["Attrition"].value_counts().to_dict()
    }

    print("\nDATA QUALITY")
    print(json.dumps(quality, indent=2))

    with open(os.path.join(REPORT_DIR, "data_quality.json"), "w") as f:
        json.dump(quality, f, indent=2, default=str)

    # Remove obvious identifier columns.
    drop_cols = [c for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ---------------- EDA ----------------
    attrition_rate = df["Attrition"].eq("Yes").mean()
    print(f"\nOverall attrition rate: {attrition_rate:.2%}")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Attrition")
    plt.title("Employee Attrition Distribution")
    save_fig("01_attrition_distribution.png")

    if "Department" in df.columns:
        dept = pd.crosstab(df["Department"], df["Attrition"], normalize="index") * 100
        dept["Yes"].sort_values().plot(kind="barh", figsize=(7, 4))
        plt.xlabel("Attrition rate (%)")
        plt.title("Attrition Rate by Department")
        save_fig("02_attrition_by_department.png")

    if "OverTime" in df.columns:
        overtime = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index") * 100
        overtime["Yes"].plot(kind="bar", figsize=(7, 4))
        plt.ylabel("Attrition rate (%)")
        plt.title("Attrition Rate by Overtime")
        save_fig("03_attrition_by_overtime.png")

    if "JobSatisfaction" in df.columns:
        sat = df.groupby("JobSatisfaction")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100)
        sat.plot(kind="bar", figsize=(7, 4))
        plt.ylabel("Attrition rate (%)")
        plt.title("Attrition Rate by Job Satisfaction")
        save_fig("04_attrition_by_satisfaction.png")

    # Numeric correlation heatmap
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        plt.figure(figsize=(12, 9))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlation Matrix")
        save_fig("05_correlation_heatmap.png")

    # ---------------- FEATURE ENGINEERING ----------------
    df["AttritionFlag"] = df["Attrition"].map({"No": 0, "Yes": 1})

    if "MonthlyIncome" in df.columns and "JobLevel" in df.columns:
        df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"].replace(0, np.nan)

    if "YearsAtCompany" in df.columns and "YearsSinceLastPromotion" in df.columns:
        df["YearsSincePromotionRatio"] = (
            df["YearsSinceLastPromotion"] /
            df["YearsAtCompany"].replace(0, np.nan)
        )

    if "YearsAtCompany" in df.columns and "YearsInCurrentRole" in df.columns:
        df["RoleTenureRatio"] = (
            df["YearsInCurrentRole"] /
            df["YearsAtCompany"].replace(0, np.nan)
        )

    # ---------------- STATISTICAL ANALYSIS ----------------
    stats_results = {}

    # Continuous variable comparison
    if "MonthlyIncome" in df.columns:
        yes = df.loc[df["Attrition"] == "Yes", "MonthlyIncome"].dropna()
        no = df.loc[df["Attrition"] == "No", "MonthlyIncome"].dropna()
        stat, p = ttest_ind(yes, no, equal_var=False)
        stats_results["monthly_income_t_test"] = {
            "t_statistic": float(stat),
            "p_value": float(p)
        }

    # Categorical association
    if "OverTime" in df.columns:
        table = pd.crosstab(df["OverTime"], df["Attrition"])
        chi2, p, dof, expected = chi2_contingency(table)
        stats_results["overtime_chi_square"] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "degrees_of_freedom": int(dof)
        }

    with open(os.path.join(REPORT_DIR, "statistical_tests.json"), "w") as f:
        json.dump(stats_results, f, indent=2)

    print("\nSTATISTICAL TESTS")
    print(json.dumps(stats_results, indent=2))

    # ---------------- ML ----------------
    X = df.drop(columns=["Attrition", "AttritionFlag"])
    y = df["AttritionFlag"]

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob)
        }
        results.append(metrics)
        fitted[name] = pipe

        print(f"\n{name}")
        print(pd.Series(metrics).to_string())

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        save_fig(name.lower().replace(" ", "_") + "_confusion_matrix.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

    # Select best by F1 for imbalanced classification.
    best_name = results_df.sort_values("F1", ascending=False).iloc[0]["Model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, os.path.join(REPORT_DIR, "models", "best_attrition_model.joblib"))

    # ---------------- CROSS VALIDATION ----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1")
    print(f"\nBest model: {best_name}")
    print(f"5-fold CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # ---------------- PERMUTATION IMPORTANCE ----------------
    try:
        perm = permutation_importance(
            best_model, X_test, y_test,
            n_repeats=10, random_state=42, scoring="f1", n_jobs=-1
        )
        importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False)

        importance.to_csv(
            os.path.join(REPORT_DIR, "feature_importance.csv"), index=False
        )

        top = importance.head(12).sort_values("importance_mean")
        top.plot(x="feature", y="importance_mean", kind="barh", figsize=(8, 6), legend=False)
        plt.xlabel("Permutation importance (F1 decrease)")
        plt.title(f"Top Features - {best_name}")
        save_fig("06_feature_importance.png")
    except Exception as e:
        print("Feature importance skipped:", e)

    print("\nFiles generated inside reports/:")
    print("- data_quality.json")
    print("- statistical_tests.json")
    print("- model_comparison.csv")
    print("- feature_importance.csv")
    print("- models/best_attrition_model.joblib")
    print("- figures/*.png")
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
