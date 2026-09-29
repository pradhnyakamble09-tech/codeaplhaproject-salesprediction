"""
CodeAlpha Data Science Internship — Task 4
Sales Prediction using Python

This script:
  1. Loads and cleans advertising-spend / sales data
  2. Explores relationships between spend (TV, Radio, Newspaper) and Sales
  3. Prepares features (encoding target segment & platform)
  4. Trains regression models to forecast Sales
  5. Analyzes how changes in advertising spend impact predicted sales
  6. Delivers actionable marketing insights

Run:
    python sales_prediction.py
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "advertising.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
TARGET = "Sales"
NUMERIC_FEATURES = ["TV", "Radio", "Newspaper"]
CATEGORICAL_FEATURES = ["Target_Segment", "Platform"]


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna().drop_duplicates()
    return df


def explore(df):
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(df.describe())
    print("\nMissing values:\n", df.isnull().sum())

    # Spend vs. Sales scatter + regression line, per channel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.regplot(data=df, x=col, y=TARGET, ax=ax, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
        ax.set_title(f"{col} Spend vs. Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "spend_vs_sales.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Target_Segment", y=TARGET, hue="Target_Segment", legend=False, palette="Set2")
    plt.title("Sales by Target Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sales_by_segment.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Platform", y=TARGET, hue="Platform", legend=False, palette="Set3")
    plt.title("Sales by Platform")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sales_by_platform.png"), dpi=150)
    plt.close()


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_pipeline, best_r2 = None, None, -np.inf

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        pipeline = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

        results[name] = {
            "R2": round(float(r2), 4),
            "RMSE": round(float(rmse), 3),
            "MAE": round(float(mae), 3),
            "CV_R2_mean": round(float(cv_scores.mean()), 4),
        }
        print(f"\n{name}")
        print(f"  R2: {r2:.4f}  |  RMSE: {rmse:.3f}  |  MAE: {mae:.3f}  |  CV R2: {cv_scores.mean():.4f}")

        if r2 > best_r2:
            best_name, best_pipeline, best_r2 = name, pipeline, r2

    plt.figure(figsize=(7, 4))
    names = list(results.keys())
    r2s = [results[n]["R2"] for n in names]
    sns.barplot(x=r2s, y=names, hue=names, palette="crest", legend=False)
    plt.xlabel("R² Score")
    plt.title("Model Comparison — R² on Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # Actual vs predicted for the best model
    y_pred_best = best_pipeline.predict(X_test)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, y_pred_best, alpha=0.5, color="#2b6cb0")
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs. Predicted Sales — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.pkl"))
    return results, best_name, best_pipeline, X_test, y_test


def advertising_impact_analysis(df, best_pipeline):
    """Simulate scaling each channel's spend to show marginal sales impact."""
    baseline = pd.DataFrame([{
        "TV": df["TV"].median(),
        "Radio": df["Radio"].median(),
        "Newspaper": df["Newspaper"].median(),
        "Target_Segment": df["Target_Segment"].mode()[0],
        "Platform": df["Platform"].mode()[0],
    }])
    baseline_pred = best_pipeline.predict(baseline)[0]

    impact = {}
    for channel in NUMERIC_FEATURES:
        scenario = baseline.copy()
        scenario[channel] = baseline[channel] * 1.20  # +20% spend
        new_pred = best_pipeline.predict(scenario)[0]
        impact[channel] = {
            "baseline_sales": round(float(baseline_pred), 3),
            "sales_after_20pct_increase": round(float(new_pred), 3),
            "estimated_lift": round(float(new_pred - baseline_pred), 3),
        }

    plt.figure(figsize=(7, 4.5))
    channels = list(impact.keys())
    lifts = [impact[c]["estimated_lift"] for c in channels]
    sns.barplot(x=channels, y=lifts, hue=channels, palette="flare", legend=False)
    plt.ylabel("Estimated Sales Lift")
    plt.title("Estimated Sales Lift from +20% Spend, by Channel")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "advertising_impact.png"), dpi=150)
    plt.close()

    return impact


def main():
    df = load_and_clean()
    explore(df)
    results, best_name, best_pipeline, X_test, y_test = train_and_evaluate(df)
    impact = advertising_impact_analysis(df, best_pipeline)

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print("=" * 60)
    print("\nEstimated impact of a +20% spend increase (from median baseline):")
    for channel, vals in impact.items():
        print(f"  {channel}: +{vals['estimated_lift']} sales lift")

    with open(os.path.join(BASE_DIR, "results.json"), "w") as f:
        json.dump({
            "model_results": results,
            "best_model": best_name,
            "advertising_impact_20pct_increase": impact,
        }, f, indent=2)

    print("\nAll plots saved to ./plots/  |  Trained model saved to ./models/best_model.pkl")


if __name__ == "__main__":
    main()
