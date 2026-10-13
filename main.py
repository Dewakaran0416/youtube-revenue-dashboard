# ============================================================
# main.py — YouTube Ad Revenue: EDA + Cleaning + Modelling
# Run once:  python main.py          (saves charts to ./figures)
#            python main.py --show   (also pops the charts up)
# Outputs (next to this file):
#   model.joblib     -> full sklearn Pipeline (impute + encode + scale + model)
#   artifacts.json   -> metrics, feature importance, coefficients, metadata
#   test_predictions.csv -> actual vs predicted (for the app's charts)
# ============================================================
import sys
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")                      # don't block on plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path(__file__).resolve().parent
DATA = BASE / "youtube_ad_revenue_dataset.csv"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
SEED = 42
TARGET = "ad_revenue_usd"


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=120)
    if "--show" in sys.argv:
        plt.show()
    plt.close()


# ============================================================
# STEP 1-2: Load & understand
# ============================================================
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())

# ============================================================
# STEP 3: EDA (charts saved to ./figures)
# ============================================================
plt.figure(figsize=(8, 4))
sns.histplot(df[TARGET], bins=50, kde=True, color="steelblue")
plt.title("Distribution of Ad Revenue (USD)"); save("01_target_distribution")

plt.figure(figsize=(9, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap"); save("02_correlation")

for col in ["category", "device", "country"]:
    plt.figure(figsize=(8, 4))
    df.groupby(col)[TARGET].mean().sort_values().plot(kind="barh", color="coral")
    plt.title(f"Average Ad Revenue by {col.title()}"); save(f"03_revenue_by_{col}")

plt.figure(figsize=(7, 4))
plt.scatter(df["watch_time_minutes"], df[TARGET], s=3, alpha=0.2)
plt.xlabel("Watch time (min)"); plt.ylabel("Ad revenue (USD)")
plt.title("Watch Time vs Ad Revenue"); save("04_watchtime_vs_revenue")

# ============================================================
# STEP 4: Cleaning
# ============================================================
df = df.drop_duplicates().reset_index(drop=True)
print("After removing duplicates:", df.shape)

# Outliers: report, don't delete. Only `views` has IQR outliers (<1% of rows)
# and they are plausible values, not errors. Tree models and a linear model
# fitted on a truly linear relationship are not hurt by them.
q1, q3 = df["views"].quantile([0.25, 0.75])
iqr = q3 - q1
n_out = ((df["views"] < q1 - 1.5 * iqr) | (df["views"] > q3 + 1.5 * iqr)).sum()
print(f"IQR outliers in views: {n_out} ({n_out / len(df):.2%}) -> kept")

# NOTE: missing values are NOT filled here. Imputation happens inside the
# Pipeline, fitted on the training split only (no test-set leakage).

# ============================================================
# STEP 5: Feature engineering (only from inputs known BEFORE revenue)
# ============================================================
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["engagement_rate"] = (d["likes"] + d["comments"]) / (d["views"] + 1)
    d["watch_efficiency"] = d["watch_time_minutes"] / (d["video_length_minutes"] + 1)
    dt = pd.to_datetime(d["date"])
    d["month"] = dt.dt.month
    d["day_of_week"] = dt.dt.dayofweek
    return d

# revenue_per_view is intentionally NOT created: it is computed from the
# target (ad_revenue_usd / views) = target leakage.
df = add_features(df)

NUM = ["views", "likes", "comments", "watch_time_minutes", "video_length_minutes",
       "subscribers", "engagement_rate", "watch_efficiency", "month", "day_of_week"]
CAT = ["category", "device", "country"]
FEATURES = NUM + CAT

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ============================================================
# STEP 6: Preprocessing pipeline (impute -> scale / one-hot)
# ============================================================
pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUM),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])

# ============================================================
# STEP 7: Train & evaluate 5 models
# ============================================================
models = {
    "Linear Regression":   LinearRegression(),
    "Decision Tree":       DecisionTreeRegressor(max_depth=12, random_state=SEED),
    "Random Forest":       RandomForestRegressor(n_estimators=100, max_depth=14,
                                                 n_jobs=-1, random_state=SEED),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, random_state=SEED),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
}

results, fitted = [], {}
for name, est in models.items():
    pipe = Pipeline([("pre", pre), ("model", est)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    n, p = X_test.shape
    results.append({
        "Model": name,
        "R2": round(r2, 4),
        "Adj_R2": round(1 - (1 - r2) * (n - 1) / (n - p - 1), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "MAE": round(float(mean_absolute_error(y_test, pred)), 3),
        "Train_sec": round(time.time() - t0, 1),
    })
    fitted[name] = pipe
    print(f"{name:20s} R2={r2:.4f}  RMSE={results[-1]['RMSE']:.2f}  MAE={results[-1]['MAE']:.2f}")

res = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n--- Model Comparison (test set) ---\n", res.to_string(index=False))
best_name = res.iloc[0]["Model"]
best = fitted[best_name]

# 5-fold CV on the training set for the winner (stability check)
cv = cross_val_score(Pipeline([("pre", pre), ("model", models[best_name])]),
                     X_train, y_train, cv=KFold(5, shuffle=True, random_state=SEED),
                     scoring="r2", n_jobs=-1)
print(f"\nBest: {best_name}  5-fold CV R2 = {cv.mean():.4f} ± {cv.std():.4f}")

# Accuracy on rows with NO missing inputs (what the app sees: users fill all fields)
complete = X_test.notna().all(axis=1)
r2_complete = r2_score(y_test[complete], best.predict(X_test[complete]))
print(f"R2 on test rows with no missing inputs: {r2_complete:.4f}")

# ============================================================
# STEP 8: Explainability
# ============================================================
# Permutation importance on raw features (model-agnostic)
sample = X_test.sample(4000, random_state=SEED)
perm = permutation_importance(best, sample, y_test.loc[sample.index],
                              n_repeats=5, random_state=SEED, n_jobs=-1)
importance = (pd.Series(perm.importances_mean, index=FEATURES)
              .clip(lower=0).sort_values(ascending=False))

plt.figure(figsize=(9, 4))
importance.plot(kind="bar", color="teal")
plt.title(f"Permutation Importance ({best_name})"); plt.ylabel("Drop in R²")
plt.xticks(rotation=45, ha="right"); save("05_feature_importance")

# Linear Regression coefficients back in original units ($ per 1 unit)
lr = fitted["Linear Regression"]
names = lr.named_steps["pre"].get_feature_names_out()
coef = lr.named_steps["model"].coef_
scales = lr.named_steps["pre"].named_transformers_["num"].named_steps["scale"].scale_
raw_coef = {}
for nm, c in zip(names, coef):
    key = nm.split("__", 1)[1]
    if nm.startswith("num__"):
        raw_coef[key] = float(c / scales[NUM.index(key)])
    else:
        raw_coef[key] = float(c)

# "Discovered formula": plain OLS on complete rows, 4 raw drivers only.
# On this dataset it fits with R² = 1.0 -> revenue is a deterministic linear
# function of these inputs (synthetic data). Used for the app's breakdown chart.
DRIVERS = ["views", "likes", "comments", "watch_time_minutes"]
cc = df.dropna(subset=DRIVERS)
ols = LinearRegression().fit(cc[DRIVERS], cc[TARGET])
formula = {"coef": dict(zip(DRIVERS, map(float, ols.coef_))),
           "intercept": float(ols.intercept_),
           "r2": float(ols.score(cc[DRIVERS], cc[TARGET]))}
print("Discovered formula:", {k: round(v, 5) for k, v in formula["coef"].items()},
      f"R2={formula['r2']:.4f}")

# ============================================================
# STEP 9: Save artifacts for the Streamlit app
# ============================================================
joblib.dump(best, BASE / "model.joblib")
joblib.dump(fitted["Gradient Boosting"], BASE / "model_gb.joblib")   # 2nd opinion in app

pd.DataFrame({"actual": y_test.values,
              "predicted": best.predict(X_test),
              "predicted_gb": fitted["Gradient Boosting"].predict(X_test),
              "has_missing": (~complete).values}).sample(5000, random_state=SEED) \
  .to_csv(BASE / "test_predictions.csv", index=False)

artifacts = {
    "best_model": best_name,
    "results": res.to_dict(orient="records"),
    "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std()),
    "r2_complete_rows": float(r2_complete),
    "permutation_importance": importance.round(5).to_dict(),
    "lr_coefficients_raw_units": raw_coef,
    "lr_intercept": float(lr.named_steps["model"].intercept_),
    "formula": formula,
    "features_num": NUM, "features_cat": CAT,
    "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT},
    "rows_raw": 122400, "rows_after_dedup": int(len(df)),
    "missing_counts": {k: int(v) for k, v in
                       pd.read_csv(DATA).isnull().sum().items() if v > 0},
    "views_iqr_outliers": int(n_out),
    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
}
(BASE / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
print("\n✅ Saved model.joblib, model_gb.joblib, artifacts.json, test_predictions.csv, figures/")
