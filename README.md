# 📺 YouTube Ad Revenue Prediction

Predict the ad revenue (USD) of a YouTube video from its performance and context metrics, and explore the data through an interactive Streamlit dashboard.

Capstone project · GUVI Master Data Science Program

![Predict tab](assets/app_predict.png)

---

## Highlights

- **Leak-free scikit-learn Pipeline**: imputation, scaling and one-hot encoding are fitted on the training split only.
- **5 regression models compared** on the same hold-out set, with 5-fold cross-validation for the winner.
- **Linear Regression wins** with R² = 0.953 and MAE = $3.11, and it is also the fastest and most interpretable model.
- **Interactive Streamlit app** with five tabs: live prediction, EDA dashboard, model performance, what-if analysis and methodology.

## Dataset

| Item | Value |
|---|---|
| Rows | 122,400 (120,000 after removing 2,400 duplicates) |
| Columns | `video_id`, `date`, `views`, `likes`, `comments`, `watch_time_minutes`, `video_length_minutes`, `subscribers`, `category`, `device`, `country`, `ad_revenue_usd` |
| Target | `ad_revenue_usd` (continuous → regression) |
| Missing values | ~5% each in `likes`, `comments`, `watch_time_minutes` |
| Period | June 2024 – June 2025 |
| Categories | 6 categories, 4 devices, 6 countries |

## Approach

```
Raw CSV → drop duplicates → 80/20 train/test split → Pipeline:
    numeric:     median imputation → StandardScaler
    categorical: mode imputation   → OneHotEncoder
→ model → evaluate (R², Adj. R², RMSE, MAE, 5-fold CV) → save with joblib → Streamlit app
```

**Feature engineering:** engagement rate `(likes + comments) / views`, watch efficiency `watch_time / video_length`, month and day of week from the publish date.

**Key decisions**

- **Dropped `revenue_per_view`**: it is computed from the target (`ad_revenue_usd / views`), which is target leakage, and it cannot be known at prediction time.
- **Imputation inside the Pipeline** so test data never influences the training statistics.
- **Outliers kept**: only 865 rows (0.7%) in `views` fall outside the IQR fences, and they are valid values rather than errors.
- **One-hot encoding** for nominal categories. Label encoding would impose a false order.

## Exploratory analysis

| Revenue distribution | Correlation heatmap |
|---|---|
| ![](figures/01_target_distribution.png) | ![](figures/02_correlation.png) |

| Watch time vs revenue | Revenue by category |
|---|---|
| ![](figures/04_watchtime_vs_revenue.png) | ![](figures/03_revenue_by_category.png) |

- Watch time correlates **0.99** with revenue.
- Average revenue is about **$252** in every category, device and country, so these factors have almost no effect.

## Results

| Model | R² | Adj. R² | RMSE ($) | MAE ($) |
|---|---|---|---|---|
| **Linear Regression** | **0.9526** | **0.9526** | **13.48** | **3.11** |
| Gradient Boosting | 0.9522 | 0.9522 | 13.53 | 3.69 |
| Random Forest | 0.9514 | 0.9514 | 13.65 | 3.49 |
| Decision Tree | 0.9473 | 0.9473 | 14.21 | 4.21 |
| K-Nearest Neighbors | 0.8820 | 0.8820 | 21.26 | 15.39 |

5-fold cross-validated R² for Linear Regression: **0.950 ± 0.001**.

![Model performance](assets/app_model.png)

### Why Linear Regression wins

An ordinary least squares fit on rows with no missing values gives R² = 1.000:

```
revenue ≈ 0.00521·views + 0.01719·(likes + comments) + 0.00471·watch_time_minutes
```

Revenue in this (synthetic) dataset is a linear function of these drivers. When the true relationship is linear, a linear model is the best fit, while tree models can only approximate a straight line with steps.

### Where the remaining error comes from

On test rows with complete inputs, R² is **1.000**. All of the remaining ~5% error comes from rows where likes, comments or watch time were missing and had to be imputed with the median. Data completeness, not model choice, is the limiting factor.

![Feature importance](figures/05_feature_importance.png)

## Streamlit app

| Tab | What it shows |
|---|---|
| 🔮 Predict | Live revenue estimate, driver breakdown (waterfall), position vs dataset, Gradient Boosting second opinion |
| 📊 EDA Dashboard | Filters by category, device, country and date; KPI cards; distribution; monthly trend; group comparisons; scatter; correlation heatmap |
| 🏆 Model Performance | Metrics table, actual vs predicted, residuals, permutation importance, discovered formula |
| 🧪 What-if | Sensitivity curve for any input and a scenario planner |
| ℹ️ Method | Pipeline, decisions and findings |

![EDA dashboard](assets/app_eda.png)

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/youtube-ad-revenue-prediction.git
cd youtube-ad-revenue-prediction

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

python -m pip install -r requirements.txt
python main.py                   # trains models, saves figures and model files (~2 min)
streamlit run app.py
```

Python 3.11–3.13 recommended.

## Project structure

```
├── main.py                  # EDA, cleaning, feature engineering, training, evaluation
├── app.py                   # Streamlit app
├── requirements.txt
├── youtube_ad_revenue_dataset.csv
├── model.joblib             # best pipeline (Linear Regression)
├── model_gb.joblib          # Gradient Boosting pipeline (second opinion in app)
├── artifacts.json           # metrics, importance, formula, metadata
├── test_predictions.csv     # sample of test predictions for app charts
├── figures/                 # EDA and importance charts from main.py
└── assets/                  # app screenshots for this README
```

## Limitations and next steps

- The dataset is synthetic, so real-world effects such as country-specific ad rates, niche and seasonality are absent.
- Next: real YouTube Analytics data, a time-based train/test split, hyperparameter tuning, and deployment on Streamlit Community Cloud.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · joblib

## Author

**Dewakaran (Dewa)** · Senior Lead, Operations · 10+ years in US healthcare revenue cycle management, building data science and analytics skills.
