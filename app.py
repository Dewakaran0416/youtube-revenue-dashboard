# ============================================================
# app.py — YouTube Ad Revenue: Predictor + Interactive Dashboard
# 1) python main.py      (trains & saves model.joblib + artifacts.json)
# 2) streamlit run app.py
# ============================================================
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
TARGET = "ad_revenue_usd"
ACCENT = "#E62117"
# Streamlit >=1.50 uses width='stretch'; older versions use use_container_width
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="YouTube Ad Revenue Studio", page_icon="📺", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  div[data-testid="stMetric"] {background: rgba(230,33,23,0.06); border: 1px solid rgba(230,33,23,0.18);
      border-radius: 12px; padding: 12px 16px;}
  .big-pred {font-size: 2.6rem; font-weight: 800; color: #E62117; line-height: 1.1;}
  .muted {opacity: .7; font-size: .9rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Loading (cached)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    missing = [f for f in ["model.joblib", "artifacts.json"] if not (BASE / f).exists()]
    if missing:
        return None, None, None
    best = joblib.load(BASE / "model.joblib")
    gb = joblib.load(BASE / "model_gb.joblib") if (BASE / "model_gb.joblib").exists() else None
    art = json.loads((BASE / "artifacts.json").read_text())
    return best, gb, art


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "youtube_ad_revenue_dataset.csv").drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["rpm"] = df[TARGET] / df["views"] * 1000        # revenue per 1,000 views
    return df


@st.cache_data
def load_preds():
    p = BASE / "test_predictions.csv"
    return pd.read_csv(p) if p.exists() else None


best_model, gb_model, ART = load_models()
if best_model is None:
    st.error("Model files not found. Run `python main.py` first, then reload this page.")
    st.stop()
df = load_data()
preds = load_preds()


def build_row(views, likes, comments, watch, length, subs, category, device, country, pub_date):
    """Same feature engineering as main.py -> one-row DataFrame for the pipeline."""
    pub = pd.Timestamp(pub_date)
    return pd.DataFrame([{
        "views": views, "likes": likes, "comments": comments,
        "watch_time_minutes": watch, "video_length_minutes": length, "subscribers": subs,
        "engagement_rate": (likes + comments) / (views + 1),
        "watch_efficiency": watch / (length + 1),
        "month": pub.month, "day_of_week": pub.dayofweek,
        "category": category, "device": device, "country": country,
    }])


# ------------------------------------------------------------
# Sidebar — video inputs (used by Predictor + What-if)
# ------------------------------------------------------------
cats = ART["categories"]
with st.sidebar:
    st.header("🎬 Video details")
    views = st.number_input("Views", 0, 1_000_000, 10_000, step=100)
    likes = st.number_input("Likes", 0, 500_000, 1_100, step=10)
    comments = st.number_input("Comments", 0, 200_000, 275, step=5)
    watch = st.number_input("Watch time (minutes, total)", 0.0, 5_000_000.0, 37_500.0, step=500.0)
    length = st.slider("Video length (minutes)", 1.0, 60.0, 16.0, 0.5)
    subs = st.number_input("Channel subscribers", 0, 10_000_000, 500_000, step=1_000)
    category = st.selectbox("Category", cats["category"])
    device = st.selectbox("Main device", cats["device"])
    country = st.selectbox("Country", cats["country"])
    pub_date = st.date_input("Publish date", date(2025, 1, 15))
    st.caption("Defaults ≈ dataset averages.")

row = build_row(views, likes, comments, watch, length, subs, category, device, country, pub_date)
pred = float(best_model.predict(row)[0])
pred_gb = float(gb_model.predict(row)[0]) if gb_model is not None else None

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("📺 YouTube Ad Revenue Studio")
st.caption(f"Best model: **{ART['best_model']}** · Test R² {ART['results'][0]['R2']:.3f} · "
           f"5-fold CV R² {ART['cv_r2_mean']:.3f} ± {ART['cv_r2_std']:.3f} · "
           f"{ART['rows_after_dedup']:,} videos")

tab_pred, tab_eda, tab_model, tab_whatif, tab_about = st.tabs(
    ["🔮 Predict", "📊 EDA Dashboard", "🏆 Model Performance", "🧪 What-if", "ℹ️ Method"])

# ============================================================
# TAB 1 — Predict
# ============================================================
with tab_pred:
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.markdown("**Estimated ad revenue**")
        st.markdown(f'<div class="big-pred">${pred:,.2f}</div>', unsafe_allow_html=True)
        pct = (df[TARGET] < pred).mean() * 100
        st.markdown(f'<div class="muted">Higher than {pct:.0f}% of videos in the dataset</div>',
                    unsafe_allow_html=True)
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("RPM ($/1k views)", f"{pred / max(views, 1) * 1000:,.2f}")
        m2.metric("Engagement rate", f"{(likes + comments) / (views + 1):.1%}")
        m3.metric("Watch / view", f"{watch / max(views, 1):.1f}m")
        if pred_gb is not None:
            st.info(f"Second opinion — Gradient Boosting: **\\${pred_gb:,.2f}** "
                    f"(difference \\${abs(pred - pred_gb):,.2f})")
        if watch / max(views, 1) > length:
            st.warning("Average watch time per view is longer than the video itself — check inputs.")
        if views > df["views"].max() * 1.5 or views < df["views"].min() * 0.5:
            st.warning(f"Views are outside the training range "
                       f"({df['views'].min():,}–{df['views'].max():,}); prediction is an extrapolation.")

    with c2:
        f = ART["formula"]
        parts = {"Watch time": f["coef"]["watch_time_minutes"] * watch,
                 "Views": f["coef"]["views"] * views,
                 "Likes": f["coef"]["likes"] * likes,
                 "Comments": f["coef"]["comments"] * comments}
        fig = go.Figure(go.Waterfall(
            x=list(parts) + ["Total"], y=list(parts.values()) + [0],
            measure=["relative"] * len(parts) + ["total"],
            text=[f"${v:,.0f}" for v in parts.values()] + [f"${sum(parts.values()):,.0f}"],
            textposition="outside", connector={"line": {"color": "#bbb"}},
            increasing={"marker": {"color": ACCENT}}, totals={"marker": {"color": "#333"}}))
        fig.update_layout(title="Where the revenue comes from (driver breakdown)",
                          height=380, margin=dict(t=50, b=10), yaxis_title="USD",
                          yaxis_range=[0, sum(parts.values()) * 1.15])
        st.plotly_chart(fig, **FULL)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pred, number={"prefix": "$"},
        gauge={"axis": {"range": [0, max(df[TARGET].max(), pred) * 1.05]},
               "bar": {"color": ACCENT},
               "steps": [{"range": [0, df[TARGET].quantile(.25)], "color": "#f3f3f3"},
                         {"range": [df[TARGET].quantile(.25), df[TARGET].quantile(.75)], "color": "#e4e4e4"}]},
        title={"text": "Prediction vs dataset (shaded = middle 50%)"}))
    gauge.update_layout(height=260, margin=dict(t=60, b=0))
    st.plotly_chart(gauge, **FULL)
    with st.expander("Model input row"):
        st.dataframe(row, **FULL)

# ============================================================
# TAB 2 — EDA dashboard (filterable)
# ============================================================
with tab_eda:
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.4])
    sel_cat = f1.multiselect("Category", cats["category"], default=cats["category"])
    sel_dev = f2.multiselect("Device", cats["device"], default=cats["device"])
    sel_cty = f3.multiselect("Country", cats["country"], default=cats["country"])
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    rng = f4.slider("Date range", dmin, dmax, (dmin, dmax))

    d = df[df["category"].isin(sel_cat) & df["device"].isin(sel_dev) & df["country"].isin(sel_cty)
           & df["date"].dt.date.between(*rng)]
    if d.empty:
        st.warning("No rows match these filters.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Videos", f"{len(d):,}")
        k2.metric("Total revenue", f"${d[TARGET].sum() / 1e6:,.2f}M")
        k3.metric("Avg revenue", f"${d[TARGET].mean():,.2f}")
        k4.metric("Avg RPM", f"${d['rpm'].mean():,.2f}")
        k5.metric("Avg watch time", f"{d['watch_time_minutes'].mean() / 1000:,.1f}k min")

        a, b = st.columns(2)
        a.plotly_chart(px.histogram(d, x=TARGET, nbins=50, color_discrete_sequence=[ACCENT],
                                    title="Revenue distribution"), **FULL)
        trend = d.groupby("month", as_index=False)[TARGET].mean()
        b.plotly_chart(px.line(trend, x="month", y=TARGET, markers=True,
                               title="Average revenue by month",
                               color_discrete_sequence=[ACCENT]).update_yaxes(rangemode="tozero"),
                       **FULL)

        group_by = st.radio("Compare by", ["category", "device", "country"], horizontal=True)
        g = d.groupby(group_by).agg(avg_revenue=(TARGET, "mean"), videos=(TARGET, "size"),
                                    total=(TARGET, "sum")).reset_index().sort_values("avg_revenue")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(g, x="avg_revenue", y=group_by, orientation="h", text_auto=".1f",
                              title=f"Average revenue by {group_by}",
                              color_discrete_sequence=[ACCENT]), **FULL)
        b.plotly_chart(px.pie(g, names=group_by, values="total", hole=.5,
                              title=f"Revenue share by {group_by}"), **FULL)

        a, b = st.columns(2)
        num_cols = ["views", "likes", "comments", "watch_time_minutes",
                    "video_length_minutes", "subscribers"]
        xcol = a.selectbox("Scatter: X axis", num_cols, index=3)
        samp = d.sample(min(len(d), 4000), random_state=1)
        a.plotly_chart(px.scatter(samp, x=xcol, y=TARGET, color=group_by, opacity=.5,
                                  title=f"{xcol} vs revenue (4k sample)"), **FULL)
        corr = d[num_cols + [TARGET]].corr().round(2)
        b.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Correlation heatmap"),
                       **FULL)

        miss = pd.Series(ART["missing_counts"]).rename("missing").reset_index()
        miss.columns = ["column", "missing"]
        miss["pct"] = miss["missing"] / ART["rows_raw"] * 100
        st.plotly_chart(px.bar(miss, x="column", y="pct", text_auto=".1f",
                               title="Missing values in raw data (%) — imputed with median inside the pipeline",
                               color_discrete_sequence=["#888"]), **FULL)

# ============================================================
# TAB 3 — Model performance
# ============================================================
with tab_model:
    res = pd.DataFrame(ART["results"])
    st.subheader("5 models compared on the same 20% hold-out test set")
    st.dataframe(res.style.highlight_max(subset=["R2", "Adj_R2"], color="#d4f4dd")
                 .highlight_min(subset=["RMSE", "MAE"], color="#d4f4dd")
                 .format({"R2": "{:.4f}", "Adj_R2": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "Train_sec": "{:.1f}"}),
                 **FULL, hide_index=True)
    metric = st.radio("Metric", ["R2", "RMSE", "MAE"], horizontal=True)
    st.plotly_chart(px.bar(res, x="Model", y=metric, text_auto=".3f",
                           color="Model", title=f"{metric} by model"), **FULL)

    a, b, c = st.columns(3)
    a.metric("Test R² (all rows)", f"{res.iloc[0]['R2']:.4f}")
    b.metric("Test R² (rows with no missing inputs)", f"{ART['r2_complete_rows']:.4f}")
    c.metric("5-fold CV R²", f"{ART['cv_r2_mean']:.4f}", f"± {ART['cv_r2_std']:.4f}", delta_color="off")
    st.caption("The remaining ~5% error comes almost entirely from rows where likes, comments or "
               "watch time were missing and had to be imputed with the median.")

    if preds is not None:
        a, b = st.columns(2)
        preds["error"] = preds["predicted"] - preds["actual"]
        preds["row type"] = np.where(preds["has_missing"], "had missing input", "complete")
        fig = px.scatter(preds, x="actual", y="predicted", color="row type", opacity=.5,
                         title="Actual vs predicted (5k test rows)",
                         color_discrete_map={"complete": "#2e7d32", "had missing input": ACCENT})
        lo, hi = preds["actual"].min(), preds["actual"].max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#555"))
        a.plotly_chart(fig, **FULL)
        b.plotly_chart(px.histogram(preds, x="error", color="row type", nbins=60, barmode="overlay",
                                    title="Residuals (predicted − actual)",
                                    color_discrete_map={"complete": "#2e7d32", "had missing input": ACCENT}),
                       **FULL)

    a, b = st.columns(2)
    imp = pd.Series(ART["permutation_importance"]).sort_values()
    a.plotly_chart(px.bar(imp, orientation="h", title="Permutation importance (drop in R²)",
                          color_discrete_sequence=["teal"]).update_layout(showlegend=False),
                   **FULL)
    fcoef = pd.Series(ART["formula"]["coef"]).rename("USD per unit").to_frame()
    b.markdown("**Discovered revenue formula** (OLS on complete rows)")
    b.dataframe(fcoef.style.format("{:.5f}"), **FULL)
    b.latex(r"\text{revenue} \approx " + " + ".join(
        f"{v:.5f}\\cdot\\text{{{k.replace('_', ' ')}}}" for k, v in ART["formula"]["coef"].items()))
    b.caption(f"R² of this 4-variable formula = {ART['formula']['r2']:.4f} → the dataset is synthetic "
              "and revenue is a linear function of these drivers. That is why Linear Regression wins.")

# ============================================================
# TAB 4 — What-if analysis
# ============================================================
with tab_whatif:
    st.subheader("Sensitivity: change one input, keep the rest from the sidebar")
    feat = st.selectbox("Input to vary", ["watch_time_minutes", "views", "likes", "comments",
                                          "subscribers", "video_length_minutes"])
    base_vals = dict(views=views, likes=likes, comments=comments, watch=watch,
                     length=length, subs=subs)
    key = {"watch_time_minutes": "watch", "views": "views", "likes": "likes",
           "comments": "comments", "subscribers": "subs", "video_length_minutes": "length"}[feat]
    lo, hi = float(df[feat].quantile(.01)), float(df[feat].quantile(.99))
    grid = np.linspace(lo, hi, 40)
    rows = []
    for v in grid:
        kv = dict(base_vals); kv[key] = v
        rows.append(build_row(kv["views"], kv["likes"], kv["comments"], kv["watch"], kv["length"],
                              kv["subs"], category, device, country, pub_date))
    batch = pd.concat(rows, ignore_index=True)
    out = pd.DataFrame({feat: grid, "Linear Regression": best_model.predict(batch)})
    if gb_model is not None:
        out["Gradient Boosting"] = gb_model.predict(batch)
    fig = px.line(out.melt(id_vars=feat, var_name="model", value_name="predicted revenue"),
                  x=feat, y="predicted revenue", color="model",
                  title=f"Predicted revenue as {feat} changes",
                  color_discrete_map={"Linear Regression": ACCENT, "Gradient Boosting": "#1f77b4"})
    fig.add_vline(x=base_vals[key], line_dash="dot", annotation_text="your value")
    st.plotly_chart(fig, **FULL)
    slope = (out["Linear Regression"].iloc[-1] - out["Linear Regression"].iloc[0]) / (hi - lo)
    st.info(f"Each extra unit of **{feat}** changes predicted revenue by about **\\${slope:,.5f}** "
            f"(≈ \\${slope * (hi - lo) / 10:,.2f} per 10% of its typical range).")

    st.subheader("Scenario planner")
    s1, s2, s3 = st.columns(3)
    up_watch = s1.slider("Change watch time (%)", -50, 100, 20)
    up_views = s2.slider("Change views (%)", -50, 100, 0)
    up_eng = s3.slider("Change likes & comments (%)", -50, 100, 0)
    scen = build_row(views * (1 + up_views / 100), likes * (1 + up_eng / 100),
                     comments * (1 + up_eng / 100), watch * (1 + up_watch / 100),
                     length, subs, category, device, country, pub_date)
    new_pred = float(best_model.predict(scen)[0])
    st.metric("Scenario revenue", f"${new_pred:,.2f}", f"{new_pred - pred:+,.2f} vs current")

# ============================================================
# TAB 5 — Method
# ============================================================
with tab_about:
    st.markdown(f"""
### Pipeline
**Raw CSV** ({ART['rows_raw']:,} rows) → drop {ART['rows_raw'] - ART['rows_after_dedup']:,} duplicates →
**train/test split 80/20** → inside a scikit-learn `Pipeline`:
median imputation + `StandardScaler` (numeric) · mode imputation + `OneHotEncoder` (category, device, country) → model.

### Key decisions
- **Imputation inside the pipeline**, fitted on the training split only → no test-set leakage.
- **Removed `revenue_per_view`** — it is computed from the target (target leakage).
- **Outliers kept** — only {ART['views_iqr_outliers']} rows ({ART['views_iqr_outliers'] / ART['rows_after_dedup']:.1%}) in `views` are IQR outliers and they are valid values.
- **One-hot encoding** for nominal categoricals (6 categories, 4 devices, 6 countries) instead of label encoding, which would impose a fake order.
- **Engineered features:** engagement rate, watch efficiency, month, day of week.

### Findings
- Watch time is the dominant driver (correlation 0.99 with revenue); likes and comments add a little; category, device, country, subscribers and date have ~no effect.
- Revenue follows an almost exact linear formula, so **Linear Regression** is both the most accurate and the fastest model, and it is fully interpretable.
- Remaining error is caused by missing inputs, not model choice: on complete rows R² = {ART['r2_complete_rows']:.4f}.
""")
