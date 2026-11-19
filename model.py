"""
MST0052 - Regional construction activity: modelling with time-series
cross-validation.

Predicts total floor area with commencement permits (m2) over the next
3 months, per region.

PROTOCOL (fixed before the test set is used)
  1) Target: sum of floor area in months t+1, t+2, t+3. All models are trained
     on log(1 + target) and predict back in m2 (TransformedTargetRegressor).
  2) The test set is locked: rows from 2023-01. Used once, at the very end.
  3) Cross-validation on 2015-2022, 8 folds with an expanding training window:
       fold for year Y: train on rows up to Sept of year Y-1, validate on year Y.
     The 3-month gap prevents target periods from overlapping. Validation
     stops in Sept 2022, so no target reaches into the test period.
  4) Model families (lectures 4, 10, 12):
       Linear: OLS (simple linear baseline), ridge, lasso
       Trees:  random forest, gradient boosting (HistGradientBoosting with
               early stopping)
     Hyperparameters: GridSearchCV over the same 8 folds, metric MAE (m2).
  5) Naive baselines on the same folds: same 3 months last year, last 3 months.
  6) Model selection (one-standard-error rule, lecture 7):
       threshold = lowest CV mean + 1 SE (SE = std over folds / sqrt(8)).
       Among the models within the threshold, the simplest is selected:
       OLS < ridge = lasso < random forest = gradient boosting.
       Equally simple models are separated by the lowest CV mean.
  7) The selected configuration is refitted on all rows up to Sept 2022 and
     tested once.

ADDED AFTER THE TEST SET HAD BEEN SEEN (reported, never used to select a model)
  8) PCA regression (lecture 9): standardise -> PCA -> OLS on the 11 numeric
     features. Number of components chosen on CV with the one-SE rule (fewest
     components within 1 SE of the best). With all 11 components it is OLS
     (same predictions up to numerical precision).
     It is a supplementary model and is not eligible in step 6.
  9) 90 % prediction intervals for the main model (split conformal, beyond the
     syllabus): absolute log residuals from the 8 CV validation years give the
     interval width; coverage is checked on the test set.

Run:     python model.py      (takes a few minutes)
Output:  results in the terminal, figures fig5-fig10, cv_summary.csv,
         cv_by_year.csv, test_predictions.csv
"""
import sys
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 42
np.random.seed(SEED)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9890"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10,
})

print("Versions: Python {} | numpy {} | pandas {} | scikit-learn {} | matplotlib {}".format(
    sys.version.split()[0], np.__version__, pd.__version__, sklearn.__version__,
    matplotlib.__version__))


# ================================================================ 1. Data and target
df = pd.read_csv("panel_region_month.csv")
df["date"] = pd.to_datetime(df["period"])
df = df.sort_values(["region", "date"]).reset_index(drop=True)
g = df.groupby("region")

nxt = g["area_next_month"]
df["target_3m"] = df["area_next_month"] + nxt.shift(-1) + nxt.shift(-2)

sum3 = g["area_m2"].transform(lambda s: s.rolling(3).sum())
df["baseline_last_year"] = sum3.groupby(df["region"]).shift(9)   # months t-11..t-9
df["baseline_last3"] = sum3                                       # months t-2..t

AREA = ["area_m2", "area_lag1", "area_lag3", "area_lag6", "area_lag12", "area_avg3"]
for k in AREA:
    df[f"log_{k}"] = np.log1p(df[k].clip(lower=0))   # stateless, no leakage

NUMERIC = [f"log_{k}" for k in AREA] + [
    "policy_rate", "rate_lag3", "rate_lag6", "rate_lag12", "pop_growth_12m"]
CATEGORICAL = ["region", "month"]
FEATURES = NUMERIC + CATEGORICAL

df = df.dropna(subset=["target_3m", "baseline_last_year"]).reset_index(drop=True)
dev = df[df["date"] <= "2022-09-01"].reset_index(drop=True)      # training + CV
test = df[df["date"] >= "2023-01-01"].reset_index(drop=True)     # locked


# ================================================================ 2. Folds
CV_YEARS = list(range(2015, 2023))
folds = []
for year in CV_YEARS:
    train_idx = np.where(dev["date"] <= f"{year - 1}-09-01")[0]
    val_idx = np.where(dev["date"].dt.year == year)[0]
    folds.append((train_idx, val_idx))
K = len(folds)

print("\n" + "=" * 76)
print("DATA AND FOLDS")
print("=" * 76)
print(f"  Development (training + CV): {dev['period'].min()} - "
      f"{dev['period'].max()}  ({len(dev)} rows)")
print(f"  Test (locked):               {test['period'].min()} - "
      f"{test['period'].max()}  ({len(test)} rows)")
print(f"  {K} folds, expanding training window:")
for year, (ti, vi) in zip(CV_YEARS, folds):
    p_t, p_v = dev.loc[ti, "period"], dev.loc[vi, "period"]
    print(f"    {year}: train {p_t.min()} - {p_t.max()} ({len(ti):>4} rows) | "
          f"validate {p_v.min()} - {p_v.max()} ({len(vi)} rows)")


# ================================================================ 3. Models
def preprocessing(scale, pca=False):
    """Linear models: standardisation (L4). Trees: no scaling (L3, L10).
    pca=True: standardise, then PCA on the numeric features (L9). Scaling comes
    first because PCA is driven by variance. Region and month stay one-hot."""
    num = StandardScaler() if scale else "passthrough"
    if pca:
        num = Pipeline([("scale", StandardScaler()), ("pca", PCA())])
    return ColumnTransformer(
        [("num", num, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        sparse_threshold=0)


def make(model, scale, pca=False):
    """Pipeline with log-transformed target: trains on log(1 + y), predicts in m2."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessing(scale, pca)), ("model", model)]),
        func=np.log1p, inverse_func=np.expm1, check_inverse=False)


P = "regressor__model__"
PCA_PARAM = "regressor__prep__num__pca__n_components"
MODELS = {
    # name: (estimator, grid, complexity for the one-SE rule)
    "OLS": (make(LinearRegression(), True), [{}], 0),
    "Ridge": (make(Ridge(), True),
              {P + "alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]}, 1),
    "Lasso": (make(Lasso(max_iter=20000), True),
              {P + "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1]}, 1),
    "Random forest": (make(RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                                 random_state=SEED), False),
                      {P + "max_features": ["sqrt", 0.33, 1.0],
                       P + "min_samples_leaf": [1, 5, 20]}, 2),
    "Gradient boosting": (make(HistGradientBoostingRegressor(
                              max_iter=2000, early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=20, random_state=SEED), False),
                          {P + "learning_rate": [0.02, 0.05, 0.1, 0.2],
                           P + "max_depth": [3, 4, 6]}, 2),
    # Supplementary (added after the test set was seen): complexity None = not
    # eligible for model selection.
    "PCA regression": (make(LinearRegression(), True, pca=True),
                       {PCA_PARAM: list(range(1, len(NUMERIC) + 1))}, None),
}
MODEL_NAMES = list(MODELS)
SELECTABLE = [n for n in MODEL_NAMES if MODELS[n][2] is not None]
BASELINE_NAMES = ["Baseline: same 3 months last year", "Baseline: last 3 months"]
BASELINE_COL = {"Baseline: same 3 months last year": "baseline_last_year",
                "Baseline: last 3 months": "baseline_last3"}

X_dev, y_dev = dev[FEATURES], dev["target_3m"]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def short(params):
    return {k.split("__")[-1]: v for k, v in params.items()}


# ================================================================ 4. Cross-validation
print("\n" + "=" * 76)
print("CROSS-VALIDATION 2015-2022 (MAE in m2, lower is better)")
print("=" * 76)

per_fold, train_cv, best_params, cv_res = {}, {}, {}, {}
for name, (model, grid, _) in MODELS.items():
    start = time.time()
    search = GridSearchCV(model, grid, cv=folds, scoring="neg_mean_absolute_error",
                          refit=False, n_jobs=-1, return_train_score=True)
    search.fit(X_dev, y_dev)
    res = pd.DataFrame(search.cv_results_)
    b = search.best_index_
    cv_res[name] = res
    best_params[name] = search.best_params_
    per_fold[name] = np.array([-res.loc[b, f"split{i}_test_score"] for i in range(K)])
    train_cv[name] = -res.loc[b, "mean_train_score"]

    print(f"\n  {name}: {len(res)} candidate(s) x {K} folds ({time.time() - start:.0f} s)")
    for _, r in res.sort_values("rank_test_score").head(5).iterrows():
        print(f"    {str(short(r['params'])):<58} {-r['mean_test_score']:>8,.0f} "
              f"± {r['std_test_score']:>6,.0f}")
    if isinstance(grid, dict):
        for param, values in grid.items():
            v = search.best_params_[param]
            if len(values) > 1 and v in (values[0], values[-1]):
                print(f"    NOTE: {param.split('__')[-1]}={v} is at the edge of the grid")

# PCA regression: number of components by the one-SE rule (as in L9), not by
# the lowest CV mean. Fewer components = simpler model.
res = cv_res["PCA regression"].sort_values(f"param_{PCA_PARAM}").reset_index(drop=True)
pca_folds = -res[[f"split{i}_test_score" for i in range(K)]].to_numpy()
res["MAE mean"] = pca_folds.mean(axis=1)
res["SE"] = pca_folds.std(axis=1, ddof=1) / np.sqrt(K)
b = res["MAE mean"].idxmin()
pca_threshold = res.loc[b, "MAE mean"] + res.loc[b, "SE"]
c = res.index[res["MAE mean"] <= pca_threshold][0]
pca_cv = res
best_params["PCA regression"] = res.loc[c, "params"]
per_fold["PCA regression"] = pca_folds[c]
train_cv["PCA regression"] = -res.loc[c, "mean_train_score"]
print(f"\n  PCA regression, one-SE rule: lowest CV mean with "
      f"{res.loc[b, f'param_{PCA_PARAM}']} components ({res.loc[b, 'MAE mean']:,.0f}, "
      f"SE {res.loc[b, 'SE']:,.0f}) -> threshold {pca_threshold:,.0f} -> "
      f"{res.loc[c, f'param_{PCA_PARAM}']} components selected")

for name in BASELINE_NAMES:
    col = BASELINE_COL[name]
    per_fold[name] = np.array([mae(y_dev.iloc[vi], dev[col].iloc[vi]) for _, vi in folds])

fold_mean = np.array([y_dev.iloc[vi].mean() for _, vi in folds])
ALL = BASELINE_NAMES + MODEL_NAMES
cv_tab = pd.DataFrame({
    "MAE mean": [per_fold[n].mean() for n in ALL],
    "MAE std": [per_fold[n].std(ddof=1) for n in ALL],
    "Rel. error mean": [(per_fold[n] / fold_mean).mean() for n in ALL],
    "Rel. error std": [(per_fold[n] / fold_mean).std(ddof=1) for n in ALL],
}, index=ALL)
cv_tab["SE"] = cv_tab["MAE std"] / np.sqrt(K)

print(f"\n  Comparison (mean ± standard deviation over {K} folds):")
for n, r in cv_tab.iterrows():
    print(f"    {n:<34} MAE {r['MAE mean']:>8,.0f} ± {r['MAE std']:>6,.0f}   "
          f"rel. error {r['Rel. error mean']:.1%} ± {r['Rel. error std']:.1%}")

fold_tab = pd.DataFrame({n: per_fold[n] for n in ALL}, index=CV_YEARS)
print("\n  MAE per validation year:")
print(fold_tab.round(0).astype(int).to_string())
best_baseline = np.minimum(per_fold[BASELINE_NAMES[0]], per_fold[BASELINE_NAMES[1]])
for n in MODEL_NAMES:
    print(f"  {n:<18} beats both baselines in {(per_fold[n] < best_baseline).sum()} of {K} years, "
          f"'last 3 months' in {(per_fold[n] < per_fold[BASELINE_NAMES[1]]).sum()} of {K}.")


# ================================================================ 5. Model selection (one-SE)
print("\n" + "=" * 76)
print("MODEL SELECTION: one-standard-error rule")
print("=" * 76)
m_cv = cv_tab.loc[SELECTABLE].copy()
m_cv["complexity"] = [MODELS[n][2] for n in SELECTABLE]
best = m_cv["MAE mean"].idxmin()
threshold = m_cv.loc[best, "MAE mean"] + m_cv.loc[best, "SE"]
within = m_cv[m_cv["MAE mean"] <= threshold].sort_values(["complexity", "MAE mean"])
main = within.index[0]
print(f"  Lowest CV mean: {best} ({m_cv.loc[best, 'MAE mean']:,.0f}, "
      f"SE {m_cv.loc[best, 'SE']:,.0f}) -> threshold {threshold:,.0f}")
for n in SELECTABLE:
    inside = "within" if n in within.index else "outside"
    print(f"    {n:<18} {m_cv.loc[n, 'MAE mean']:>8,.0f}  ({inside}, complexity "
          f"{m_cv.loc[n, 'complexity']})")
for n in MODEL_NAMES:
    if n not in SELECTABLE:
        side = "within" if cv_tab.loc[n, "MAE mean"] <= threshold else "outside"
        print(f"    {n:<18} {cv_tab.loc[n, 'MAE mean']:>8,.0f}  ({side}; supplementary, "
              f"not eligible)")
print(f"  Main model: {main} (simplest model within 1 SE)")
for n in MODEL_NAMES:
    print(f"    {n:<18} parameters: {short(best_params[n])}")

cv_tab.to_csv("cv_summary.csv")
fold_tab.to_csv("cv_by_year.csv", index_label="validation_year")


# ================================================================ 6. Test (once)
print("\n" + "=" * 76)
print("TEST 2023-2026: trained on all rows up to Sept 2022, evaluated once")
print("=" * 76)
final = {}
for name, (model, _, _) in MODELS.items():
    m = clone(model).set_params(**best_params[name])
    m.fit(X_dev, y_dev)
    final[name] = m
    test[f"pred_{name}"] = m.predict(test[FEATURES])
gb = final["Gradient boosting"].regressor_.named_steps["model"]
print(f"  Gradient boosting: early stopping used {gb.n_iter_} of {gb.max_iter} trees.")
if gb.n_iter_ >= gb.max_iter:
    print("  NOTE: early stopping did not trigger before the maximum. Consider a higher max_iter.")

COL = {**BASELINE_COL, **{n: f"pred_{n}" for n in MODEL_NAMES}}
mean_test = test["target_3m"].mean()
test_tab = pd.DataFrame({"MAE": [mae(test["target_3m"], test[COL[n]]) for n in ALL]}, index=ALL)
test_tab["Rel. error"] = test_tab["MAE"] / mean_test
test_tab["vs last year"] = 1 - test_tab["MAE"] / test_tab.loc[BASELINE_NAMES[0], "MAE"]
test_tab["vs last 3 months"] = 1 - test_tab["MAE"] / test_tab.loc[BASELINE_NAMES[1], "MAE"]
show = test_tab.copy()
show.index = [n + ("  <- main model" if n == main else "") for n in show.index]
print("  Improvement = reduction in MAE relative to the baseline (positive = better)")
print(show.to_string(formatters={"MAE": "{:,.0f}".format, "Rel. error": "{:.1%}".format,
                                 "vs last year": "{:+.0%}".format,
                                 "vs last 3 months": "{:+.0%}".format}))
print("\n  Mean bias in the test period (positive = overestimates):")
for n in ALL:
    bias = (test[COL[n]] - test["target_3m"]).mean() / mean_test
    print(f"    {n:<34} {bias:+.1%}")


# ================================================================ 6b. Prediction intervals (split conformal)
# Beyond the syllabus. Out-of-fold predictions from the 8 CV years give
# calibration residuals that the model did not train on. Score = |log(1+y) -
# log(1+pred)|, so the interval is multiplicative: pred x [e^-q, e^q]. This
# matches the log target and gives wider intervals for larger regions.
# Assumption: future residuals behave like the CV residuals (exchangeability).
# A time series can break this, so coverage on test is checked, not assumed.
ALPHA = 0.10
oof = np.full(len(dev), np.nan)
for ti, vi in folds:
    m = clone(MODELS[main][0]).set_params(**best_params[main])
    m.fit(X_dev.iloc[ti], y_dev.iloc[ti])
    oof[vi] = m.predict(X_dev.iloc[vi])
cal = ~np.isnan(oof)
scores = np.abs(np.log1p(y_dev[cal]) - np.log1p(np.clip(oof[cal], 0, None)))
n_cal = len(scores)
q = np.quantile(scores, min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal), method="higher")
log_pred = np.log1p(np.clip(test[COL[main]], 0, None))
test["lower"], test["upper"] = np.expm1(log_pred - q), np.expm1(log_pred + q)
y_t = test["target_3m"]
inside = (y_t >= test["lower"]) & (y_t <= test["upper"])
print(f"\n  {1 - ALPHA:.0%} prediction intervals ({main}, split conformal, {n_cal} calibration "
      f"rows from CV 2015-2022):")
print(f"    q = {q:.3f}  -> interval = prediction x [{np.exp(-q):.2f}, {np.exp(q):.2f}]")
print(f"    Test coverage: {inside.mean():.1%} (target {1 - ALPHA:.0%}).  Actual below lower "
      f"bound: {(y_t < test['lower']).mean():.1%}, above upper bound: "
      f"{(y_t > test['upper']).mean():.1%}")
cover_year = inside.groupby(test["date"].dt.year).agg(["mean", "size"])
for year, r in cover_year.iterrows():
    print(f"      {year}: coverage {r['mean']:.1%}  ({int(r['size'])} rows)")


# ================================================================ 7. Bias-variance: train vs CV vs test (L6)
print("\n  Training vs validation vs test, MAE in m2 (L6: large gap = overfitting):")
bv = pd.DataFrame({
    "Training (CV mean)": [train_cv[n] for n in MODEL_NAMES],
    "Validation (CV)": [cv_tab.loc[n, "MAE mean"] for n in MODEL_NAMES],
    "Test": [test_tab.loc[n, "MAE"] for n in MODEL_NAMES],
}, index=MODEL_NAMES)
bv["Validation / training"] = bv["Validation (CV)"] / bv["Training (CV mean)"]
print(bv.to_string(formatters={"Training (CV mean)": "{:,.0f}".format,
                               "Validation (CV)": "{:,.0f}".format, "Test": "{:,.0f}".format,
                               "Validation / training": "{:.2f}".format}))


# ================================================================ 8. By region
main_col = COL[main]
print(f"\n  Error by region (test, main model: {main}):")
rows = []
for r, d in test.groupby("region"):
    rows.append({"region": r, "mean_3m_m2": d["target_3m"].mean(),
                 "MAE_model": mae(d["target_3m"], d[main_col]),
                 "MAE_last_year": mae(d["target_3m"], d["baseline_last_year"]),
                 "MAE_last3": mae(d["target_3m"], d["baseline_last3"]),
                 "Bias": (d[main_col] - d["target_3m"]).mean() / d["target_3m"].mean()})
reg = pd.DataFrame(rows).set_index("region")
reg["Rel. error"] = reg["MAE_model"] / reg["mean_3m_m2"]
reg["Beats last year"] = np.where(reg["MAE_model"] < reg["MAE_last_year"], "yes", "no")
reg["Beats last 3"] = np.where(reg["MAE_model"] < reg["MAE_last3"], "yes", "no")
reg = reg.sort_values("Rel. error")
fmt = "{:,.0f}".format
print(reg.to_string(formatters={"mean_3m_m2": fmt, "MAE_model": fmt, "MAE_last_year": fmt,
                                "MAE_last3": fmt, "Rel. error": "{:.1%}".format,
                                "Bias": "{:+.1%}".format}))
print(f"  Beats 'same 3 months last year' in {(reg['Beats last year'] == 'yes').sum()} and "
      f"'last 3 months' in {(reg['Beats last 3'] == 'yes').sum()} of {len(reg)} regions.")


# ================================================================ 8b. By year
# Shows when in the test period the main model beats the naive baselines.
yr = test["date"].dt.year
by_year = pd.DataFrame({n: (test[COL[n]] - test["target_3m"]).abs().groupby(yr).mean()
                        for n in [main, *BASELINE_NAMES]})
by_year["Bias"] = ((test[main_col] - test["target_3m"]).groupby(yr).mean()
                   / test["target_3m"].groupby(yr).mean())
by_year["rows"] = yr.value_counts().sort_index()
print(f"\n  Error by year (test, MAE in m2; bias of {main}, positive = overestimates):")
print(by_year.to_string(formatters={**{n: fmt for n in [main, *BASELINE_NAMES]},
                                    "Bias": "{:+.1%}".format}))


# ================================================================ 8c. Residuals by forecast size (L4)
# Residual = actual - predicted (as in fig 8), so negative = overestimate.
res_t = test["target_3m"] - test[main_col]
print(f"\n  Residuals (test, {main}): forecast above actual in {(res_t < 0).mean():.1%} of rows")
size = pd.cut(test[main_col], [0, 100_000, 200_000, 300_000, np.inf],
              labels=["< 100k", "100-200k", "200-300k", "> 300k"])
by_size = pd.DataFrame({"rows": size.value_counts(sort=False),
                        "Mean residual": res_t.groupby(size, observed=False).mean(),
                        "Bias": -res_t.groupby(size, observed=False).mean()
                                / test["target_3m"].groupby(size, observed=False).mean()})
print("  By forecast size (m2):")
print(by_size.to_string(formatters={"Mean residual": fmt, "Bias": "{:+.1%}".format}))
for s, d in test.groupby(size, observed=False):
    print(f"    {s}: regions {', '.join(sorted(d['region'].unique()))}")
i = res_t.abs().idxmax()
print(f"  Largest single error: {test.loc[i, 'region']} {test.loc[i, 'date']:%Y-%m}, "
      f"forecast {test.loc[i, main_col]:,.0f}, actual {test.loc[i, 'target_3m']:,.0f}, "
      f"residual {res_t[i]:,.0f}")


# ================================================================ 9. Linear coefficients (L4)
print("\n  Coefficients, linear models (standardised features):")
coef = pd.DataFrame({n: final[n].regressor_.named_steps["model"].coef_[:len(NUMERIC)]
                     for n in ["OLS", "Ridge", "Lasso"]}, index=NUMERIC)
print(coef.to_string(float_format="{:+.4f}".format))
rate_cols = [c for c in NUMERIC if "rate" in c]
print("  Sum of interest-rate coefficients: " + "  ".join(
    f"{n} {coef.loc[rate_cols, n].sum():+.4f}" for n in coef.columns)
      + "  (negative = higher rate gives a lower forecast)")
zero = coef.index[coef["Lasso"].abs() < 1e-8].tolist()
print(f"  Lasso set {len(zero)} numeric coefficient(s) to zero: {zero}")


# ================================================================ 9b. PCA (L9)
# Fitted on the development rows only (the test set plays no part).
pca_full = PCA().fit(StandardScaler().fit_transform(dev[NUMERIC]))
evr = pca_full.explained_variance_ratio_
k = best_params["PCA regression"][PCA_PARAM]
print(f"\n  PCA on the {len(NUMERIC)} standardised numeric features (development rows):")
print("    Cumulative explained variance: " + "  ".join(
    f"{i + 1}:{v:.0%}" for i, v in enumerate(np.cumsum(evr))))
print(f"    PCA regression uses {k} component(s) = {np.cumsum(evr)[k - 1]:.1%} of the variance")
loadings = pd.DataFrame(pca_full.components_[:3].T, index=NUMERIC,
                        columns=[f"PC{i + 1}" for i in range(3)])
print("    Loadings of the first 3 components:")
print(loadings.to_string(float_format="{:+.2f}".format))


# ================================================================ 10. Permutation importance (L10, L12)
# a) Per feature (scikit-learn). Correlated features share information, so a
#    single lag can look unimportant because the other lags stand in for it (L10).
print(f"\n  Permutation importance per feature on test ({main}, increase in MAE in m2):")
pi = permutation_importance(final[main], test[FEATURES], test["target_3m"],
                            n_repeats=10, random_state=SEED,
                            scoring="neg_mean_absolute_error")
importance = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
for f, v in importance.sort_values(ascending=False).items():
    print(f"    {f:<24} {v:>8,.0f}")

# b) Per group: all features in a group are shuffled together (same order), so
#    correlated features cannot stand in for each other.
GROUPS = {
    "Floor area (current, lags, avg)": [f"log_{k}" for k in AREA],
    "Policy rate (current and lags)": ["policy_rate", "rate_lag3", "rate_lag6", "rate_lag12"],
    "Population growth": ["pop_growth_12m"],
    "Region": ["region"],
    "Month": ["month"],
}
rng = np.random.default_rng(SEED)
X_test = test[FEATURES]
base_err = mae(test["target_3m"], final[main].predict(X_test))
group_mean, group_std = {}, {}
for name, cols in GROUPS.items():
    increase = []
    for _ in range(10):
        X = X_test.copy()
        idx = rng.permutation(len(X))
        X[cols] = X_test[cols].to_numpy()[idx]
        increase.append(mae(test["target_3m"], final[main].predict(X)) - base_err)
    group_mean[name], group_std[name] = np.mean(increase), np.std(increase)
group = pd.Series(group_mean).sort_values()
print(f"\n  Permutation importance per group on test ({main}, increase in MAE in m2, "
      f"baseline error {base_err:,.0f}):")
for n in group.index[::-1]:
    print(f"    {n:<32} {group_mean[n]:>8,.0f} ± {group_std[n]:>6,.0f}   "
          f"({group_mean[n] / base_err:+.1%})")


# ================================================================ 11. Figures
# Fig 5: test, actual vs predicted per region
regions = sorted(test["region"].unique())
fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharex=True)
for ax, r in zip(axes.flat, regions):
    d = test[test["region"] == r]
    ax.fill_between(d["date"], d["lower"] / 1000, d["upper"] / 1000, color=BLUE, alpha=0.12,
                    linewidth=0, label=f"{1 - ALPHA:.0%} interval")
    ax.plot(d["date"], d["target_3m"] / 1000, color=GREY, linewidth=1.6, label="Actual")
    ax.plot(d["date"], d[main_col] / 1000, color=BLUE, linewidth=2, label="Predicted")
    ax.set_title(r, loc="left")
    ax.set_ylim(bottom=0)
axes.flat[-1].axis("off")
axes[2, 2].xaxis.set_tick_params(labelbottom=True)
for ax in axes[:, 0]:
    ax.set_ylabel("1,000 m² (3 months)")
axes.flat[0].legend(frameon=False, loc="lower left")
for ax in axes.flat:
    for t in ax.get_xticklabels():
        t.set_rotation(30)
fig.suptitle(f"Test period: actual vs predicted floor area over the next 3 months ({main})",
             x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig("fig5_test_predictions.png", dpi=150)
plt.close(fig)

# Fig 6: permutation importance per group
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(group.index, group.values, xerr=[group_std[n] for n in group.index],
        color=BLUE, height=0.6, error_kw={"ecolor": INK2, "linewidth": 1})
ax.axvline(0, color=INK2, linewidth=0.8)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Increase in MAE (m²) when the group is shuffled (mean ± std over 10 repeats)")
ax.set_title(f"Permutation importance per feature group, {main} (test period)", loc="left")
fig.tight_layout()
fig.savefig("fig6_permutation_importance.png", dpi=150)
plt.close(fig)

# Fig 7: CV error per year: baselines, main model and best tree model
best_tree = min(["Random forest", "Gradient boosting"], key=lambda n: cv_tab.loc[n, "MAE mean"])
lines = [(BASELINE_NAMES[0], GREY, ":"), (BASELINE_NAMES[1], GREY, "--"), (main, BLUE, "-")]
if best_tree != main:
    lines.append((best_tree, ORANGE, "-"))
fig, ax = plt.subplots(figsize=(8, 4))
for n, colour, style in lines:
    ax.plot(CV_YEARS, per_fold[n] / fold_mean * 100, color=colour, linestyle=style,
            linewidth=2, marker="o", markersize=5, label=n)
ax.set_xticks(CV_YEARS)
ax.set_ylabel("Relative error (%)")
ax.set_ylim(bottom=0)
ax.set_title("Cross-validation: relative error per validation year", loc="left")
ax.legend(frameon=False, ncol=2, loc="lower left")
fig.tight_layout()
fig.savefig("fig7_cv_by_year.png", dpi=150)
plt.close(fig)

# Fig 8: residuals (L4) for the main model on test
resid = test["target_3m"] - test[main_col]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(test[main_col] / 1000, resid / 1000, s=14, color=BLUE, alpha=0.6, edgecolors="none")
ax.axhline(0, color=INK2, linewidth=0.8)
ax.set_xlabel("Predicted (1,000 m², 3 months)")
ax.set_ylabel("Residual = actual − predicted (1,000 m²)")
ax.set_title(f"Residuals on the test set ({main})", loc="left")
fig.tight_layout()
fig.savefig("fig8_residuals.png", dpi=150)
plt.close(fig)


# Fig 9: validation curves for the complexity knob (L6)
def curve(ax, name, param, logx=False, fixed=None):
    r = cv_res[name].copy()
    if fixed:
        for p, v in fixed.items():
            r = r[r[f"param_{p}"].astype(str) == str(v)]
    r = r.sort_values(f"param_{param}")
    x = r[f"param_{param}"].astype(float)
    for col, std, colour, lab in [("mean_train_score", "std_train_score", GREY, "Training"),
                                  ("mean_test_score", "std_test_score", BLUE, "Validation")]:
        y, s = -r[col] / 1000, r[std] / 1000
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=5, label=lab)
        ax.fill_between(x, y - s, y + s, color=colour, alpha=0.15, linewidth=0)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(param.split("__")[-1])
    ax.set_title(name, loc="left")


fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
curve(axes[0], "Ridge", P + "alpha", logx=True)
curve(axes[1], "Random forest", P + "min_samples_leaf",
      fixed={P + "max_features": best_params["Random forest"][P + "max_features"]})
curve(axes[2], "Gradient boosting", P + "max_depth",
      fixed={P + "learning_rate": best_params["Gradient boosting"][P + "learning_rate"]})
axes[0].set_ylabel("MAE (1,000 m²), mean ± 1 std")
axes[0].legend(frameon=False)
fig.suptitle("Validation curves: error against the complexity knob (other parameters at best value)",
             x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig("fig9_validation_curves.png", dpi=150)
plt.close(fig)

# Fig 10: PCA regression (L9): CV error vs number of components, explained variance
ncomp = pca_cv[f"param_{PCA_PARAM}"].astype(int)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].errorbar(ncomp, pca_cv["MAE mean"] / 1000, yerr=pca_cv["SE"] / 1000, color=BLUE,
                 linewidth=2, marker="o", markersize=5, capsize=3, label="CV MAE ± 1 SE")
axes[0].axhline(pca_threshold / 1000, color=ORANGE, linestyle="--", linewidth=1.2,
                label="One-SE threshold")
axes[0].plot(k, cv_tab.loc["PCA regression", "MAE mean"] / 1000, marker="o", markersize=11,
             markerfacecolor="none", markeredgecolor=INK, linestyle="none", label=f"Selected ({k})")
axes[0].set_xticks(ncomp)
axes[0].set_xlabel(f"Number of components ({len(NUMERIC)} = OLS)")
axes[0].set_ylabel("MAE (1,000 m²)")
axes[0].set_title("PCA regression: cross-validation error", loc="left")
axes[0].legend(frameon=False)
axes[1].bar(np.arange(1, len(evr) + 1), evr * 100, color=BLUE, width=0.7, label="Component")
axes[1].plot(np.arange(1, len(evr) + 1), np.cumsum(evr) * 100, color=INK2, marker="o",
             markersize=4, linewidth=1.2, label="Cumulative")
axes[1].set_xticks(np.arange(1, len(evr) + 1))
axes[1].set_xlabel("Component")
axes[1].set_ylabel("Explained variance (%)")
axes[1].set_title("Explained variance (development rows)", loc="left")
axes[1].legend(frameon=False, loc="center right")
axes[1].grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("fig10_pca.png", dpi=150)
plt.close(fig)

test[["period", "region", "target_3m", "baseline_last_year", "baseline_last3"]
     + [f"pred_{n}" for n in MODEL_NAMES] + ["lower", "upper"]].to_csv(
    "test_predictions.csv", index=False)
print("\nSaved: fig5_test_predictions.png, fig6_permutation_importance.png, fig7_cv_by_year.png,"
      " fig8_residuals.png, fig9_validation_curves.png, fig10_pca.png, cv_summary.csv,"
      " cv_by_year.csv, test_predictions.csv")
