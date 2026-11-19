"""
MST0052 - Regional construction activity: exploratory data analysis.

Reads panel_region_month.csv and produces five figures plus key figures that
inform two choices:
  1) Target: floor area next month, or the sum of the next 3 months?
  2) Log transformation of floor area: yes or no?
It also clusters the regions by how their construction cycles move together
(lecture 13). The clustering is exploratory only and is not used as a feature.

Run:     python explore_data.py
Output:  fig1_time_series.png, fig2_seasonality.png, fig3_volatility.png,
         fig4_distribution.png, fig11_region_clusters.png + key figures in the terminal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

BLUE = "#2a78d6"
GREY = "#c9c8c2"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10,
})

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

df = pd.read_csv("panel_region_month.csv")
df["date"] = pd.to_datetime(df["period"])
df = df.sort_values(["region", "date"])
regions = sorted(df["region"].unique())


# ------------------------------------------------ Figure 1: time series per region
fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharex=True)
for ax, reg in zip(axes.flat, regions):
    d = df[df["region"] == reg]
    ax.plot(d["date"], d["area_m2"] / 1000, color=GREY, linewidth=0.8)
    ax.plot(d["date"], d["area_m2"].rolling(12).mean() / 1000, color=BLUE, linewidth=2)
    ax.set_title(reg, loc="left")
    ax.set_ylim(bottom=0)
axes.flat[-1].axis("off")
axes[2, 2].xaxis.set_tick_params(labelbottom=True)
for ax in axes[:, 0]:
    ax.set_ylabel("1,000 m²")
fig.suptitle("Floor area with commencement permits per region: monthly (grey) and "
             "12-month moving average (blue)\nNote: the y-axis differs by region",
             x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig("fig1_time_series.png", dpi=150)
plt.close(fig)


# ------------------------------------------------ Figure 2: seasonal pattern
# Index = monthly value / the region's mean in the same year (1.0 = average month)
df["year_mean"] = df.groupby(["region", "year"])["area_m2"].transform("mean")
df["season_index"] = df["area_m2"] / df["year_mean"]
season = df.groupby("month")["season_index"].agg(["median", lambda s: s.quantile(0.25),
                                                   lambda s: s.quantile(0.75)])
season.columns = ["median", "q25", "q75"]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, 13)
ax.grid(axis="x", visible=False)
ax.bar(x, season["median"], color=BLUE, width=0.7)
ax.vlines(x, season["q25"], season["q75"], color=INK2, linewidth=1.2)
ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle="--")
ax.set_xticks(x, MONTHS)
ax.set_ylabel("Index (1.0 = average month)")
ax.set_title("Seasonal pattern: median across regions and years (line = middle 50 %)", loc="left")
fig.tight_layout()
fig.savefig("fig2_seasonality.png", dpi=150)
plt.close(fig)


# ------------------------------------------------ Figure 3: 1 month vs 3 months
example = "Oslo" if "Oslo" in regions else regions[0]
d = df[df["region"] == example]
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, sharey=True)
axes[0].plot(d["date"], d["area_m2"] / 1000, color=BLUE, linewidth=1.2)
axes[0].set_title(f"{example}: floor area per month", loc="left")
axes[1].plot(d["date"], d["area_m2"].rolling(3).sum() / 3 / 1000, color=BLUE, linewidth=1.2)
axes[1].set_title(f"{example}: 3-month average (what a 3-month target would predict, divided by 3)",
                  loc="left")
for ax in axes:
    ax.set_ylabel("1,000 m²")
    ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig("fig3_volatility.png", dpi=150)
plt.close(fig)


# ------------------------------------------------ Figure 4: distribution, raw vs log
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].hist(df["area_m2"] / 1000, bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
axes[0].set_title("Floor area, raw values", loc="left")
axes[0].set_xlabel("1,000 m²")
axes[0].set_ylabel("Number of months")
axes[1].hist(np.log1p(df["area_m2"].clip(lower=0)), bins=60, color=BLUE,
             edgecolor="white", linewidth=0.5)
axes[1].set_title("Floor area, log(1 + m²)", loc="left")
axes[1].set_xlabel("log(1 + m²)")
fig.tight_layout()
fig.savefig("fig4_distribution.png", dpi=150)
plt.close(fig)


# ------------------------------------------------ Key figures
print("=" * 64)
print("KEY FIGURES")
print("=" * 64)

# A) How much does each region fluctuate from month to month?
df["abs_change"] = df.groupby("region")["area_m2"].pct_change().abs()
tab = df.groupby("region").agg(
    mean_m2=("area_m2", "mean"),
    typical_change=("abs_change", "median"),
).sort_values("mean_m2", ascending=False)
tab["mean_m2"] = tab["mean_m2"].round(0).astype(int)
tab["typical_change"] = (tab["typical_change"] * 100).round(0).astype(int).astype(str) + " %"
print("\nA) Mean per month and typical change from the previous month (median):")
print(tab.to_string())


# B) Choice of target: how well does a simple baseline (same period last year) do?
#    Relative error = MAE / mean. Lower = more predictable target.
def relative_error(column):
    errors, levels = [], []
    for reg in regions:
        s = df[df["region"] == reg].set_index("date")[column]
        last_year = s.shift(12)
        ok = s.notna() & last_year.notna()
        errors.append((s[ok] - last_year[ok]).abs())
        levels.append(s[ok])
    return pd.concat(errors).mean() / pd.concat(levels).mean()


df["sum_next_3"] = df.groupby("region")["area_m2"].transform(
    lambda s: s.rolling(3).sum().shift(-3))
df["next_1"] = df.groupby("region")["area_m2"].shift(-1)
e1 = relative_error("next_1")
e3 = relative_error("sum_next_3")
print("\nB) Baseline 'same period last year', relative error (MAE / mean):")
print(f"   Next month:           {e1:.0%}")
print(f"   Sum of next 3 months: {e3:.0%}")

# C) Choice of log transform: skewness (0 = symmetric, >1 = strongly right-skewed)
raw = df["area_m2"].skew()
logged = np.log1p(df["area_m2"].clip(lower=0)).skew()
print("\nC) Skewness of the distribution (0 = symmetric):")
print(f"   Raw values:  {raw:.2f}")
print(f"   Log:         {logged:.2f}")
largest, smallest = df.groupby("region")["area_m2"].mean().agg(["max", "min"])
print(f"   The largest region has a mean {largest / smallest:.0f} times that of the smallest.")

# D) Seasonal index per month (the numbers behind fig2_seasonality.png)
season_tab = season.round(2).copy()
season_tab.index = MONTHS
season_tab.columns = ["median", "25 %", "75 %"]
print("\nD) Seasonal index per month (1.00 = average month):")
print(season_tab.to_string())


# ------------------------------------------------ E) Clustering: which regions share a cycle? (L13)
# Series: annual growth of the 12-month sum, log(sum12_t) - log(sum12_{t-12}).
# The 12-month sum removes seasonality and the log removes region size, so
# regions are compared on the shape of their cycle, not on their level.
# Distance = 1 - correlation (0 = move together, 2 = move in opposite directions).
# Hierarchical clustering with average linkage. Only data up to 2022 is used,
# so the test period plays no part.
pre = df[df["date"] <= "2022-12-01"].copy()
pre["sum12"] = pre.groupby("region")["area_m2"].transform(lambda s: s.rolling(12).sum())
pre["growth"] = np.log(pre["sum12"]) - np.log(pre.groupby("region")["sum12"].shift(12))
growth = pre.pivot(index="date", columns="region", values="growth").dropna()
corr = growth.corr()
dist = 1 - corr
Z = linkage(squareform(dist.to_numpy(), checks=False), method="average")

print(f"\nE) Clustering of regions on annual growth ({growth.index.min():%Y-%m} - "
      f"{growth.index.max():%Y-%m}, {len(growth)} months), average linkage:")
sil = {}
for k in range(2, 6):
    labels = fcluster(Z, k, criterion="maxclust")
    sil[k] = silhouette_score(dist.to_numpy(), labels, metric="precomputed")
    print(f"   k = {k}: silhouette {sil[k]:.2f}")
k_best = max(sil, key=sil.get)
labels = pd.Series(fcluster(Z, k_best, criterion="maxclust"), index=corr.index)
print(f"   Highest silhouette at k = {k_best}:")
for c, members in labels.groupby(labels):
    print(f"     Cluster {c}: {', '.join(members.index)}")
mean_corr = (corr.sum() - 1) / (len(corr) - 1)
print("   Mean correlation with the other regions:")
for reg, v in mean_corr.sort_values().items():
    print(f"     {reg:<20} {v:+.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
d = dendrogram(Z, labels=corr.index.tolist(), orientation="left", ax=axes[0],
               color_threshold=0, above_threshold_color=BLUE)
axes[0].set_xlabel("Distance = 1 − correlation")
axes[0].set_title("Hierarchical clustering (average linkage)", loc="left")
axes[0].grid(axis="y", visible=False)
order = d["ivl"][::-1]
im = axes[1].imshow(corr.loc[order, order], cmap="RdBu", vmin=-1, vmax=1)
axes[1].set_xticks(range(len(order)), order, rotation=60, ha="right")
axes[1].set_yticks(range(len(order)), order)
axes[1].grid(False)
axes[1].set_title("Correlation of annual growth (same order)", loc="left")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle(f"Regions grouped by how their construction cycles move together "
             f"({growth.index.min():%Y}–{growth.index.max():%Y})", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig("fig11_region_clusters.png", dpi=150)
plt.close(fig)

print("\nFigures saved: fig1_time_series.png, fig2_seasonality.png, "
      "fig3_volatility.png, fig4_distribution.png, fig11_region_clusters.png")
