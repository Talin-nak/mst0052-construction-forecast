"""
MST0052 - Regional construction activity: data pipeline.

Builds a model-ready panel: one row per region per month, 2001-2026.

  Floor area = TOTAL floor area with approved commencement permits (m2):
                 floor area for residential use (SSB table 08843)
               + floor area for non-residential use (SSB table 14543, summed
                 over all 107 building types; the table has no total code).
  Columns: next-month floor area (used to build the target), current floor
           area and lags (1, 3, 6, 12 months), 3-month average, policy rate
           (current and 3/6/12 months back), 12-month population growth,
           month and region.

REGIONS - 11 regions (the counties of 2020-2023):
  County boundaries changed in 2018, 2020 and 2024. SSB does not publish
  floor area for Østfold/Akershus/Buskerud, Vestfold/Telemark or
  Troms/Finnmark separately for 2020-2023 (only Viken, Vestfold og Telemark
  and Troms og Finnmark). The 2020-2023 structure is therefore the only one
  that every year can be aggregated into:
    - before 2020: old counties are summed (e.g. Hedmark + Oppland = Innlandet)
    - from 2024:   new counties are summed (e.g. Østfold + Akershus + Buskerud = Viken)
  Each county code is used only in the years it was valid, so nothing is
  counted twice. Finally, the sum of the regions is checked against SSB's
  national totals.

Self-contained: fetches all data itself and needs no other project files.

Run:     python build_dataset_b.py
Output:  panel_region_month.csv
"""
import itertools
import time

import pandas as pd
import requests

SSB_BASE = "https://data.ssb.no/api/pxwebapi/v2/tables"
NB_BASE = "https://data.norges-bank.no/api/data"


# ---------------------------------------------------------------- API helpers

def fetch_policy_rate():
    """Norges Bank key policy rate, monthly series (IR/M.KPRA.SD)."""
    url = f"{NB_BASE}/IR/M.KPRA.SD"
    r = requests.get(url, params={"format": "sdmx-json"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    series = data["data"]["dataSets"][0]["series"]
    time_periods = data["data"]["structure"]["dimensions"]["observation"][0]["values"]
    key = list(series.keys())[0]
    obs = series[key]["observations"]
    rows = []
    for idx, period in enumerate(time_periods):
        if str(idx) in obs:
            rows.append({"period": period["id"], "policy_rate": obs[str(idx)][0]})
    df = pd.DataFrame(rows)
    df["year"] = df["period"].str.slice(0, 4).astype(int)
    df["month"] = df["period"].str.slice(-2).astype(int)
    return df[["year", "month", "policy_rate"]]


def _jsonstat_to_long_df(js, value_name):
    """Generic JSON-stat2 -> long-format DataFrame. Uses the actual dimension
    order from js["id"], so it works for any number of dimensions."""
    dim_order = js["id"]
    dims = js["dimension"]
    index_lists = [list(dims[d]["category"]["index"].keys()) for d in dim_order]
    rows = []
    for combo, v in zip(itertools.product(*index_lists), js["value"]):
        row = dict(zip(dim_order, combo))
        row[value_name] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _jsonstat_to_df(js, value_name):
    """Minimal JSON-stat2 -> long-format DataFrame (Region x Tid), for tables
    where all other dimensions have a single selected value."""
    dims = js["dimension"]
    region_cat = dims["Region"]["category"]
    region_ids = list(region_cat["index"].keys())
    period_ids = list(dims["Tid"]["category"]["index"].keys())
    region_labels = region_cat.get("label", {})
    values = js["value"]
    dim_order, sizes = js["id"], js["size"]
    n_region = sizes[dim_order.index("Region")]
    n_period = sizes[dim_order.index("Tid")]

    rows, i = [], 0
    for r_idx in range(n_region):
        for t_idx in range(n_period):
            rows.append({
                "county": region_labels.get(region_ids[r_idx], region_ids[r_idx]),
                "county_code": region_ids[r_idx],
                "period": period_ids[t_idx],
                value_name: values[i],
            })
            i += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- county codes

END = 9999

# county code: (region, first year, last year the code is valid)
COUNTIES = {
    "01": ("Viken", 2000, 2019), "02": ("Viken", 2000, 2019), "06": ("Viken", 2000, 2019),
    "30": ("Viken", 2020, 2023),
    "31": ("Viken", 2024, END), "32": ("Viken", 2024, END), "33": ("Viken", 2024, END),

    "03": ("Oslo", 2000, END),

    "04": ("Innlandet", 2000, 2019), "05": ("Innlandet", 2000, 2019),
    "34": ("Innlandet", 2020, END),

    "07": ("Vestfold og Telemark", 2000, 2019), "08": ("Vestfold og Telemark", 2000, 2019),
    "38": ("Vestfold og Telemark", 2020, 2023),
    "39": ("Vestfold og Telemark", 2024, END), "40": ("Vestfold og Telemark", 2024, END),

    "09": ("Agder", 2000, 2019), "10": ("Agder", 2000, 2019),
    "42": ("Agder", 2020, END),

    "11": ("Rogaland", 2000, END),

    "12": ("Vestland", 2000, 2019), "14": ("Vestland", 2000, 2019),
    "46": ("Vestland", 2020, END),

    "15": ("Møre og Romsdal", 2000, END),

    "16": ("Trøndelag", 2000, 2017), "17": ("Trøndelag", 2000, 2017),
    "50": ("Trøndelag", 2018, END),

    "18": ("Nordland", 2000, END),

    "19": ("Troms og Finnmark", 2000, 2019), "20": ("Troms og Finnmark", 2000, 2019),
    "54": ("Troms og Finnmark", 2020, 2023),
    "55": ("Troms og Finnmark", 2024, END), "56": ("Troms og Finnmark", 2024, END),
}


# ---------------------------------------------------------------- fetching

def _fetch(table, region_codelist, extra):
    params = {
        "lang": "no",
        "valuecodes[Region]": "*",
        "codelist[Region]": region_codelist,
        "valuecodes[Tid]": "*",
        "outputformat": "json-stat2",
        **extra,
    }
    r = requests.get(f"{SSB_BASE}/{table}/data", params=params, timeout=120)
    r.raise_for_status()
    return _jsonstat_to_df(r.json(), value_name="value")


AREA = {"valuecodes[Byggeareal]": "_T", "valuecodes[ContentsCode]": "Bruksarealgang"}
POP = {"valuecodes[ContentsCode]": "Folketallet11"}


def _fetch_residential(region_codelist):
    """Floor area for residential use, year-to-date (08843, total code _T)."""
    df = _fetch("08843", region_codelist, AREA)
    return df[["county_code", "period", "value"]].rename(columns={"value": "residential"})


def _fetch_non_residential(codes, region_codelist):
    """Floor area for non-residential use, year-to-date (14543). The table has
    no total code, so all building types are fetched and summed. One region
    per request keeps each query below SSB's limit of 800,000 cells."""
    parts = []
    for i, code in enumerate(codes):
        params = {
            "lang": "no",
            "valuecodes[Region]": code,
            "codelist[Region]": region_codelist,
            "valuecodes[Byggeareal]": "*",
            "valuecodes[ContentsCode]": "BruksaarealIgang",
            "valuecodes[Tid]": "*",
            "outputformat": "json-stat2",
        }
        r = requests.get(f"{SSB_BASE}/14543/data", params=params, timeout=120)
        r.raise_for_status()
        d = _jsonstat_to_long_df(r.json(), value_name="value")
        d = d.groupby(["Region", "Tid"], as_index=False)["value"].sum(min_count=1)
        parts.append(d)
        print(f"    non-residential {i + 1}/{len(codes)} (code {code})", end="\r")
        time.sleep(2.1)          # SSB allows roughly 30 requests per minute
    print()
    df = pd.concat(parts, ignore_index=True)
    return df.rename(columns={"Region": "county_code", "Tid": "period",
                              "value": "non_residential"})


def _combine(residential, non_residential):
    df = residential.merge(non_residential, on=["county_code", "period"], how="outer")
    df["value"] = df["residential"] + df["non_residential"]
    df["year"] = df["period"].str.slice(0, 4).astype(int)
    df["month"] = df["period"].str.slice(-2).astype(int)
    return df


def fetch_floor_area_by_code():
    """Total floor area (residential + non-residential) per county code, year-to-date."""
    residential = _fetch_residential("vs_Fylker")
    non_residential = _fetch_non_residential(list(COUNTIES), "vs_Fylker")
    df = _combine(residential, non_residential)
    return df[["county_code", "year", "month", "value"]]


def fetch_population_by_code():
    df = _fetch("01222", "vs_Fylker", POP)
    df["year"] = df["period"].str.slice(0, 4).astype(int)
    df["quarter"] = df["period"].str.slice(-1).astype(int)
    return df[["county_code", "year", "quarter", "value"]]


# ------------------------------------------------- county codes -> regions

def _valid(df):
    """Keep county codes only in the years they were valid, and add the region."""
    df = df[df["county_code"].isin(COUNTIES)].copy()
    info = df["county_code"].map(COUNTIES)
    df["region"] = info.str[0]
    valid = (df["year"] >= info.str[1]) & (df["year"] <= info.str[2])
    return df[valid]


def _sum_to_region(df, keys, column):
    """Sum county codes into regions. If one code is missing in a period, the
    whole region is set to NaN, so a partial sum never looks like a real fall."""
    g = df.groupby(keys)[column]
    out = g.agg(sum="sum", n="count", N="size").reset_index()
    out.loc[out["n"] < out["N"], "sum"] = float("nan")
    return out.drop(columns=["n", "N"]).rename(columns={"sum": column})


def floor_area_to_regions(codes):
    """Year-to-date -> monthly figures per county code, then summed per region."""
    df = codes.sort_values(["county_code", "year", "month"]).copy()
    df["area_m2"] = df.groupby(["county_code", "year"])["value"].diff()
    jan = df["month"] == 1
    df.loc[jan, "area_m2"] = df.loc[jan, "value"]
    df = _valid(df)
    return _sum_to_region(df, ["region", "year", "month"], "area_m2")


def population_to_regions(codes):
    df = codes.rename(columns={"value": "population"})
    df = df[df["population"] > 0]          # 0 = the county did not exist
    df = _valid(df)
    df = _sum_to_region(df, ["region", "year", "quarter"], "population")
    df["month"] = df["quarter"] * 3        # population refers to the end of the quarter
    return df[["region", "year", "month", "population"]]


# ---------------------------------------------------------------- panel

def build_panel(area, population, rate):
    df = area.merge(population, on=["region", "year", "month"], how="left")
    df = df.merge(rate, on=["year", "month"], how="left")
    df = df.sort_values(["region", "year", "month"]).reset_index(drop=True)
    df["population"] = df.groupby("region")["population"].ffill()
    df["policy_rate"] = df["policy_rate"].ffill()

    g = df.groupby("region")

    # Floor area next month in the same region (used to build the 3-month target)
    df["area_next_month"] = g["area_m2"].shift(-1)

    # Features
    for lag in [1, 3, 6, 12]:
        df[f"area_lag{lag}"] = g["area_m2"].shift(lag)
    df["area_avg3"] = g["area_m2"].transform(lambda s: s.rolling(3).mean())
    for lag in [3, 6, 12]:
        df[f"rate_lag{lag}"] = g["policy_rate"].shift(lag)
    df["pop_growth_12m"] = g["population"].pct_change(12, fill_method=None)

    df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

    columns = [
        "period", "year", "month", "region",
        "area_next_month",
        "area_m2", "area_lag1", "area_lag3", "area_lag6", "area_lag12", "area_avg3",
        "policy_rate", "rate_lag3", "rate_lag6", "rate_lag12",
        "population", "pop_growth_12m",
    ]
    return df[columns].dropna().reset_index(drop=True)


# ---------------------------------------------------------------- checks

def check_against_national(area_reg, pop_reg):
    """The sum of the 11 regions should be close to SSB's national totals.
    Small deviations are expected (Svalbard and 'unknown' are not in a region)."""
    print("\nCheck against national totals (deviations above 2 % are listed):")
    try:
        nat = _combine(_fetch_residential("vs_Landet"), _fetch_non_residential(["0"], "vs_Landet"))
        dec = nat[nat["month"] == 12].set_index("year")      # full-year totals
        share = (dec["residential"] / dec["value"]).dropna()
        print(f"  Residential share of total floor area, whole country: "
              f"{share.min():.0%}-{share.max():.0%} per year (mean {share.mean():.0%})")
        dec = dec["value"]
        reg = area_reg.groupby("year")["area_m2"].sum(min_count=1)
        dev = ((reg - dec) / dec).dropna()
        large = dev[dev.abs() > 0.02]
        print(f"  Floor area: {len(dev)} years checked, largest deviation "
              f"{dev.abs().max():.2%}" + ("" if large.empty else f", NOTE: {large.round(3).to_dict()}"))
    except Exception as e:
        print(f"  Floor area: could not fetch national totals ({e})")

    try:
        nat = _fetch("01222", "vs_Landet", POP)
        nat["year"] = nat["period"].str.slice(0, 4).astype(int)
        nat["month"] = nat["period"].str.slice(-1).astype(int) * 3
        nat = nat.set_index(["year", "month"])["value"]
        reg = pop_reg.groupby(["year", "month"])["population"].sum(min_count=1)
        dev = ((reg - nat) / nat).dropna()
        dev = dev[dev.index.get_level_values(0) >= 2000]
        large = dev[dev.abs() > 0.02]
        print(f"  Population: {len(dev)} quarters checked, largest deviation "
              f"{dev.abs().max():.2%}" + ("" if large.empty else f", NOTE: {len(large)} quarters"))
    except Exception as e:
        print(f"  Population: could not fetch national totals ({e})")


def check_panel(df):
    print("\nPanel:")
    print(f"  Rows:    {len(df)}")
    print(f"  Regions: {df['region'].nunique()}")
    print(f"  Period:  {df['period'].min()} - {df['period'].max()}")

    per_region = df.groupby("region").size()
    if per_region.nunique() > 1:
        print("  NOTE: different number of months per region:")
        print(per_region.to_string())
    else:
        print(f"  Months per region: {per_region.iloc[0]} (same for all)")

    negative = (df["area_m2"] < 0).sum()
    if negative:
        print(f"  NOTE: {negative} negative monthly values "
              "(SSB revisions in the year-to-date figures).")


if __name__ == "__main__":
    print("Fetching floor area (all county codes)...")
    area_reg = floor_area_to_regions(fetch_floor_area_by_code())
    print("Fetching population (all county codes)...")
    pop_reg = population_to_regions(fetch_population_by_code())
    print("Fetching policy rate...")
    rate = fetch_policy_rate()

    check_against_national(area_reg, pop_reg)

    print("\nBuilding panel...")
    panel = build_panel(area_reg, pop_reg, rate)
    check_panel(panel)

    panel.to_csv("panel_region_month.csv", index=False)
    print("\nSaved: panel_region_month.csv")
    print(panel.head())
