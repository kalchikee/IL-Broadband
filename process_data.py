"""
Illinois Broadband Gap Analysis Tool — Data Processing Pipeline
GEOG 778: Practicum in GIS Development | Ethan Kalchik

Generates:
  - data/illinois_tracts.geojson   (Leaflet choropleth layer)
  - data/illinois_counties.geojson (county boundary overlay)
  - data/summary_stats.json        (sidebar summary stats)

Data sources used:
  - TIGER/Line tl_2023_17_tract.zip            (census tract geometries)
  - ACSDT5Y2024.B19013 CSV                     (median household income)
  - ACSDT5Y2024.B28002 CSV                     (internet subscriptions)
  - ACSDT5Y2024.B01003 CSV                     (total population)
  - ACSDT5Y2024.B17001 CSV                     (poverty status) [optional — download from Census]
  - FCC BDC bdc_us_fixed_broadband_summary     (county-level 25/3 coverage)
"""

import csv
import io
import json
import math
import os
import re
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "Data")
OUT  = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)

TIGER_ZIP   = os.path.join(DATA, "tl_2023_17_tract.zip")
FCC_US_ZIP  = os.path.join(DATA, "bdc_us_fixed_broadband_summary_by_geography_J25_03feb2026.zip")
FCC_US_CSV  = "bdc_us_fixed_broadband_summary_by_geography_J25_03feb2026.csv"
ACS_INCOME  = os.path.join(DATA, "ACSDT5Y2024.B19013-2026-02-16T233226.csv")
ACS_INTERNET= os.path.join(DATA, "ACSDT5Y2024.B28002-2026-02-16T233303.csv")
ACS_POP     = os.path.join(DATA, "ACSDT5Y2024.B01003-2026-02-16T233330.csv")

# B17001 poverty — prefer API-format download, fall back to wide-format CSV
_pov_api = os.path.join(DATA, "B17001_poverty_api.csv")
_pov_candidates = [f for f in os.listdir(DATA) if f.startswith("ACSDT5Y2024.B17001") and f.endswith(".csv")]
ACS_POVERTY      = _pov_api if os.path.exists(_pov_api) else (
                   os.path.join(DATA, _pov_candidates[0]) if _pov_candidates else None)
ACS_POVERTY_MODE  = "api" if (ACS_POVERTY and ACS_POVERTY == _pov_api) else "wide"
ACS_B28002_2019   = os.path.join(DATA, "B28002_2019_api.csv")
if ACS_POVERTY:
    print(f"  Found poverty file ({ACS_POVERTY_MODE} format): {os.path.basename(ACS_POVERTY)}")
else:
    print("  ! ACS B17001 not found — poverty_pct will be 0.")

# ── Step 1: County FIPS + FCC coverage ────────────────────────────────────────
print("Loading FCC county-level broadband data...")
county_fips_map = {}    # "Adams" -> "001"
county_coverage = {}    # "001"  -> 95.0  (% of residential locations w/ 25/3 Mbps)

with zipfile.ZipFile(FCC_US_ZIP) as z:
    with z.open(FCC_US_CSV) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
        for row in reader:
            if (row["geography_type"] == "County"
                    and row["geography_id"].startswith("17")):
                fips5   = row["geography_id"]          # "17001"
                cfips   = fips5[2:]                    # "001"
                cname   = row["geography_desc"].replace(" County", "").strip()
                county_fips_map[cname] = cfips

                if (row["biz_res"] == "R"
                        and row["technology"] == "Any Technology"
                        and row["area_data_type"] == "Total"):
                    pct = round(float(row["speed_25_3"]) * 100, 1)
                    county_coverage[cfips] = pct

print(f"  {len(county_fips_map)} counties mapped, {len(county_coverage)} with coverage data")

# ── Step 2: Parse ACS "wide" CSVs ─────────────────────────────────────────────
def _parse_col(header: str):
    """
    'Census Tract 2.01; Adams County; Illinois!!Estimate'
    -> ('Adams', '000201')  or None for MOE / non-tract columns
    """
    if "!!" not in header or "Margin" in header:
        return None
    base = header.split("!!")[0]   # ...before !!Estimate
    parts = [p.strip() for p in base.split(";")]
    if len(parts) < 2:
        return None
    m = re.match(r"Census Tract\s+(.+)", parts[0], re.IGNORECASE)
    if not m:
        return None
    tract_num = m.group(1).strip()
    county_raw = parts[1]
    county_name = re.sub(r"\s+County$", "", county_raw, flags=re.IGNORECASE).strip()
    return county_name, tract_num


def _tract_code(tract_num: str) -> str | None:
    """'1' -> '000100',  '2.01' -> '000201',  '9800' -> '980000'"""
    try:
        code = int(round(float(tract_num) * 100))
        return f"{code:06d}"
    except (ValueError, TypeError):
        return None


def read_acs_wide(filepath: str) -> dict[str, list]:
    """
    Returns {geoid: [row0_val, row1_val, ...]}  (only Estimate columns matched
    to a known GEOID; values are raw strings from the CSV).
    Also returns the row-label list so the caller can pick the right index.
    """
    with open(filepath, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    headers   = rows[0]
    data_rows = rows[1:]   # list of rows; each row[0] is the label
    row_labels = [r[0] for r in data_rows]

    result = {}          # geoid -> list of col values (one per data row)
    no_match = []

    for col_idx, hdr in enumerate(headers):
        if col_idx == 0:
            continue   # label column
        parsed = _parse_col(hdr)
        if parsed is None:
            continue
        cname, tnum = parsed
        cfips = county_fips_map.get(cname)
        if cfips is None:
            no_match.append(cname)
            continue
        tcode = _tract_code(tnum)
        if tcode is None:
            continue
        geoid = f"17{cfips}{tcode}"
        result[geoid] = [r[col_idx] for r in data_rows]

    if no_match:
        unique = sorted(set(no_match))
        print(f"  ! Unmatched counties in {os.path.basename(filepath)}: {unique[:10]}")

    return result, row_labels


def _num(s: str) -> float | None:
    try:
        v = float(str(s).replace(",", "").replace("+", "").strip())
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (ValueError, TypeError):
        return None


class _NaNSafeEncoder(json.JSONEncoder):
    """Converts float NaN/Inf to JSON null so browsers don't choke."""
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._clean(o), _one_shot)

    def _clean(self, obj):
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        return obj


print("Parsing ACS B19013 (median income)...")
acs_income, labels_income = read_acs_wide(ACS_INCOME)
# Row label is "Total:" — median household income estimate
income_row_idx = next((i for i, l in enumerate(labels_income) if "Total" in l), 0)

print("Parsing ACS B28002 (internet subscriptions)...")
acs_internet, labels_internet = read_acs_wide(ACS_INTERNET)
total_hh_idx   = next((i for i, l in enumerate(labels_internet) if l.strip() == "Total:"), 0)
no_int_idx     = next((i for i, l in enumerate(labels_internet)
                        if "No Internet" in l), None)
broadband_idx  = next((i for i, l in enumerate(labels_internet)
                        if "Broadband of any type" in l), None)
if no_int_idx is None:
    no_int_idx = len(labels_internet) - 1
if broadband_idx is None:
    # fallback: "With an Internet subscription"
    broadband_idx = next((i for i, l in enumerate(labels_internet)
                          if "With an Internet" in l), 1)
print(f"  total-hh row={total_hh_idx} '{labels_internet[total_hh_idx]}'")
print(f"  broadband row={broadband_idx} '{labels_internet[broadband_idx]}'")
print(f"  no-internet row={no_int_idx} '{labels_internet[no_int_idx]}'")

# ── 2019 B28002 — for change-over-time layer ──────────────────────────────────
b28002_2019 = {}   # geoid -> no_internet_pct_2019
if os.path.exists(ACS_B28002_2019):
    print("Loading 2019 ACS B28002 (internet subscriptions)...")
    with open(ACS_B28002_2019, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geoid   = row["state"] + row["county"] + row["tract"]
            total   = _num(row.get("B28002_001E"))
            no_int  = _num(row.get("B28002_013E"))
            if total and total > 0 and no_int is not None:
                b28002_2019[geoid] = round((no_int / total) * 100, 1)
    print(f"  Loaded {len(b28002_2019)} 2019 tract records")
else:
    print("  ! 2019 B28002 not found — change layer will be unavailable")

print("Parsing ACS B01003 (population)...")
acs_pop, labels_pop = read_acs_wide(ACS_POP)
pop_row_idx = next((i for i, l in enumerate(labels_pop) if "Total" in l), 0)

# ── ACS B17001 (poverty) — optional ───────────────────────────────────────────
# acs_poverty_api: geoid -> (total, below_poverty)  [API format]
# acs_poverty     : geoid -> [row values]            [wide format fallback]
acs_poverty_api  = {}   # populated if API CSV found
acs_poverty      = {}   # populated if wide CSV found
below_poverty_idx = None
poverty_total_idx = 0

if ACS_POVERTY and ACS_POVERTY_MODE == "api":
    print("Parsing B17001_poverty_api.csv (Census API format)...")
    with open(ACS_POVERTY, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Columns: NAME, B17001_001E, B17001_002E, state, county, tract
            geoid = row["state"] + row["county"] + row["tract"]
            total = _num(row.get("B17001_001E"))
            below = _num(row.get("B17001_002E"))
            if total is not None and below is not None:
                acs_poverty_api[geoid] = (total, below)
    print(f"  Loaded {len(acs_poverty_api)} tract poverty records")

elif ACS_POVERTY and ACS_POVERTY_MODE == "wide":
    print("Parsing ACS B17001 (wide format)...")
    acs_poverty, labels_poverty = read_acs_wide(ACS_POVERTY)
    poverty_total_idx = next((i for i, l in enumerate(labels_poverty) if l.strip() == "Total:"), 0)
    below_poverty_idx = next((i for i, l in enumerate(labels_poverty)
                              if "below poverty" in l.lower()), None)
    if below_poverty_idx is None:
        print("  ! Could not find 'below poverty' row — poverty_pct will be 0")

# ── Step 3: Load shapefile ─────────────────────────────────────────────────────
print("Loading TIGER/Line tract shapefile...")
gdf = gpd.read_file(f"zip://{TIGER_ZIP}")
gdf = gdf.to_crs(epsg=4326)   # WGS-84 for Leaflet
print(f"  {len(gdf)} tracts loaded")

# ── Step 4: Attach attributes ─────────────────────────────────────────────────
print("Joining attributes...")

IL_COUNTIES = {}   # cfips -> county display name (built from FCC map)
for cname, cfips in county_fips_map.items():
    IL_COUNTIES[cfips] = cname

records = []
skipped = 0

for _, row in gdf.iterrows():
    geoid   = row["GEOID"]        # "17001000100"
    cfips   = row["COUNTYFP"]     # "001"
    tractce = row["TRACTCE"]      # "000100"
    aland   = row["ALAND"]        # m²

    county_name  = IL_COUNTIES.get(cfips, cfips)

    # ── Income ──
    raw_inc = acs_income.get(geoid)
    income  = _num(raw_inc[income_row_idx]) if raw_inc else None
    if income is None or income <= 0:
        income = None   # will fill with county median later

    # ── Population ──
    raw_pop = acs_pop.get(geoid)
    pop     = int(_num(raw_pop[pop_row_idx]) or 0) if raw_pop else 0

    # ── Internet stats from ACS B28002 ──
    raw_int    = acs_internet.get(geoid)
    no_int_pct   = None
    coverage_pct = None
    total_hh     = 0
    if raw_int:
        total_hh   = _num(raw_int[total_hh_idx]) or 0
        no_int     = _num(raw_int[no_int_idx])
        broadband  = _num(raw_int[broadband_idx])
        if total_hh and total_hh > 0:
            if no_int is not None and no_int >= 0:
                no_int_pct   = round((no_int / total_hh) * 100, 1)
            if broadband is not None and broadband >= 0:
                coverage_pct = round((broadband / total_hh) * 100, 1)

    # Fall back to county-level FCC data if ACS broadband is unavailable
    if coverage_pct is None:
        coverage_pct = county_coverage.get(cfips, 70.0)

    # ── Poverty rate from ACS B17001 ──
    poverty_pct = None
    if acs_poverty_api:
        entry = acs_poverty_api.get(geoid)
        if entry:
            pov_total, pov_below = entry
            if pov_total > 0 and pov_below >= 0:
                poverty_pct = round((pov_below / pov_total) * 100, 1)
    elif acs_poverty and below_poverty_idx is not None:
        raw_pov = acs_poverty.get(geoid)
        if raw_pov:
            pov_total = _num(raw_pov[poverty_total_idx])
            pov_below = _num(raw_pov[below_poverty_idx])
            if pov_total and pov_total > 0 and pov_below is not None and pov_below >= 0:
                poverty_pct = round((pov_below / pov_total) * 100, 1)

    records.append({
        "geoid":        geoid,
        "cfips":        cfips,
        "county":       county_name,
        "pop":          pop,
        "income":       income,
        "no_int_pct":   no_int_pct,
        "coverage_pct": coverage_pct,
        "poverty_pct":     poverty_pct,
        "no_int_pct_2019": b28002_2019.get(geoid),
        "total_hh":        int(total_hh),
        "aland":           aland,
        "geometry":        row.geometry,
    })

attr_df = pd.DataFrame(records)

# ── Fill missing income / no-internet with county medians ─────────────────────
county_income_med = (
    attr_df[attr_df["income"].notna()]
    .groupby("cfips")["income"]
    .median()
)
county_noint_med = (
    attr_df[attr_df["no_int_pct"].notna()]
    .groupby("cfips")["no_int_pct"]
    .median()
)
statewide_income = attr_df["income"].median()
statewide_noint  = attr_df["no_int_pct"].median()

def fill_income(r):
    if r["income"] is not None and not np.isnan(r["income"]):
        return r["income"]
    return county_income_med.get(r["cfips"], statewide_income or 55000)

def fill_noint(r):
    if r["no_int_pct"] is not None and not np.isnan(r["no_int_pct"]):
        return r["no_int_pct"]
    return county_noint_med.get(r["cfips"], statewide_noint or 20.0)

attr_df["income"]     = attr_df.apply(fill_income, axis=1)
attr_df["no_int_pct"] = attr_df.apply(fill_noint,  axis=1)

# ── Fill poverty with county median if available, else 0 ─────────────────────
if acs_poverty_api or (acs_poverty and below_poverty_idx is not None):
    county_poverty_med = (
        attr_df[attr_df["poverty_pct"].notna()]
        .groupby("cfips")["poverty_pct"]
        .median()
    )
    statewide_poverty = attr_df["poverty_pct"].median() or 15.0

    def fill_poverty(r):
        if r["poverty_pct"] is not None and not np.isnan(r["poverty_pct"]):
            return r["poverty_pct"]
        return county_poverty_med.get(r["cfips"], statewide_poverty)

    attr_df["poverty_pct"] = attr_df.apply(fill_poverty, axis=1)
else:
    attr_df["poverty_pct"] = 0.0

# ── Area type from population density ─────────────────────────────────────────
def area_type(row):
    aland_km2 = row["aland"] / 1_000_000
    if aland_km2 <= 0 or row["pop"] == 0:
        return "rural"
    density = row["pop"] / aland_km2
    if density >= 1000:
        return "urban"
    if density >= 200:
        return "suburban"
    if density >= 50:
        return "mixed"
    return "rural"

attr_df["area_type"] = attr_df.apply(area_type, axis=1)

# ── Broadband Need Index ──────────────────────────────────────────────────────
# Weighted composite (0–100):
#   40% broadband gap  (100 - coverage_pct)
#   30% income gap     normalised over $20k–$120k range
#   30% no-internet %  (0–50 capped)

def need_index(row):
    gap_score    = (100 - row["coverage_pct"]) / 100        # 0–1
    income_norm  = max(0.0, min(1.0,
                       (120_000 - row["income"]) / 100_000))
    noint_norm   = min(row["no_int_pct"] / 50, 1.0)        # cap at 50%
    raw = 0.40 * gap_score + 0.30 * income_norm + 0.30 * noint_norm
    return round(raw * 100, 1)

attr_df["need_index"] = attr_df.apply(need_index, axis=1)

# ── Estimated provider count (coarse proxy from coverage tier) ────────────────
def est_providers(cov):
    if cov >= 90: return 4
    if cov >= 70: return 3
    if cov >= 50: return 2
    if cov >= 25: return 1
    return 0

attr_df["provider_count"] = attr_df["coverage_pct"].apply(est_providers)

print(f"  Built {len(attr_df)} tract records")
print(f"  Need index range: {attr_df['need_index'].min():.1f} – {attr_df['need_index'].max():.1f}")
print(f"  Coverage range:   {attr_df['coverage_pct'].min():.1f} – {attr_df['coverage_pct'].max():.1f}")
print(f"  Income range:     ${attr_df['income'].min():,.0f} – ${attr_df['income'].max():,.0f}")
print(f"  Area types: {attr_df['area_type'].value_counts().to_dict()}")

# ── Step 5: Build GeoJSON ──────────────────────────────────────────────────────
print("Building and simplifying GeoJSON...")

# Rebuild as GeoDataFrame
gdf2 = gpd.GeoDataFrame(attr_df, geometry="geometry", crs="EPSG:4326")

# Simplify geometries for web performance (tolerance in degrees ≈ ~50m)
gdf2["geometry"] = gdf2["geometry"].simplify(0.001, preserve_topology=True)

# Drop water-only tracts (ALAND == 0) but keep all others
gdf2 = gdf2[gdf2["aland"] > 0].copy()

features = []
for _, r in gdf2.iterrows():
    if r.geometry is None or r.geometry.is_empty:
        continue
    features.append({
        "type": "Feature",
        "geometry": r.geometry.__geo_interface__,
        "properties": {
            "tract_id":           r["geoid"],
            "county":             r["county"],
            "state":              "Illinois",
            "coverage_pct":       round(float(r["coverage_pct"]), 1),
            "median_income":      round(float(r["income"]), 0),
            "population":         int(r["pop"]),
            "no_internet_pct":    round(float(r["no_int_pct"]), 1),
            "no_internet_2019":   round(float(r["no_int_pct_2019"]), 1) if r["no_int_pct_2019"] is not None else None,
            "change_no_internet": round(float(r["no_int_pct"]) - float(r["no_int_pct_2019"]), 1) if r["no_int_pct_2019"] is not None else None,
            "poverty_pct":        round(float(r["poverty_pct"]), 1),
            "provider_count":     int(r["provider_count"]),
            "need_index":         round(float(r["need_index"]), 1),
            "area_type":          r["area_type"],
            "reliability":        "low" if r["total_hh"] < 100 else ("moderate" if r["total_hh"] < 400 else "high"),
        },
    })

geojson = {"type": "FeatureCollection", "features": features}

out_geo = os.path.join(OUT, "illinois_tracts.geojson")
with open(out_geo, "w", encoding="utf-8") as f:
    json.dump(geojson, f, cls=_NaNSafeEncoder, separators=(",", ":"))

size_mb = os.path.getsize(out_geo) / 1_048_576
print(f"  Wrote {len(features)} features -> {out_geo}  ({size_mb:.1f} MB)")

# ── Step 6: County boundary GeoJSON ───────────────────────────────────────────
# Use authoritative Census TIGER county shapefile — clean polygons, no internal lines
print("Generating county boundary layer from TIGER county shapefile...")
COUNTY_ZIP = os.path.join(DATA, "tl_2023_us_county.zip")
county_src = gpd.read_file(f"zip://{COUNTY_ZIP}").to_crs(epsg=4326)
county_il  = county_src[county_src["STATEFP"] == "17"].copy()
county_il["geometry"] = county_il["geometry"].simplify(0.001, preserve_topology=True)
print(f"  {len(county_il)} Illinois counties loaded")

county_features = []
for _, r in county_il.iterrows():
    if r.geometry is None or r.geometry.is_empty:
        continue
    cfips = r["COUNTYFP"]
    county_features.append({
        "type": "Feature",
        "geometry": r.geometry.__geo_interface__,
        "properties": {
            "cfips":  cfips,
            "county": IL_COUNTIES.get(cfips, r.get("NAME", cfips)),
        },
    })

county_geojson = {"type": "FeatureCollection", "features": county_features}
out_county = os.path.join(OUT, "illinois_counties.geojson")
with open(out_county, "w", encoding="utf-8") as f:
    json.dump(county_geojson, f, cls=_NaNSafeEncoder, separators=(",", ":"))
size_county = os.path.getsize(out_county) / 1_048_576
print(f"  Wrote {len(county_features)} counties -> {out_county}  ({size_county:.2f} MB)")

# ── Step 7: Summary stats JSON ─────────────────────────────────────────────────
print("Computing summary statistics...")

df = attr_df.copy()
total   = len(df)
high    = int((df["need_index"] >= 60).sum())
medium  = int(((df["need_index"] >= 40) & (df["need_index"] < 60)).sum())
low_n   = int(((df["need_index"] >= 20) & (df["need_index"] < 40)).sum())
adequate= int((df["need_index"] <  20).sum())

underserved = int(df[df["coverage_pct"] < 80]["pop"].sum())
avg_cov  = round(df["coverage_pct"].mean(), 1)

cov_by_type = {}
for at in ["urban", "suburban", "mixed", "rural"]:
    sub = df[df["area_type"] == at]
    cov_by_type[at] = round(sub["coverage_pct"].mean(), 1) if len(sub) > 0 else 0.0

summary = {
    "total_tracts":         total,
    "total_population":     int(df["pop"].sum()),
    "underserved_population": underserved,
    "avg_coverage":         avg_cov,
    "high_need_tracts":     high,
    "medium_need_tracts":   medium,
    "low_need_tracts":      low_n,
    "adequate_tracts":      adequate,
    "coverage_by_type":     cov_by_type,
}

out_stats = os.path.join(OUT, "summary_stats.json")
with open(out_stats, "w", encoding="utf-8") as f:
    json.dump(summary, f, cls=_NaNSafeEncoder, indent=2)

print(f"  Wrote -> {out_stats}")
print()
print("=== Summary ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
print()
print("Done! OK")
