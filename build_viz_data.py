"""
build_viz_data.py — Extract ALL visualization data from the real clinopyroxene dataset.
Outputs web/data/viz_data.json with:
  - Dataset overview (rows, cols, target stats)
  - Class distribution (full counts)
  - Per-oxide statistics (mean, std, median, Q1, Q3 per class)
  - Missing rates (per feature, per class)
  - Feature correlations
  - Feature discrimination power (CV of class means)
  - Geographic samples for map (stratified sample ~20K)
  - Model results (from scripts)
  - X/y summary: feature distributions, class balance
"""
import csv, json, os, sys, random, time
import numpy as np

t0 = time.time()

BASE = "/home/laz/proj/geochemical-research"
CSV_PATH = os.path.join(BASE, "clinopyroxene_data.csv")
OUT_DIR = os.path.join(BASE, "web", "data")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "TECTONIC SETTING"

RAW_FE_COLS = ["FEOT(WT%)", "FEO(WT%)", "FE2O3(WT%)", "FE2O3T(WT%)"]

MAJOR_OXIDES = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "CR2O3(WT%)",
    "FEOT_CALC(WT%)",
    "CAO(WT%)", "MGO(WT%)", "MNO(WT%)", "NA2O(WT%)",
    "K2O(WT%)", "NIO(WT%)",
]

TRACE_ELEMENTS = [
    "SC(PPM)", "V(PPM)", "CR(PPM)", "CO(PPM)", "NI(PPM)",
    "ZN(PPM)", "GA(PPM)", "RB(PPM)", "SR(PPM)", "Y(PPM)",
    "ZR(PPM)", "NB(PPM)", "BA(PPM)", "LA(PPM)", "CE(PPM)",
    "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)", "GD(PPM)",
    "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "TM(PPM)",
    "YB(PPM)", "LU(PPM)", "HF(PPM)", "TA(PPM)", "PB(PPM)",
    "TH(PPM)", "U(PPM)",
]

ENDMEMBERS = [
    "ENSTATITE(MOL%)", "FERROSILITE(MOL%)", "WOLLASTONITE(MOL%)",
    "DIOPSIDE(MOL%)", "HEDENBERGITE(MOL%)", "JADEITE(MOL%)", "ACMITE(MOL%)",
]

ALL_FEATURES = MAJOR_OXIDES + TRACE_ELEMENTS + ENDMEMBERS

# Short labels for display
SHORT = {
    "SIO2(WT%)": "SiO₂", "TIO2(WT%)": "TiO₂", "AL2O3(WT%)": "Al₂O₃",
    "CR2O3(WT%)": "Cr₂O₃", "FEOT_CALC(WT%)": "FeOT",
    "FEOT(WT%)": "FeOt", "FEO(WT%)": "FeO",
    "FE2O3(WT%)": "Fe₂O₃", "FE2O3T(WT%)": "Fe₂O₃T",
    "CAO(WT%)": "CaO", "MGO(WT%)": "MgO", "MNO(WT%)": "MnO",
    "NA2O(WT%)": "Na₂O", "K2O(WT%)": "K₂O", "NIO(WT%)": "NiO",
}
for t in TRACE_ELEMENTS:
    SHORT[t] = t.replace("(PPM)", "")
for e in ENDMEMBERS:
    SHORT[e] = e.replace("(MOL%)", "").capitalize()

SETTING_SHORT = {
    "ARCHEAN CRATON (INCLUDING GREENSTONE BELTS)": "Archean Craton",
    "COMPLEX VOLCANIC SETTINGS": "Complex Volcanic",
    "CONTINENTAL FLOOD BASALT": "Continental Flood Basalt",
    "CONVERGENT MARGIN": "Convergent Margin",
    "INTRAPLATE VOLCANICS": "Intraplate Volcanics",
    "OCEAN ISLAND": "Ocean Island",
    "OCEAN-BASIN FLOOD BASALT": "Ocean-Basin Flood",
    "OCEANIC PLATEAU": "Oceanic Plateau",
    "RIFT VOLCANICS": "Rift Volcanics",
    "SEAMOUNT": "Seamount",
    "SUBMARINE RIDGE": "Submarine Ridge",
}

# ── 1. Load raw data ─────────────────────────────────────────────────
print("[1/7] Loading CSV...", flush=True)
rows_raw = []
with open(CSV_PATH, encoding="latin-1") as f:
    reader = csv.DictReader(f)
    all_columns = reader.fieldnames
    for r in reader:
        rows_raw.append(r)
        if len(rows_raw) % 50000 == 0:
            print(f"  Read {len(rows_raw):,} rows...", flush=True)

n_total = len(rows_raw)
n_cols = len(all_columns)
print(f"  Total: {n_total:,} rows x {n_cols} columns", flush=True)

# ── 2. Filter rows with valid target ────────────────────────────────
print("[2/7] Filtering valid target rows...", flush=True)
rows = [r for r in rows_raw if (r.get(TARGET) or "").strip()]
n_valid = len(rows)
n_no_target = n_total - n_valid
print(f"  Valid target: {n_valid:,}, No target: {n_no_target:,}", flush=True)

def safe_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None

# ── 2b. Consolidate Fe columns into FEOT_CALC(WT%) ─────────────────
print("[2b/7] Consolidating Fe columns into FEOT_CALC(WT%)...", flush=True)
fe2o3_to_feo = 2 * 71.844 / 159.688
fe_priority = [
    ("FEOT(WT%)", 1.0),
    ("FEO(WT%)", 1.0),
    ("FE2O3T(WT%)", fe2o3_to_feo),
    ("FE2O3(WT%)", fe2o3_to_feo),
]
n_fe_filled = 0
for r in rows:
    feot_calc = None
    for src_col, factor in fe_priority:
        v = safe_float(r.get(src_col, ""))
        if v is not None:
            feot_calc = v * factor
            break
    if feot_calc is not None:
        r["FEOT_CALC(WT%)"] = str(feot_calc)
        n_fe_filled += 1
    else:
        r["FEOT_CALC(WT%)"] = ""
print(f"  FEOT_CALC filled for {n_fe_filled:,} / {len(rows):,} rows", flush=True)

# ── 3. Class distribution ───────────────────────────────────────────
print("[3/7] Computing class distribution...", flush=True)
class_counts = {}
for r in rows:
    ts = r[TARGET].strip()
    class_counts[ts] = class_counts.get(ts, 0) + 1

# Sort by count descending
class_counts_sorted = dict(sorted(class_counts.items(), key=lambda x: -x[1]))
print(f"  {len(class_counts_sorted)} classes", flush=True)
for ts, cnt in class_counts_sorted.items():
    print(f"    {ts}: {cnt:,} ({100*cnt/n_valid:.1f}%)", flush=True)

# ── 4. Feature analysis ─────────────────────────────────────────────
print("[4/7] Analyzing features (missing rates, stats, correlations)...", flush=True)

# Build numeric arrays for features
feature_data = {}  # col -> list of (value_or_None)
for col in ALL_FEATURES:
    vals = []
    for r in rows:
        vals.append(safe_float(r.get(col, "")))
    feature_data[col] = vals

# Also build numeric arrays for raw Fe columns (for raw data quality view)
feature_data_raw_fe = {}
for col in RAW_FE_COLS:
    vals = []
    for r in rows:
        vals.append(safe_float(r.get(col, "")))
    feature_data_raw_fe[col] = vals

# Missing rates per feature
missing_rates = {}
for col in ALL_FEATURES:
    vals = feature_data[col]
    n_miss = sum(1 for v in vals if v is None)
    missing_rates[col] = round(100.0 * n_miss / len(vals), 2)

# Missing rates per feature per class
missing_by_class = {}
for col in MAJOR_OXIDES:
    missing_by_class[col] = {}
    for ts in class_counts_sorted:
        n_total_cls = 0
        n_miss_cls = 0
        for i, r in enumerate(rows):
            if r[TARGET].strip() == ts:
                n_total_cls += 1
                if feature_data[col][i] is None:
                    n_miss_cls += 1
        missing_by_class[col][ts] = round(100.0 * n_miss_cls / n_total_cls, 1) if n_total_cls > 0 else 0

# Per-feature statistics (overall)
feature_stats = {}
for col in ALL_FEATURES:
    valid = [v for v in feature_data[col] if v is not None]
    if len(valid) > 0:
        arr = np.array(valid)
        feature_stats[col] = {
            "n": len(valid),
            "mean": round(float(np.mean(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "q1": round(float(np.percentile(arr, 25)), 3),
            "q3": round(float(np.percentile(arr, 75)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
        }
    else:
        feature_stats[col] = {"n": 0}

# Per-class means for major oxides (for discrimination & box plots)
class_oxide_stats = {}
for ts in class_counts_sorted:
    class_oxide_stats[ts] = {}
    indices = [i for i, r in enumerate(rows) if r[TARGET].strip() == ts]
    for col in MAJOR_OXIDES:
        vals = [feature_data[col][i] for i in indices if feature_data[col][i] is not None]
        if vals:
            arr = np.array(vals)
            class_oxide_stats[ts][col] = {
                "n": len(vals),
                "mean": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "median": round(float(np.median(arr)), 3),
                "q1": round(float(np.percentile(arr, 25)), 3),
                "q3": round(float(np.percentile(arr, 75)), 3),
            }

# Feature discrimination power (CV of class means)
discrimination = {}
for col in MAJOR_OXIDES:
    class_means = []
    for ts in class_counts_sorted:
        if ts in class_oxide_stats and col in class_oxide_stats[ts]:
            class_means.append(class_oxide_stats[ts][col]["mean"])
    if len(class_means) >= 2:
        arr = np.array(class_means)
        overall_mean = np.mean(arr)
        if overall_mean != 0:
            discrimination[col] = round(float(np.std(arr) / abs(overall_mean)), 4)

# Missing rates for raw individual Fe columns (for raw data quality view)
missing_rates_raw_fe = {}
for col in RAW_FE_COLS:
    vals = feature_data_raw_fe[col]
    n_miss = sum(1 for v in vals if v is None)
    missing_rates_raw_fe[col] = round(100.0 * n_miss / len(vals), 2)

# Correlation matrix for major oxides
print("  Computing correlations...", flush=True)
corr_cols = ["SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "CR2O3(WT%)",
             "FEOT_CALC(WT%)", "CAO(WT%)", "MGO(WT%)", "MNO(WT%)", "NA2O(WT%)"]
# Build matrix: only rows where ALL corr_cols have values
corr_rows = []
for i in range(len(rows)):
    vals = [feature_data[c][i] for c in corr_cols]
    if all(v is not None for v in vals):
        corr_rows.append(vals)
print(f"  Correlation matrix: {len(corr_rows):,} complete rows", flush=True)

corr_matrix = {}
if len(corr_rows) > 100:
    arr = np.array(corr_rows)
    cc = np.corrcoef(arr.T)
    for i, ci in enumerate(corr_cols):
        corr_matrix[ci] = {}
        for j, cj in enumerate(corr_cols):
            corr_matrix[ci][cj] = round(float(cc[i, j]), 3)

# ── 5. Geographic samples for map ───────────────────────────────────
print("[5/7] Extracting geographic samples...", flush=True)
MAP_OXIDES = ["SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "CAO(WT%)", "MGO(WT%)",
              "NA2O(WT%)", "FEOT(WT%)", "CR2O3(WT%)"]

geo_rows = []
for r in rows:
    try:
        lat = float(r.get("LATITUDE (MIN.)", ""))
        lon = float(r.get("LONGITUDE (MIN.)", ""))
    except (ValueError, TypeError):
        continue
    ts = r[TARGET].strip()
    rec = {"lat": round(lat, 4), "lon": round(lon, 4), "ts": ts}
    loc = (r.get("LOCATION") or "").strip()
    rock = (r.get("ROCK NAME") or "").strip()
    if loc: rec["loc"] = loc[:80]
    if rock: rec["rock"] = rock[:60]
    for ox in MAP_OXIDES:
        try:
            rec[ox.split("(")[0].lower()] = round(float(r.get(ox, "")), 2)
        except (ValueError, TypeError):
            pass
    geo_rows.append(rec)

print(f"  Rows with coordinates: {len(geo_rows):,}", flush=True)

# Stratified sample ~20K
by_ts = {}
for r in geo_rows:
    by_ts.setdefault(r["ts"], []).append(r)

random.seed(42)
sampled = []
for ts_name in sorted(by_ts):
    recs = by_ts[ts_name]
    n = min(len(recs), max(50, int(20000 * len(recs) / len(geo_rows))))
    sampled.extend(random.sample(recs, n))
    print(f"  {SETTING_SHORT.get(ts_name, ts_name)}: {len(recs)} -> {n}", flush=True)
print(f"  Sampled: {len(sampled):,}", flush=True)

# ── 6. Model results (from scripts) ─────────────────────────────────
print("[6/7] Compiling model results...", flush=True)
model_results = {
    "baseline": {
        "name": "Baseline (median impute)",
        "algorithm": "XGBoost (tuned)",
        "features": 14, "rows": 177947,
        "acc": 0.8369, "f1m": 0.7316, "f1w": 0.8361,
        "desc": "Drop >80% missing cols, median impute, RF/XGB/LGBM/MLP compared"
    },
    "native_nan": {
        "name": "Native NaN (XGBoost)",
        "algorithm": "XGBoost + LightGBM",
        "features": 50, "rows": 203179,
        "acc": 0.8631, "f1m": 0.7717,
        "desc": "No imputation — tree models learn split directions for NaN"
    },
    "subset_M1": {
        "name": "Subset M1 (major miss≤1)",
        "algorithm": "XGBoost",
        "features": 18, "rows": 99840,
        "acc": 0.8722, "f1m": 0.8216,
        "desc": "Restrict to near-complete major oxide + endmember data"
    },
    "subset_M2T0": {
        "name": "Subset M2_T0 (full trace)",
        "algorithm": "XGBoost",
        "features": 50, "rows": 1195,
        "acc": 0.9749, "f1m": 0.9717,
        "desc": "Complete trace elements — highest accuracy, fewest samples"
    },
    "hierarchical": {
        "name": "Hierarchical (S1+S2)",
        "algorithm": "Two-stage XGBoost",
        "features": "18+32", "rows": 42355,
        "acc": 0.8352, "f1m": 0.7422,
        "desc": "Stage1: major oxides → Stage2: refine with trace elements"
    },
}

# ── 7. X/y summary ──────────────────────────────────────────────────
print("[7/7] Building X/y summary...", flush=True)

# Feature group completeness
n_rows_valid = len(rows)
major_complete = sum(1 for i in range(n_rows_valid) if all(
    feature_data[c][i] is not None for c in MAJOR_OXIDES))
trace_complete = sum(1 for i in range(n_rows_valid) if all(
    feature_data[c][i] is not None for c in TRACE_ELEMENTS))
endmember_complete = sum(1 for i in range(n_rows_valid) if all(
    feature_data[c][i] is not None for c in ENDMEMBERS))

# Per-row missing count distribution for major oxides
major_miss_dist = {}
for i in range(n_rows_valid):
    n_miss = sum(1 for c in MAJOR_OXIDES if feature_data[c][i] is None)
    major_miss_dist[n_miss] = major_miss_dist.get(n_miss, 0) + 1

trace_miss_dist = {}
for i in range(n_rows_valid):
    n_miss = sum(1 for c in TRACE_ELEMENTS if feature_data[c][i] is None)
    bucket = min(n_miss, 32)
    trace_miss_dist[bucket] = trace_miss_dist.get(bucket, 0) + 1

# ── Output ───────────────────────────────────────────────────────────
print("Writing output JSON...", flush=True)

output = {
    "dataset": {
        "total_rows": n_total,
        "total_columns": n_cols,
        "valid_target_rows": n_valid,
        "no_target_rows": n_no_target,
        "n_classes": len(class_counts_sorted),
        "rows_with_coords": len(geo_rows),
        "feature_groups": {
            "major_oxides": len(MAJOR_OXIDES),
            "trace_elements": len(TRACE_ELEMENTS),
            "endmembers": len(ENDMEMBERS),
            "total": len(ALL_FEATURES),
        },
        "completeness": {
            "major_all_present": major_complete,
            "trace_all_present": trace_complete,
            "endmember_all_present": endmember_complete,
        },
    },
    "class_distribution": class_counts_sorted,
    "missing_rates": {SHORT.get(k, k): v for k, v in missing_rates.items()},
    "missing_rates_raw_fe": {SHORT.get(k, k): v for k, v in missing_rates_raw_fe.items()},
    "missing_by_class": {
        SHORT.get(col, col): {SETTING_SHORT.get(ts, ts): v for ts, v in by_cls.items()}
        for col, by_cls in missing_by_class.items()
    },
    "feature_stats": {SHORT.get(k, k): v for k, v in feature_stats.items()},
    "class_oxide_stats": {
        SETTING_SHORT.get(ts, ts): {SHORT.get(col, col): stats for col, stats in cols.items()}
        for ts, cols in class_oxide_stats.items()
    },
    "discrimination": {SHORT.get(k, k): v for k, v in
                       sorted(discrimination.items(), key=lambda x: -x[1])},
    "correlation_matrix": {
        "labels": [SHORT.get(c, c) for c in corr_cols],
        "values": [[corr_matrix.get(ci, {}).get(cj, 0) for cj in corr_cols] for ci in corr_cols]
    },
    "xy_summary": {
        "major_miss_distribution": {str(k): v for k, v in sorted(major_miss_dist.items())},
        "trace_miss_distribution": {str(k): v for k, v in sorted(trace_miss_dist.items())},
        "class_balance_ratio": round(max(class_counts.values()) / max(1, min(class_counts.values())), 1),
        "largest_class": max(class_counts, key=class_counts.get),
        "smallest_class": min(class_counts, key=class_counts.get),
    },
    "models": model_results,
    "samples": sampled,
    "short_labels": SHORT,
    "setting_short": SETTING_SHORT,
}

# Write compact JSON
outpath = os.path.join(OUT_DIR, "viz_data.json")
with open(outpath, "w") as f:
    json.dump(output, f, separators=(",", ":"), ensure_ascii=False)

sz = os.path.getsize(outpath)
print(f"\nOutput: {outpath} ({sz/1024:.0f} KB)", flush=True)
print(f"Total time: {time.time()-t0:.1f}s", flush=True)
print("Done.", flush=True)
