"""Extract lat/lon + tectonic setting + oxides from CSV, output lightweight JSON for web map."""
import csv, json, sys, random, os

os.makedirs('/home/laz/proj/geochemical-research/web/data', exist_ok=True)

rows = []
oxides = ['SIO2(WT%)','TIO2(WT%)','AL2O3(WT%)','CAO(WT%)','MGO(WT%)','NA2O(WT%)','FEOT(WT%)','CR2O3(WT%)']

with open('/home/laz/proj/geochemical-research/clinopyroxene_data.csv', encoding='latin-1') as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        ts = (r.get('TECTONIC SETTING') or '').strip()
        if not ts:
            continue
        try:
            lat = float(r.get('LATITUDE (MIN.)') or '')
            lon = float(r.get('LONGITUDE (MIN.)') or '')
        except (ValueError, TypeError):
            continue
        rec = {'lat': round(lat, 4), 'lon': round(lon, 4), 'ts': ts}
        loc = (r.get('LOCATION') or '').strip()
        rock = (r.get('ROCK NAME') or '').strip()
        if loc: rec['loc'] = loc[:80]
        if rock: rec['rock'] = rock[:60]
        for ox in oxides:
            try: rec[ox.split('(')[0].lower()] = round(float(r.get(ox, '')), 2)
            except (ValueError, TypeError): pass
        rows.append(rec)
        if (i + 1) % 50000 == 0:
            print(f'  Processed {i+1} rows, {len(rows)} with coords...', flush=True)

print(f'Total with coords: {len(rows)}', flush=True)

by_ts = {}
for r in rows:
    by_ts.setdefault(r['ts'], []).append(r)

sampled = []
random.seed(42)
for ts_name in sorted(by_ts):
    recs = by_ts[ts_name]
    n = min(len(recs), max(50, int(20000 * len(recs) / len(rows))))
    sampled.extend(random.sample(recs, n))
    print(f'  {ts_name}: {len(recs)} -> {n}', flush=True)
print(f'Sampled: {len(sampled)}', flush=True)

model_results = {
    "baseline": {"name": "Baseline (median impute)", "features": 14, "rows": 177947, "acc": 0.8369, "f1m": 0.7316},
    "native_nan": {"name": "Native NaN (XGBoost)", "features": 50, "rows": 203179, "acc": 0.8631, "f1m": 0.7717},
    "subset_M1": {"name": "Subset M1 (major miss\u22641)", "features": 18, "rows": 99840, "acc": 0.8722, "f1m": 0.8216},
    "subset_M2T0": {"name": "Subset M2_T0 (full trace)", "features": 50, "rows": 1195, "acc": 0.9749, "f1m": 0.9717},
    "hierarchical": {"name": "Hierarchical (S1+S2)", "features": "18+32", "rows": 42355, "acc": 0.8352, "f1m": 0.7422},
}

output = {"samples": sampled, "models": model_results,
          "class_counts": {ts: len(recs) for ts, recs in sorted(by_ts.items())}}

outpath = '/home/laz/proj/geochemical-research/web/data/samples.json'
with open(outpath, 'w') as f:
    json.dump(output, f, separators=(',', ':'))
sz = os.path.getsize(outpath)
print(f'Output: {sz/1024:.0f}KB', flush=True)
