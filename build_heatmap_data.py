"""
Export ALL 211K coordinates as compact arrays for heatmap visualization.
Also export per-setting coordinate arrays for colored density rendering.
Output: web/data/geo_all.json (~4MB)
"""
import csv, json, os, time

t0 = time.time()
CSV_PATH = "/home/laz/proj/geochemical-research/clinopyroxene_data.csv"
OUT = "/home/laz/proj/geochemical-research/web/data/geo_all.json"

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

print("Loading CSV...", flush=True)
# Collect: all_points for heatmap, per-setting points for colored view
all_pts = []       # [lat, lon] for heatmap
by_setting = {}    # setting -> [[lat, lon], ...]
counts = {}

with open(CSV_PATH, encoding="latin-1") as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        ts = (r.get("TECTONIC SETTING") or "").strip()
        if not ts:
            continue
        try:
            lat = round(float(r.get("LATITUDE (MIN.)", "")), 3)
            lon = round(float(r.get("LONGITUDE (MIN.)", "")), 3)
        except (ValueError, TypeError):
            continue
        all_pts.append([lat, lon])
        short = SETTING_SHORT.get(ts, ts)
        if short not in by_setting:
            by_setting[short] = []
        by_setting[short].append([lat, lon])
        counts[short] = counts.get(short, 0) + 1
        if (i + 1) % 50000 == 0:
            print(f"  {i+1} rows, {len(all_pts)} with coords...", flush=True)

print(f"Total points: {len(all_pts):,}", flush=True)
for s, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c:,}", flush=True)

# Output compact JSON
output = {
    "all": all_pts,
    "by_setting": by_setting,
    "counts": counts,
}

with open(OUT, "w") as f:
    json.dump(output, f, separators=(",", ":"))

sz = os.path.getsize(OUT)
print(f"\nOutput: {OUT} ({sz/1024/1024:.1f} MB)")
print(f"Time: {time.time()-t0:.1f}s")
