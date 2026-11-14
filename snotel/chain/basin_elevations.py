"""How high each headwater basin sits, where the snow sites sit in and around those basins, and whether the
10,000-foot snow sites (Cumbres Trestle, Hopewell) have lost ground against the 11,000-foot ones (Wolf Creek
Summit, Beartown). Numbers for the Headwaters page's elevation notes; writes drafts/basin_elevations.csv.

Basin outlines: USGS NLDI upstream basins for each gage. Terrain: AWS Terrain Tiles (terrarium PNGs, zoom 10,
about 120 m). Both are cached in chain/cache/ without expiry, since neither changes.

Run from the snotel folder:
    .venv/bin/python chain/basin_elevations.py
"""
import io
import json
import math

import numpy as np
import pandas as pd
from matplotlib.path import Path as Poly
from PIL import Image
from scipy import stats

from sources import CACHE, HERE, SESSION, snotel_daily

OUT = HERE.parent / "drafts"
EARLY, LATE = (1987, 2006), (2007, 2026)
ZOOM = 10
SITES = {  # name: (triplet, elevation ft, lon, lat), from NRCS station metadata
    "Cumbres Trestle": ("431:CO:SNTL", 10030, -106.45275, 37.01877),
    "Hopewell": ("532:NM:SNTL", 10080, -106.2637, 36.71632),
    "Wolf Creek Summit": ("874:CO:SNTL", 10930, -106.80234, 37.47903),
    "Beartown": ("327:CO:SNTL", 11580, -107.5124, 37.71433),
}
BASINS = {  # USGS site: (name, snow sites in its index; None for the sub-basins used only for site locations)
    "08284100": ("Rio Chama near La Puente", ["Cumbres Trestle", "Hopewell"]),
    "08246500": ("Conejos River near Mogote", ["Cumbres Trestle", "Wolf Creek Summit"]),
    "08220000": ("Rio Grande near Del Norte", ["Wolf Creek Summit", "Beartown"]),
    "08282300": ("Rio Brazos at Fishtail Road", None),
    "08281400": ("Rio Chama above Chama", None),
}


def basin_polygons(site):
    path = CACHE / f"nldi_basin_{site}.geojson"
    if not path.exists():
        r = SESSION.get(f"https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}/basin", timeout=180)
        r.raise_for_status()
        path.write_text(r.text)
    geom = json.loads(path.read_text())["features"][0]["geometry"]
    return [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]


def tile(x, y):
    path = CACHE / "terrarium" / f"{ZOOM}_{x}_{y}.png"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        r = SESSION.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{x}/{y}.png", timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    a = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB")).astype(float)
    return (a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768) * 3.28084  # meters -> feet


def inside(polys, pts):
    m = np.zeros(len(pts), bool)
    for p in polys:
        k = Poly(np.array(p[0])).contains_points(pts)
        for hole in p[1:]:
            k &= ~Poly(np.array(hole)).contains_points(pts)
        m |= k
    return m


def hypsometry(polys):
    """Elevation (ft) and ground area (m²) of every terrain pixel inside the basin."""
    n = 2 ** ZOOM
    lon_lat = np.vstack([np.array(p[0]) for p in polys])
    x0 = int((lon_lat[:, 0].min() + 180) / 360 * n)
    x1 = int((lon_lat[:, 0].max() + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    y0, y1 = ty(lon_lat[:, 1].max()), ty(lon_lat[:, 1].min())
    res = 2 * math.pi * 6378137.0 / (256 * n)  # web-mercator pixel size at the equator (m)
    elev, area = [], []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            lon = ((x * 256 + np.arange(256) + 0.5) / (256 * n)) * 360 - 180
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y * 256 + np.arange(256) + 0.5) / (256 * n)))))
            LON, LAT = np.meshgrid(lon, lat)
            m = inside(polys, np.column_stack([LON.ravel(), LAT.ravel()]))
            elev.append(tile(x, y).ravel()[m])
            area.append(((res * np.cos(np.radians(LAT.ravel()))) ** 2)[m])
    return np.concatenate(elev), np.concatenate(area)


def km_to_edge(polys, lon, lat):
    best = np.inf
    for p in polys:
        r = np.array(p[0])
        t = np.linspace(0, 1, 20)[:, None]
        xs, ys = r[:-1, 0] + t * (r[1:, 0] - r[:-1, 0]), r[:-1, 1] + t * (r[1:, 1] - r[:-1, 1])
        best = min(best, np.hypot((xs - lon) * 111.0 * math.cos(math.radians(lat)), (ys - lat) * 111.0).min())
    return best


# ================================================================ basin elevations
rows = []
for site, (name, pair) in BASINS.items():
    if pair is None:
        continue
    e, a = hypsometry(basin_polygons(site))
    below = lambda v: 100 * a[e < v].sum() / a.sum()
    rows.append({"basin": name, "usgs_site": site, "area_sq_mi": round(a.sum() / 2.59e6), "mean_elevation_ft": round(np.average(e, weights=a)),
                 **{f"pct_above_{k}k_ft": round(100 - below(k * 1000), 1) for k in (9, 10, 11, 12)},
                 **{f"pct_below_{s.replace(' ', '_')}": round(below(SITES[s][1])) for s in pair}})
    r = rows[-1]
    print(f"{name}: {r['area_sq_mi']:,} sq mi, mean {r['mean_elevation_ft']:,} ft; above 10k {r['pct_above_10k_ft']}%, above 11k {r['pct_above_11k_ft']}%; "
          + ", ".join(f"{s} {SITES[s][1]:,} ft is higher than {below(SITES[s][1]):.0f}% of the basin" for s in pair))
pd.DataFrame(rows).to_csv(OUT / "basin_elevations.csv", index=False, encoding="utf-8-sig")

# ================================================================ which snow sites sit inside which basins
print("\nSnow sites vs. basin outlines (distance to the outline; outlines are simplified, so treat <0.5 km as on the line):")
for site, (name, _) in BASINS.items():
    polys = basin_polygons(site)
    print(f"  {name}: " + "; ".join(
        f"{s} {'inside' if inside(polys, np.array([[lon, lat]]))[0] else 'outside'} {km_to_edge(polys, lon, lat):.1f} km"
        for s, (_, _, lon, lat) in SITES.items()))

# ================================================================ 10,000-ft pair vs. 11,000-ft pair
peaks = {}
for s, (trip, *_) in SITES.items():
    d = snotel_daily(trip, "WTEQ")
    wy = d.index.year + (d.index.month >= 10)
    before_july = d.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6])
    ok = d[before_july].groupby(wy[before_july]).count() >= 250  # same completeness rule as the basin indices
    peaks[s] = d.groupby(wy).max().where(ok)
pk = pd.DataFrame(peaks).loc[EARLY[0]:LATE[1]].dropna()
print(f"\nPeak snowpack by site, WY{EARLY[0]}–{EARLY[1]} vs. WY{LATE[0]}–{LATE[1]} ({len(pk)} years with all four sites):")
for s in SITES:
    a, b = pk.loc[:EARLY[1], s], pk.loc[LATE[0]:, s]
    print(f"  {s}: median {a.median():.1f} -> {b.median():.1f} in ({100 * (b.median() / a.median() - 1):+.0f}%), mean {100 * (b.mean() / a.mean() - 1):+.0f}%")
low, high = pk[["Cumbres Trestle", "Hopewell"]].mean(axis=1), pk[["Wolf Creek Summit", "Beartown"]].mean(axis=1)
ratio = low / high
print(f"  10,000-ft pair vs. 11,000-ft pair: rank correlation {stats.spearmanr(low, high)[0]:.2f}; lower pair as a share of the higher: "
      f"median {ratio.loc[:EARLY[1]].median():.2f} -> {ratio.loc[LATE[0]:].median():.2f}, "
      f"Sen slope {10 * stats.theilslopes(ratio.values, ratio.index.values).slope:+.3f}/decade, Kendall p = {stats.kendalltau(ratio.index, ratio.values)[1]:.2f}")
