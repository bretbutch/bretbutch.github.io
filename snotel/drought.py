"""Drought-year comparison: daily SWE/flow vs 1991–2020 normals, Apr–Jul response, leave-one-out expectations."""
import json, math
import numpy as np, pandas as pd

YEARS = [2002, 2013, 2026]
NORMALS = (1991, 2020)
CFSDAY_AF = 1.98347

d = pd.read_csv("daily.csv", parse_dates=["date"])
d["wy"] = d.date.dt.year + (d.date.dt.month >= 10)
d["dowy"] = (d.date - pd.to_datetime((d.wy - 1).astype(str) + "-10-01")).dt.days
d = d[d.dowy < 365]
w = pd.read_csv("water_years.csv")

def clean(x, nd=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)

out = {"years": YEARS, "normals": NORMALS, "basins": {}}
for b, g in d.groupby("basin"):
    norm = g[g.wy.between(*NORMALS)]
    clim = {}
    for col in ["WTEQ", "Q"]:
        q = norm.groupby("dowy")[col].quantile([.1, .5, .9]).unstack()
        # light 7-day smoothing so the band reads as a climate envelope, not daily noise
        q = q.rolling(7, center=True, min_periods=1).mean()
        clim[col] = {k: [clean(v, 1) for v in q[p]] for k, p in (("p10", .1), ("p50", .5), ("p90", .9))}
    # Nov–Feb mean flow as an antecedent (baseflow/groundwater) proxy
    base = g[g.date.dt.month.isin([11, 12, 1, 2])].groupby("wy").Q.mean()
    base_med = base.loc[NORMALS[0]:NORMALS[1]].median()

    wb = w[w.basin == b].set_index("wy")
    nrm = wb.loc[NORMALS[0]:NORMALS[1]]
    med = nrm[["peak_swe", "cold_prec", "wy_prec", "amjj_runoff_kaf", "wy_runoff_kaf"]].median()
    allyrs = []
    for wy, r in wb.iterrows():
        if pd.isna(r.peak_swe) or pd.isna(r.amjj_runoff_kaf):
            continue
        allyrs.append({"wy": int(wy), "swe_pct": clean(100 * r.peak_swe / med.peak_swe, 0),
                       "amjj_pct": clean(100 * r.amjj_runoff_kaf / med.amjj_runoff_kaf, 0)})
    fit_pool = wb[["peak_swe", "amjj_runoff_kaf"]].dropna()
    years = {}
    for y in YEARS:
        gy = g[g.wy == y].sort_values("dowy")
        r = wb.loc[y]
        # leave-one-out linear fit of Apr–Jul runoff on peak SWE
        pool = fit_pool.drop(index=y, errors="ignore")
        k, c = np.polyfit(pool.peak_swe, pool.amjj_runoff_kaf, 1)
        expected = k * r.peak_swe + c
        swe = gy.set_index("dowy").WTEQ
        pk = swe.idxmax()
        after = swe.loc[pk:]
        melt_out = after[after <= 0.05].index.min() if (after <= 0.05).any() else None
        years[y] = {
            "swe": [clean(v, 1) for v in swe.reindex(range(365))],
            "q": [clean(v, 0) for v in gy.set_index("dowy").Q.reindex(range(365))],
            "peak_swe": clean(r.peak_swe, 1), "peak_doy": clean(r.peak_doy, 0),
            "melt_out_doy": None if melt_out is None or pd.isna(melt_out) else int(melt_out),
            "swe_pct": clean(100 * r.peak_swe / med.peak_swe, 0),
            "cold_prec_pct": clean(100 * r.cold_prec / med.cold_prec, 0),
            "wy_prec_pct": clean(100 * r.wy_prec / med.wy_prec, 0),
            "amjj_kaf": clean(r.amjj_runoff_kaf, 1), "amjj_pct": clean(100 * r.amjj_runoff_kaf / med.amjj_runoff_kaf, 0),
            "wy_pct": clean(100 * r.wy_runoff_kaf / med.wy_runoff_kaf, 0),
            "amjj_expected": clean(expected, 1), "amjj_vs_expected": clean(100 * r.amjj_runoff_kaf / expected, 0),
            "base_pct": clean(100 * base.get(y, np.nan) / base_med, 0),
        }
    out["basins"][b] = {"clim": clim, "years": years, "all": allyrs,
                        "medians": {k: clean(v, 1) for k, v in med.items()}}
json.dump(out, open("drought_data.json", "w"), separators=(",", ":"))
for b, v in out["basins"].items():
    print(b, v["medians"])
    for y, r in v["years"].items():
        print(f"  {y} SWE {r['swe_pct']}% | AMJJ {r['amjj_pct']}% | vs expected {r['amjj_vs_expected']}% ({r['amjj_kaf']} vs {r['amjj_expected']}) | base {r['base_pct']}% | peak {r['peak_doy']} melt-out {r['melt_out_doy']}")
