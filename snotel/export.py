"""Bundle water-year table + trend fits into JSON for the chart page."""
import json, math
import pandas as pd
from scipy import stats

w = pd.read_csv("water_years.csv")
meta = {
    "Rio Chama":  {"station": "Cumbres", "sites": "Cumbres Trestle · 10,030 ft", "gage": "Rio Chama nr La Puente", "area": 480},
    "Conejos":    {"station": "Cumbres + Wolf Ck", "sites": "½ Cumbres Trestle + ½ Wolf Creek Summit", "gage": "Conejos R nr Mogote", "area": 282},
    "Rio Grande": {"station": "Wolf Ck + Beartown", "sites": "½ Wolf Creek Summit + ½ Beartown", "gage": "Rio Grande nr Del Norte", "area": 1320},
}
metrics = ["peak_swe", "peak_doy", "wy_prec", "cold_prec", "swe_frac_cold", "swe_frac_wy",
           "wy_runoff_kaf", "amjj_runoff_kaf", "runoff_per_swe", "amjj_per_swe", "runoff_per_prec"]
out = {"meta": meta, "rows": [], "fits": {}}
for r in w.to_dict("records"):
    out["rows"].append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
for b, g in w.groupby("basin"):
    out["fits"][b] = {}
    for m in metrics:
        d = g[["wy", m]].dropna()
        s = stats.theilslopes(d[m], d.wy)
        out["fits"][b][m] = {"slope": s.slope, "intercept": s.intercept, "p": stats.kendalltau(d.wy, d[m]).pvalue,
                             "n": len(d), "mean": d[m].mean(), "x0": int(d.wy.min()), "x1": int(d.wy.max())}
    d = g[["peak_swe", "wy_runoff_kaf"]].dropna()
    f = stats.linregress(d.peak_swe, d.wy_runoff_kaf)
    out["fits"][b]["scatter"] = {"slope": f.slope, "intercept": f.intercept, "r2": f.rvalue ** 2}
json.dump(out, open("chart_data.json", "w"), separators=(",", ":"))
print(len(json.dumps(out)))
