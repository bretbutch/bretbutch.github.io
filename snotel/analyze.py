"""Water-year summaries: peak SWE, WY / cold-season precip, runoff volumes and efficiencies, plus Sen/Kendall trends."""
import numpy as np, pandas as pd
from scipy import stats

AREA_SQMI = {"Rio Chama": 480, "Conejos": 282, "Rio Grande": 1320}
CFSDAY_AF = 1.98347

d = pd.read_csv("daily.csv", parse_dates=["date"])
d["wy"] = d.date.dt.year + (d.date.dt.month >= 10)
d = d[(d.wy >= 1980) & (d.wy <= 2026)]

rows = []
for (basin, wy), g in d.groupby(["basin", "wy"]):
    g = g.set_index("date").sort_index()
    swe, prec, q = g.WTEQ, g.PREC, g.Q
    ondjfm = g[g.index < f"{wy}-04-01"]
    amjj = q[(q.index >= f"{wy}-04-01") & (q.index <= f"{wy}-07-31")]
    ok_swe = swe[g.index < f"{wy}-07-01"].count() >= 250
    ok_q = q.count() >= 360
    pk_date = swe.idxmax() if ok_swe else pd.NaT
    # PREC is accumulated from Oct 1; take value at a date, falling back to the nearest prior reading
    p_at = lambda t: prec[:t].dropna().iloc[-1] if prec[:t].count() else np.nan
    rows.append(dict(
        basin=basin, wy=wy,
        peak_swe=swe.max() if ok_swe else np.nan,
        peak_doy=(pk_date - pd.Timestamp(f"{wy-1}-10-01")).days if ok_swe else np.nan,
        apr1_swe=swe.get(pd.Timestamp(f"{wy}-04-01"), np.nan),
        wy_prec=p_at(f"{wy}-09-30") if prec.count() > 330 else np.nan,
        cold_prec=p_at(f"{wy}-03-31") if ondjfm.PREC.count() > 160 else np.nan,
        prec_to_peak=p_at(pk_date) if ok_swe else np.nan,
        wy_runoff_kaf=q.sum() * CFSDAY_AF / 1000 if ok_q else np.nan,
        amjj_runoff_kaf=amjj.sum() * CFSDAY_AF / 1000 if amjj.count() >= 120 else np.nan,
    ))
w = pd.DataFrame(rows)
af_per_in = w.basin.map(AREA_SQMI) * 640 / 12  # acre-feet per inch of runoff over the drainage
w["wy_runoff_in"] = w.wy_runoff_kaf * 1000 / af_per_in
w["amjj_runoff_in"] = w.amjj_runoff_kaf * 1000 / af_per_in
w["swe_frac_cold"] = w.peak_swe / w.cold_prec          # how much of Oct–Mar precip is held as peak snowpack
w["swe_frac_topeak"] = w.peak_swe / w.prec_to_peak     # snowpack retained of precip that fell before peak
w["swe_frac_wy"] = w.peak_swe / w.wy_prec
w["runoff_per_swe"] = w.wy_runoff_in / w.peak_swe       # basin runoff depth per inch of index-site peak SWE
w["amjj_per_swe"] = w.amjj_runoff_in / w.peak_swe
w["runoff_per_prec"] = w.wy_runoff_in / w.wy_prec
w.round(3).to_csv("water_years.csv", index=False)

def trend(x, y):
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(y) < 15: return None
    s = stats.theilslopes(y, x)
    t = stats.kendalltau(x, y)
    return dict(n=len(y), mean=y.mean(), sen_per_decade=s.slope * 10, pct_per_decade=100 * s.slope * 10 / y.mean(), p=t.pvalue)

cols = ["peak_swe", "peak_doy", "wy_prec", "cold_prec", "swe_frac_cold", "swe_frac_topeak", "swe_frac_wy",
        "wy_runoff_kaf", "amjj_runoff_kaf", "runoff_per_swe", "amjj_per_swe", "runoff_per_prec"]
out = []
for b, g in w.groupby("basin"):
    for c in cols:
        r = trend(g.wy, g[c])
        if r: out.append(dict(basin=b, metric=c, **r))
    # residual approach: does runoff for a given peak SWE decline over time?
    m = g[["peak_swe", "wy_runoff_kaf", "wy"]].dropna()
    fit = stats.linregress(m.peak_swe, m.wy_runoff_kaf)
    r = trend(m.wy, m.wy_runoff_kaf - (fit.intercept + fit.slope * m.peak_swe))
    out.append(dict(basin=b, metric=f"runoff_resid_vs_swe (r2={fit.rvalue**2:.2f}, {fit.slope:.1f} kaf/in)", **r))
    m = g[["wy_prec", "wy_runoff_kaf", "wy"]].dropna()
    fit = stats.linregress(m.wy_prec, m.wy_runoff_kaf)
    out.append(dict(basin=b, metric=f"runoff_resid_vs_prec (r2={fit.rvalue**2:.2f})",
                    **trend(m.wy, m.wy_runoff_kaf - (fit.intercept + fit.slope * m.wy_prec))))
t = pd.DataFrame(out)
t.round(3).to_csv("trends.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(t.round(3).to_string(index=False))
