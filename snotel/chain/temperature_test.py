"""Temperature test for the Rio Grande cascade: do warmer seasons deliver less water for the same snow and rain?

Run from the snotel folder (needs snotel_sites.csv from fetch.py):
    .venv/bin/python chain/temperature_test.py

Choices set before looking at results:
  Snow           mean of Cumbres Trestle, Wolf Creek Summit and Beartown daily SWE; peak per water year.
  Temperature    NOAA nClimDiv April–September mean over CO division 5 (Rio Grande basin), NM division 2
                 (Northern Mountains) and NM division 5 (Central Valley).
  Precipitation  water-year (Oct–Sep) total over the same divisions. Hot years are dry years, so without this
                 the "heat" effect would partly be a "less rain" effect.
  Shared trend   the heat effect is re-estimated from year-to-year swings only (temperature detrended, year as a
                 covariate), because warming and every other slow change happened over the same decades.
  Years          complete water years from 1987 (first full Wolf Creek season) with April–September temperatures.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sources import CACHE, climdiv, flows_wy, season_mean, season_total, snow_index

DIVS = ["CO5", "NM2", "NM5"]

f = flows_wy(["DelNorte", "Mogote", "Lobatos", "LaPuente", "Otowi", "Albuquerque", "SanMarcialFloodway", "SanMarcialLFCC"])
f["CO_head"] = f.DelNorte + f.Mogote
f["Headwaters"] = f.DelNorte + f.Mogote + f.LaPuente

tname, t = climdiv("tmpcdv")
pname, p = climdiv("pcpndv")
clim = pd.DataFrame({"T": season_mean(t, range(4, 10), DIVS), "Pwy": season_total(p, range(1, 13), DIVS)})

d = f.join(snow_index()).join(clim).loc[1987:].dropna(subset=["snow", "T", "Pwy"])
d["wy"] = d.index
d["Tdt"] = d["T"] - np.polyval(np.polyfit(d.wy, d["T"], 1), d.wy)
print(f"NOAA files: {tname}, {pname}. Water years used: WY{d.wy.min()}–{d.wy.max()} (n={len(d)})")

half = 2007  # same split as the site drafts: WY1987–2006 vs WY2007 on
e, l = d.wy < half, d.wy >= half
dT = d.loc[l, "T"].mean() - d.loc[e, "T"].mean()
print(f"Apr–Sep temperature: {d.loc[e, 'T'].mean():.2f}°F (WY{d.wy.min()}–{half - 1}) → {d.loc[l, 'T'].mean():.2f}°F (WY{half}–{d.wy.max()}), "
      f"+{dT:.2f}°F; trend {10 * np.polyfit(d.wy, d['T'], 1)[0]:.2f}°F/decade")
print(f"Correlation of Apr–Sep temperature with water-year precipitation: {d['T'].corr(d['Pwy']):.2f}")


def fp(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


print("\nHeat effect at each gage, % of mean flow per °F (p):")
rows = []
for g in ["Headwaters", "DelNorte", "LaPuente", "Lobatos", "Otowi", "Albuquerque", "SanMarcial"]:
    x = d.dropna(subset=[g])
    simple = smf.ols(f"{g} ~ snow + T", x).fit()
    strict = smf.ols(f"{g} ~ snow + T + Pwy", x).fit()
    yy = smf.ols(f"{g} ~ snow + Tdt + Pwy + wy", x).fit()
    m = x[g].mean()
    rows.append({"gage": g, "snow only": f"{100 * simple.params['T'] / m:.1f} ({fp(simple.pvalues['T'])})",
                 "snow + precip": f"{100 * strict.params['T'] / m:.1f} ({fp(strict.pvalues['T'])})",
                 "snow + precip, year-to-year only": f"{100 * yy.params['Tdt'] / m:.1f} ({fp(yy.pvalues['Tdt'])})"})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))

print(f"\nChange in mean water-year flow, second half vs first half (kaf/yr), split by cause:")
rows = []
for g in ["Headwaters", "Lobatos", "Otowi", "Albuquerque", "SanMarcial"]:
    x = d.dropna(subset=[g])
    m = smf.ols(f"{g} ~ snow + T + Pwy", x).fit()
    m2 = smf.ols(f"{g} ~ snow + Tdt + Pwy + wy", x).fit()
    xe, xl = x.wy < half, x.wy >= half
    dq = x.loc[xl, g].mean() - x.loc[xe, g].mean()
    sn = m.params["snow"] * (x.loc[xl, "snow"].mean() - x.loc[xe, "snow"].mean())
    pr = m.params["Pwy"] * (x.loc[xl, "Pwy"].mean() - x.loc[xe, "Pwy"].mean())
    lo, hi = sorted([m2.params["Tdt"] * dT, m.params["T"] * dT])
    rows.append({"gage": g, "total": round(dq), "less snow": round(sn), "less precip": round(pr),
                 "warmer": f"{round(hi)} to {round(lo)}", "not explained": f"{round(dq - sn - pr - hi)} to {round(dq - sn - pr - lo)}"})
print(pd.DataFrame(rows).to_string(index=False))

print("\nReach by reach (outflow ~ inflow + temperature + precipitation):")
rows = []
for name, out, inflow in [("San Luis Valley: headwaters → Lobatos", "Lobatos", "CO_head"),
                          ("northern NM + Chama: Lobatos + La Puente → Otowi", "Otowi", "Lobatos + LaPuente"),
                          ("Otowi → Albuquerque", "Albuquerque", "Otowi"),
                          ("Albuquerque → San Marcial", "SanMarcial", "Albuquerque")]:
    ins = [c.strip() for c in inflow.split("+")]
    x = d.dropna(subset=[out] + ins)
    m = smf.ols(f"{out} ~ {inflow} + T + Pwy", x).fit()
    m2 = smf.ols(f"{out} ~ {inflow} + Tdt + Pwy + wy", x).fit()
    rows.append({"reach": name, "heat kaf/°F (p)": f"{m.params['T']:.0f} ({fp(m.pvalues['T'])})",
                 "heat, year-to-year (p)": f"{m2.params['Tdt']:.0f} ({fp(m2.pvalues['Tdt'])})",
                 "decline not tied to heat, kaf/decade (p)": f"{10 * m2.params['wy']:.0f} ({fp(m2.pvalues['wy'])})"})
print(pd.DataFrame(rows).to_string(index=False))
d.to_csv(CACHE / "temperature_test_data.csv")
