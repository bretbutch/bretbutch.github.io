"""Draft figures and CSVs for paperwater.net: snowpack vs. river flow in three upper Rio Grande headwater basins.

Reads water_years.csv (from analyze.py) and lapuente_dv.rdb (USGS daily discharge, 08284100, WY1956+).
Writes everything to drafts/.
"""
import io
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

OUT = "drafts"
os.makedirs(OUT, exist_ok=True)
CFSDAY_AF = 1.983471

# Bar and trend-line colors sampled from the existing Del Norte chart. The two period colors were
# checked with the dataviz palette validator (colorblind separation, lightness, chroma, contrast on white).
BAR, NAVY, ORANGE = "#b8c4cb", "#1f4a62", "#bc7b2f"
EARLY, LATE = "#1d6194", ORANGE
INK, INK2, GRID = "#1a1a1a", "#555555", "#e4e8ea"

# Equal 20-year halves of the record all three basins share (Wolf Creek's first full year is WY1987).
FIRST, LAST = (1987, 2006), (2007, 2026)
PROVISIONAL = {2025, 2026}
LABEL_YEARS = [2002, 2013, 2026]
# Year labels are hand-placed (data units) in open space, with a leader line back to the dot
LABEL_POS = {
    "Rio Chama":  {2002: (14.0, 20), 2013: (19.0, 55), 2026: (4.0, 120)},
    "Conejos":    {2002: (13.5, 45), 2013: (22.0, 92), 2026: (6.0, 175)},
    "Rio Grande": {2002: (13.5, 150), 2013: (21.5, 300), 2026: (5.0, 520)},
}

BASINS = [  # key in water_years.csv, gage, snow index
    ("Rio Chama", "Rio Chama near La Puente, NM", "½ Cumbres Trestle + ½ Hopewell"),
    ("Conejos", "Conejos River near Mogote, CO", "½ Cumbres Trestle + ½ Wolf Creek Summit"),
    ("Rio Grande", "Rio Grande near Del Norte, CO", "½ Wolf Creek Summit + ½ Beartown"),
]
GAGE_ID = {"Rio Chama": "USGS 08284100", "Conejos": "USGS 08246500 / CO DWR CONMOGCO", "Rio Grande": "USGS 08220000 / CO DWR RIODELCO"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.facecolor": "white",
})


def footnote(fig, text, y=0.012):
    fig.text(0.012, y, text, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.4)


# ---------------------------------------------------------------- water-year table
w = pd.read_csv("water_years.csv")
w = w[w.wy.between(FIRST[0], LAST[1])].copy()
w["period"] = np.where(w.wy <= FIRST[1], f"{FIRST[0]}–{FIRST[1]}", f"{LAST[0]}–{LAST[1]}")
w["peak_swe_date"] = (pd.to_datetime((w.wy - 1).astype(str) + "-10-01") + pd.to_timedelta(w.peak_doy, unit="D")).dt.date
w["kaf_per_inch"] = w.wy_runoff_kaf / w.peak_swe
assert w[["peak_swe", "wy_runoff_kaf"]].notna().all().all(), "a basin is missing a year in the comparison period"

table = pd.DataFrame({
    "basin": w.basin, "gage": w.basin.map({b: g for b, g, _ in BASINS}), "gage_id": w.basin.map(GAGE_ID),
    "snow_index": w.basin.map({b: s for b, _, s in BASINS}), "water_year": w.wy, "period": w.period,
    "peak_swe_in": w.peak_swe.round(1), "peak_swe_date": w.peak_swe_date,
    "wy_flow_kaf": w.wy_runoff_kaf.round(1), "aprjul_flow_kaf": w.amjj_runoff_kaf.round(1),
    "kaf_per_inch_peak_swe": w.kaf_per_inch.round(2), "flow_provisional": w.wy.isin(PROVISIONAL),
}).sort_values(["basin", "water_year"])
table.to_csv(f"{OUT}/snow_to_flow_water_years.csv", index=False, encoding="utf-8-sig")  # BOM so Excel reads ½ and – correctly

rows = []
for b, g, s in BASINS:
    a, z = w[(w.basin == b) & (w.wy <= FIRST[1])], w[(w.basin == b) & (w.wy >= LAST[0])]
    rows.append({
        "basin": b, "gage": g, "snow_index": s,
        "peak_swe_median_first_in": round(a.peak_swe.median(), 1), "peak_swe_median_last_in": round(z.peak_swe.median(), 1),
        "wy_flow_median_first_kaf": round(a.wy_runoff_kaf.median(), 1), "wy_flow_median_last_kaf": round(z.wy_runoff_kaf.median(), 1),
        "kaf_per_inch_median_first": round(a.kaf_per_inch.median(), 2), "kaf_per_inch_median_last": round(z.kaf_per_inch.median(), 2),
        "years_first": len(a), "years_last": len(z),
    })
summ = pd.DataFrame(rows)
for k in ["peak_swe", "wy_flow", "kaf_per_inch"]:
    u = "in" if k == "peak_swe" else ("kaf" if k == "wy_flow" else "")
    f, l = [c for c in summ.columns if c.startswith(k + "_median_first")][0], [c for c in summ.columns if c.startswith(k + "_median_last")][0]
    summ[f"{k}_change_pct"] = (100 * (summ[l] / summ[f] - 1)).round(0)
summ.to_csv(f"{OUT}/snow_to_flow_summary.csv", index=False, encoding="utf-8-sig")  # BOM so Excel reads ½ and – correctly
print(summ[["basin", "peak_swe_median_first_in", "peak_swe_median_last_in", "peak_swe_change_pct",
            "wy_flow_median_first_kaf", "wy_flow_median_last_kaf", "wy_flow_change_pct",
            "kaf_per_inch_median_first", "kaf_per_inch_median_last", "kaf_per_inch_change_pct"]].to_string(index=False))

SOURCES = ("Peak SWE: NRCS SNOTEL daily snow water equivalent, the daily average of two sites per basin (Hopewell sits just outside "
           "the Chama drainage and stands for its Rio Brazos side).\nFlow: water-year (Oct–Sep) total at the gage, not adjusted for "
           "upstream reservoirs or diversions; USGS daily values, Colorado DWR for 2025–26 at the Colorado gages. WY2025–26 flows are provisional.")

# ---------------------------------------------------------------- 1. snow change vs flow change
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.25, wspace=0.42)
names = [b for b, _, _ in BASINS]
ypos = np.arange(len(names))[::-1]
for ax, key, unit, head in [
    (axes[0], "peak_swe", "in", "Peak snowpack (median peak SWE)"),
    (axes[1], "wy_flow", "kaf", "River flow at the gage (median water-year total)"),
]:
    first = summ[[c for c in summ.columns if c.startswith(key + "_median_first")][0]]
    last = summ[[c for c in summ.columns if c.startswith(key + "_median_last")][0]]
    decline = -summ[f"{key}_change_pct"]
    colors = [NAVY if b == "Rio Chama" else BAR for b in names]
    ax.barh(ypos, decline, height=0.58, color=colors, zorder=2)
    ax.set_ylim(-0.8, len(names) - 0.45)  # room for the value line under the bottom bar
    for y, d, f0, l0 in zip(ypos, decline, first, last):
        vals = f"{f0:.1f} → {l0:.1f} in" if unit == "in" else f"{f0:,.0f} → {l0:,.0f} kaf"
        ax.text(d + 0.8, y, f"−{d:.0f}%", va="center", ha="left", fontsize=11, fontweight="bold", color=INK)
        ax.text(d + 0.8, y - 0.36, vals, va="center", ha="left", fontsize=8.6, color=INK2)
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels([f"{v}%" for v in [0, 10, 20, 30, 40, 50]])
    ax.set_xlabel("Decline in median, 2007–2026 vs. 1987–2006", fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_title(head, fontsize=10.5, loc="left", pad=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(names, fontsize=10.5)
fig.suptitle("Snowpack vs. River Flow, Upper Rio Grande Headwaters, WY1987–2006 vs. WY2007–2026",
             x=0.012, ha="left", y=0.965, fontsize=13, fontweight="bold")
r = summ.set_index("basin")
fig.text(0.012, 0.865,
         f"Median peak snowpack fell {-r.loc['Rio Chama', 'peak_swe_change_pct']:.0f}% on the Rio Chama, "
         f"{-r.loc['Conejos', 'peak_swe_change_pct']:.0f}% on the Conejos and {-r.loc['Rio Grande', 'peak_swe_change_pct']:.0f}% on the upper Rio Grande. "
         f"Median river flow fell {-r.loc['Rio Chama', 'wy_flow_change_pct']:.0f}%, {-r.loc['Conejos', 'wy_flow_change_pct']:.0f}% "
         f"and {-r.loc['Rio Grande', 'wy_flow_change_pct']:.0f}%.",
         fontsize=10, color=INK)
footnote(fig, "Snow index → gage:  Rio Chama: ½ Cumbres Trestle + ½ Hopewell → La Puente.  Conejos: ½ Cumbres Trestle + ½ Wolf Creek Summit → Mogote.  "
              "Rio Grande: ½ Wolf Creek Summit + ½ Beartown → Del Norte.\n" + SOURCES)
fig.savefig(f"{OUT}/1_snow_vs_flow_change.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. flow vs peak snow, one dot per year
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharex=True)
fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.21, wspace=0.24)
for ax, (b, gage, snow) in zip(axes, BASINS):
    g = w[w.basin == b]
    for (lo, hi), col, lab in [(FIRST, EARLY, f"{FIRST[0]}–{FIRST[1]}"), (LAST, LATE, f"{LAST[0]}–{LAST[1]}")]:
        s = g[g.wy.between(lo, hi)]
        ax.scatter(s.peak_swe, s.wy_runoff_kaf, s=40, color=col, edgecolor="white", linewidth=0.8, zorder=3, label=lab)
    ymax = g.wy_runoff_kaf.max() * 1.12
    for y in LABEL_YEARS:
        r0 = g[g.wy == y].iloc[0]
        ax.annotate(str(y), (r0.peak_swe, r0.wy_runoff_kaf), xytext=LABEL_POS[b][y], textcoords="data",
                    fontsize=8.5, color=INK, va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=4))
    ax.set_xlim(0, 50)
    ax.set_ylim(0, ymax)
    ax.grid(color=GRID, zorder=0)
    ax.set_title(gage, fontsize=10.5, loc="left", pad=20)
    ax.text(0, 1.025, f"Snow: {snow}", transform=ax.transAxes, fontsize=8.6, color=INK2, va="bottom")
    ax.set_xlabel("Peak snow water equivalent (inches)", fontsize=9)
axes[0].set_ylabel("Water-year flow (thousand acre-feet)", fontsize=9)
axes[0].legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.2)
fig.suptitle("Water-Year Flow vs. Peak Snowpack, WY1987–2026",
             x=0.012, ha="left", y=0.965, fontsize=13, fontweight="bold")
fig.text(0.012, 0.895, "Each dot is one water year. In all three basins, recent years mostly fall among earlier years with similar "
                       "snowpacks; more of them sit at the low-snow end.", fontsize=10, color=INK)
footnote(fig, SOURCES)
fig.savefig(f"{OUT}/2_flow_vs_snow_by_year.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. La Puente long record
lines = [l for l in open("lapuente_dv.rdb") if not l.startswith("#")]
dv = pd.read_csv(io.StringIO("".join([lines[0]] + lines[2:])), sep="\t")
dv["date"] = pd.to_datetime(dv.datetime)
dv["q"] = pd.to_numeric(dv.iloc[:, 3], errors="coerce")
dv["prov"] = dv.iloc[:, 4].astype(str).str.startswith("P")
dv["wy"] = dv.date.dt.year + (dv.date.dt.month >= 10)
lp = dv.groupby("wy").agg(days=("q", "count"), cfs_days=("q", "sum"), provisional_days=("prov", "sum"))
lp["flow_kaf"] = lp.cfs_days * CFSDAY_AF / 1000
lp = lp[lp.days >= 365].loc[1956:2026]
lp[["flow_kaf", "days", "provisional_days"]].round({"flow_kaf": 1}).rename_axis("water_year").to_csv(f"{OUT}/la_puente_wy_flow_1956_2026.csv", encoding="utf-8-sig")

yrs, flow = lp.index.values, lp.flow_kaf.values
smooth = lowess(flow, yrs, frac=0.2, it=3, return_sorted=False)
ewma = pd.Series(flow, index=yrs).ewm(span=10, adjust=False).mean()
med = lp.flow_kaf.loc[1991:2020].median()
last = yrs[-1]

fig, ax = plt.subplots(figsize=(13.5, 6.6))
fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.17)
bars = ax.bar(yrs[:-1], flow[:-1], width=0.72, color=BAR, zorder=2, label="Annual water-year flow (KAF)")
ax.bar([last], [flow[-1]], width=0.72, color=INK, zorder=2)
med_line = ax.axhline(med, color=INK2, lw=1, ls=(0, (1, 2)), zorder=1, label=f"1991–2020 median ({med:.0f} KAF)")
loess_line, = ax.plot(yrs, smooth, color=NAVY, lw=2.4, zorder=3, label="LOESS, locally linear (frac=0.2, robust)")
ewma_line, = ax.plot(yrs, ewma.values, color=ORANGE, lw=2, ls=(0, (5, 3)), zorder=3, label="10-yr EWMA (trailing / causal only)")
ax.scatter([last], [smooth[-1]], color=NAVY, s=28, zorder=4)
ax.scatter([last], [ewma.values[-1]], color=ORANGE, s=28, zorder=4)
ax.annotate(f"WY{last}\n{flow[-1]:.0f} KAF", (last, flow[-1]), xytext=(last - 1.2, max(flow) * 1.02),
            fontsize=8.5, ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
ax.set_xlim(yrs[0] - 1.5, last + 1.5)
ax.set_ylim(0, max(flow) * 1.13)
ax.grid(axis="y", color=GRID, zorder=0)
ax.set_xlabel("Water Year", fontsize=11)
ax.set_ylabel("Annual flow at La Puente (thousand acre-feet)", fontsize=10.5)
legend_items = [loess_line, ewma_line, med_line, bars]  # same order as the Del Norte chart
ax.legend(legend_items, [a.get_label() for a in legend_items], loc="upper left", frameon=False, fontsize=9.5)
ax.set_title(f"Rio Chama near La Puente, NM — Total Gage Flow, WY{yrs[0]}–WY{last}", fontsize=13, pad=10)
footnote(fig, "USGS 08284100 Rio Chama near La Puente, NM. Upstream of Heron and El Vado reservoirs and of the San Juan–Chama Project inflow "
              "(Willow Creek); not adjusted for upstream irrigation diversions.\nWater-year (Oct 1–Sep 30) sums of daily mean discharge "
              f"(1 cfs-day = 1.983471 AF). Complete water years only. WY2025 partly and WY2026 fully provisional.")
fig.savefig(f"{OUT}/3_la_puente_wy_flow.png")
plt.close(fig)
print(f"La Puente: {len(lp)} complete years, median 1991–2020 {med:.0f} kaf, WY{last} {flow[-1]:.0f} kaf, missing:",
      sorted(set(range(1956, 2027)) - set(lp.index)))
