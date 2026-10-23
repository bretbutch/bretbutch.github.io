"""Draft charts: Elephant Butte evaporation, from Reclamation's published daily series only (no derived quantities).
  6_eb_evaporation_and_lake_area.png      annual evaporation and average lake surface area, WY1951–2025
  7_eb_same_storage_more_evaporation.png  evaporation, lake area and rate per acre against storage, early vs. recent
  8_eb_rate_and_temperature.png           evaporation per acre and April–September temperature, WY1951–2025

Run from the snotel folder:
    .venv/bin/python chain/eb_evaporation_charts.py
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sources import HERE, climdiv, season_mean, usbr_reservoir

OUT = HERE.parent / "drafts"
EARLY, LATE = (1951, 1979), (2003, 2025)  # the two low-lake eras, with overlapping storage ranges
PAIR_TOLERANCE = 0.15                      # matched pairs: average storage within 15%

# Same palette as site_charts.py: bar and line colors from the Del Norte chart, period colors validator-checked.
BAR, NAVY, ORANGE = "#b8c4cb", "#1f4a62", "#bc7b2f"
C_EARLY, C_LATE = "#1d6194", ORANGE
INK, INK2, GRID = "#1a1a1a", "#555555", "#e4e8ea"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.facecolor": "white",
})


def footnote(fig, text):
    fig.text(0.012, 0.012, text, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.4)


def title(fig, main, sub, y_sub=0.885):
    fig.suptitle(main, x=0.012, ha="left", y=0.965, fontsize=13, fontweight="bold")
    fig.text(0.012, y_sub, sub, fontsize=10, color=INK, va="top", linespacing=1.5)


# ---------------------------------------------------------------- data: Reclamation HydroData, Elephant Butte (site 1119)
d = pd.DataFrame({"evap": usbr_reservoir(1119, 25), "storage": usbr_reservoir(1119, 17), "area": usbr_reservoir(1119, 89)})
d = d.loc["1950-10-01":"2025-09-30"]
neg = int((d.evap < 0).sum())
spike = int(((d.evap > 5 * d.evap.rolling(31, center=True, min_periods=10).median()) & (d.evap > 500)).sum())
print(f"Daily evaporation check, WY1951–2025: {neg} negative values, {spike} isolated spikes (>5x the 31-day median and >500 AF)")
d["wy"] = d.index.year + (d.index.month >= 10)
g = d.groupby("wy")
yr = pd.DataFrame({"evaporation_kaf": g.evap.sum() / 1000, "avg_storage_kaf": g.storage.mean() / 1000,
                   "avg_area_acres": g.area.mean(), "rate_ft": (d.evap / d.area).groupby(d.wy).sum(),
                   "evap_days": g.evap.count(), "area_days": g.area.count()})
yr = yr[(yr.evap_days >= 360) & (yr.area_days >= 360)].drop(columns=["evap_days", "area_days"])
_, temps = climdiv("tmpcdv")
yr["apr_sep_temp_f"] = season_mean(temps, range(4, 10), ["NM5", "NM8"])
yr["era"] = np.where(yr.index <= EARLY[1], f"{EARLY[0]}–{EARLY[1]}",
                     np.where(yr.index >= LATE[0], f"{LATE[0]}–{LATE[1]}", f"{EARLY[1] + 1}–{LATE[0] - 1}"))
yr.round({"evaporation_kaf": 2, "avg_storage_kaf": 1, "avg_area_acres": 0, "rate_ft": 3, "apr_sep_temp_f": 2}) \
  .rename_axis("water_year").to_csv(OUT / "elephant_butte_evaporation_wy.csv", encoding="utf-8-sig")
e, l = yr.loc[EARLY[0]:EARLY[1]], yr.loc[LATE[0]:LATE[1]]

# ---------------------------------------------------------------- statistics quoted on the charts
r_area = yr.evaporation_kaf.corr(yr.avg_area_acres)
pairs = [(i, j) for i in e.index for j in l.index
         if abs(e.avg_storage_kaf[i] - l.avg_storage_kaf[j]) / ((e.avg_storage_kaf[i] + l.avg_storage_kaf[j]) / 2) < PAIR_TOLERANCE]
med = lambda col: 100 * (np.median([l.loc[j, col] / e.loc[i, col] for i, j in pairs]) - 1)
p_evap, p_area, p_rate = med("evaporation_kaf"), med("avg_area_acres"), med("rate_ft")
n_more = sum(l.evaporation_kaf[j] > e.evaporation_kaf[i] for i, j in pairs)
rate_up = 100 * (l.rate_ft.mean() / e.rate_ft.mean() - 1)
dT = l.apr_sep_temp_f.mean() - e.apr_sep_temp_f.mean()
sens = {}
for name, x in [("early", e), ("late", l)]:
    f = stats.linregress(x.apr_sep_temp_f, x.rate_ft)
    sens[name] = (100 * f.slope / x.rate_ft.mean(), f.rvalue, f.pvalue)
share_lo, share_hi = sorted(100 * dT * s[0] / rate_up for s in sens.values())
print(f"Evaporation vs lake area r = {r_area:.2f}")
print(f"Matched pairs: {len(pairs)}, recent higher in {n_more}; median evaporation {p_evap:+.0f}%, area {p_area:+.0f}%, rate {p_rate:+.0f}%")
print(f"Rate per acre {e.rate_ft.mean():.2f} -> {l.rate_ft.mean():.2f} ft/yr ({rate_up:+.0f}%); Apr–Sep temperature {dT:+.1f}°F")
print("Temperature sensitivity of the rate:", {k: f"{v[0]:+.1f}%/°F (r {v[1]:.2f}, p {v[2]:.3f})" for k, v in sens.items()},
      f"-> warming accounts for {share_lo:.0f}–{share_hi:.0f}% of the rise")

SOURCE = ("Reclamation HydroData, Elephant Butte Reservoir (site 1119): daily evaporation, storage and surface area, water years 1951–2025. "
          "WY2026 left out: its provisional values are incomplete.\nReclamation's evaporation is a computed estimate, not a direct measurement.")

# ---------------------------------------------------------------- 6. evaporation and lake area
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.065, right=0.985, top=0.85, bottom=0.13, hspace=0.32)
a1.bar(yr.index, yr.evaporation_kaf, width=0.75, color=BAR, zorder=2)
a1.set_ylabel("Evaporated\n(thousand acre-feet)", fontsize=9.5)
a1.set_title("Evaporation per water year", fontsize=10.5, loc="left", fontweight="bold")
a2.bar(yr.index, yr.avg_area_acres / 1000, width=0.75, color=NAVY, zorder=2)
a2.set_ylabel("Average lake area\n(thousand acres)", fontsize=9.5)
a2.set_title("Average lake surface area", fontsize=10.5, loc="left", fontweight="bold")
a2.set_xlabel("Water Year", fontsize=10)
for ax in (a1, a2):
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_xlim(yr.index.min() - 1, yr.index.max() + 1)
top = yr.evaporation_kaf.idxmax()
title(fig, "Elephant Butte Evaporation Follows the Size of the Lake, WY1951–2025",
      f"Each year's evaporation tracks the lake's average surface area (r = {r_area:.2f}). The most evaporated in WY{top} "
      f"({yr.evaporation_kaf.max():.0f} thousand acre-feet), when the lake was nearly full; the least in WY{yr.evaporation_kaf.idxmin()} "
      f"({yr.evaporation_kaf.min():.0f}).", y_sub=0.918)
footnote(fig, SOURCE)
fig.savefig(OUT / "6_eb_evaporation_and_lake_area.png")
plt.close(fig)

# ---------------------------------------------------------------- 7. same storage, more evaporation: area and rate
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4), sharex=True)
fig.subplots_adjust(left=0.055, right=0.985, top=0.74, bottom=0.25, wspace=0.28)
e, l = e.assign(avg_area_kac=e.avg_area_acres / 1000), l.assign(avg_area_kac=l.avg_area_acres / 1000)
panels = [("evaporation_kaf", "Evaporated (thousand acre-feet)", f"Evaporation: {p_evap:+.0f}%"),
          ("avg_area_kac", "Average lake area (thousand acres)", f"Lake area: {p_area:+.0f}%"),
          ("rate_ft", "Evaporation per acre (feet)", f"Per acre: {p_rate:+.0f}%")]
for ax, (col, ylab, head) in zip(axes, panels):
    for x, c, lab in [(e, C_EARLY, f"WY{EARLY[0]}–{EARLY[1]}"), (l, C_LATE, f"WY{LATE[0]}–{LATE[1]}")]:
        ax.scatter(x.avg_storage_kaf, x[col], s=38, color=c, edgecolor="white", linewidth=0.8, zorder=3, label=lab)
    ax.set_xlim(0, max(e.avg_storage_kaf.max(), l.avg_storage_kaf.max()) * 1.05)
    ax.set_ylim(0, max(e[col].max(), l[col].max()) * 1.15)
    ax.grid(color=GRID, zorder=0)
    ax.set_xlabel("Average storage in the water year (thousand acre-feet)", fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(head, fontsize=10.5, loc="left", fontweight="bold")
axes[0].legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.2)
title(fig, "Elephant Butte: Same Storage, More Evaporation, WY1951–1979 vs. WY2003–2025",
      f"At the same storage the lake now evaporates about {p_evap:.0f}% more. Sediment has spread the same water over about {p_area:.0f}% more surface,\n"
      f"and each acre loses about {p_rate:.0f}% more. Percentages are medians of all {len(pairs)} early/recent pairs of years with storage within "
      f"{100 * PAIR_TOLERANCE:.0f}%; the recent year evaporated more in all {n_more}.")
footnote(fig, "Each dot is one water year. The full-lake years of WY1980–2002 have no early counterparts at the same storage and are left out.\n" + SOURCE)
fig.savefig(OUT / "7_eb_same_storage_more_evaporation.png")
plt.close(fig)

# ---------------------------------------------------------------- 8. rate per acre and temperature
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.065, right=0.985, top=0.81, bottom=0.13, hspace=0.32)
a1.bar(yr.index, yr.rate_ft, width=0.75, color=BAR, zorder=2)
for x, c in [(e, C_EARLY), (l, C_LATE)]:
    a1.plot([x.index.min() - 0.4, x.index.max() + 0.4], [x.rate_ft.mean()] * 2, color=c, lw=2.4, zorder=3)
    a1.text(x.index.min(), x.rate_ft.mean() + 0.25, f"{x.rate_ft.mean():.1f} ft/yr", color=INK, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)
a1.set_ylabel("Feet of water\nper acre per year", fontsize=9.5)
a1.set_title("Evaporation per acre of lake surface", fontsize=10.5, loc="left", fontweight="bold")
a1.set_ylim(0, yr.rate_ft.max() * 1.18)
a2.plot(yr.index, yr.apr_sep_temp_f, color=INK2, lw=1.2, marker="o", ms=3, zorder=2)
for x, c in [(e, C_EARLY), (l, C_LATE)]:
    a2.plot([x.index.min() - 0.4, x.index.max() + 0.4], [x.apr_sep_temp_f.mean()] * 2, color=c, lw=2.4, zorder=3)
    a2.text(x.index.min(), x.apr_sep_temp_f.mean() + 0.25, f"{x.apr_sep_temp_f.mean():.1f}°F", color=INK, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)
a2.set_ylabel("April–September\naverage, °F", fontsize=9.5)
a2.set_title("Growing-season temperature (NOAA, New Mexico divisions 5 and 8)", fontsize=10.5, loc="left", fontweight="bold")
a2.set_xlabel("Water Year", fontsize=10)
for ax in (a1, a2):
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_xlim(yr.index.min() - 1, yr.index.max() + 1)
title(fig, "Elephant Butte: Evaporation per Acre and Growing-Season Temperature, WY1951–2025",
      f"Each acre now loses about {rate_up:.0f}% more water a year than in WY{EARLY[0]}–{EARLY[1]}, as April–September warmed {dT:.1f}°F. "
      f"Warmer years evaporate more in both eras\n({sens['early'][0]:.1f}% and {sens['late'][0]:.1f}% per °F), "
      f"enough to account for roughly {5 * round(share_lo / 5):.0f}–{5 * round(share_hi / 5):.0f}% of the rise.", y_sub=0.925)
footnote(fig, "Evaporation per acre = daily evaporation ÷ Reclamation's daily lake surface area, summed over the water year. Colored lines: "
              f"era averages (WY{EARLY[0]}–{EARLY[1]}, WY{LATE[0]}–{LATE[1]}).\n" + SOURCE)
fig.savefig(OUT / "8_eb_rate_and_temperature.png")
plt.close(fig)
print("Wrote charts 6, 7 and 8 to", OUT)
