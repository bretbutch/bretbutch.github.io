"""Draft charts for paperwater.net:
  4_cascade_expected_vs_delivered.png  what each gage's snowpack predicts vs. what arrived, headwaters to Elephant Butte
(Elephant Butte evaporation charts: eb_evaporation_charts.py)

Run from the snotel folder (needs snotel_sites.csv from fetch.py):
    .venv/bin/python chain/cascade_charts.py
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from sources import CACHE, HERE, flows_wy, snow_index, usbr_reservoir

OUT = HERE.parent / "drafts"
OUT.mkdir(exist_ok=True)
BASE, LATE_YRS = (1987, 2006), (2007, 2026)

# Same palette as site_charts.py: bar and line colors from the Del Norte chart, period colors validator-checked.
BAR, NAVY, ORANGE = "#b8c4cb", "#1f4a62", "#bc7b2f"
EARLY, LATE = "#1d6194", ORANGE
INK, INK2, GRID = "#1a1a1a", "#555555", "#e4e8ea"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.facecolor": "white",
})


def footnote(fig, text):
    fig.text(0.012, 0.012, text, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.4)


def signed(v, decimals=0, comma=False):
    """+12 / −34 with a true minus sign."""
    s = f"{v:+,.{decimals}f}" if comma else f"{v:+.{decimals}f}"
    return s.replace("-", "−")


# ================================================================ data
flows = flows_wy(["DelNorte", "Mogote", "Lobatos", "Embudo", "Otowi", "Albuquerque", "SanMarcialFloodway", "SanMarcialLFCC"])
flows["COhead"] = flows.DelNorte + flows.Mogote
snow = snow_index()

eb = pd.DataFrame({"storage": usbr_reservoir(1119, 17), "evap": usbr_reservoir(1119, 25), "release": usbr_reservoir(1119, 43)})

# ================================================================ expected vs delivered
NODES = [("COhead", "Colorado headwaters\n(Del Norte + Mogote)"), ("Lobatos", "Lobatos\n(state line)"),
         ("Embudo", "Embudo"), ("Otowi", "Otowi\n(after the Rio Chama joins)"), ("Albuquerque", "Albuquerque"),
         ("SanMarcial", "San Marcial\n(into Elephant Butte)")]
rows = []
for key, label in NODES:
    d = pd.concat([snow, flows[key].rename("q")], axis=1).dropna().loc[BASE[0]:LATE_YRS[1]]
    b, l = d.loc[BASE[0]:BASE[1]], d.loc[LATE_YRS[0]:LATE_YRS[1]]
    slope, intercept = np.polyfit(b.snow, b.q, 1)
    resid_sd = np.sqrt(((b.q - (intercept + slope * b.snow)) ** 2).sum() / (len(b) - 2))
    expected = intercept + slope * l.snow.mean()
    # spread of a 20-year average around the expectation if the 1987–2006 relationship still held
    se = resid_sd * np.sqrt(1 / len(l) + 1 / len(b) + (l.snow.mean() - b.snow.mean()) ** 2 / ((b.snow - b.snow.mean()) ** 2).sum())
    half = stats.t.ppf(0.95, len(b) - 2) * se
    observed = l.q.mean()
    rows.append({"node": key, "label": label.replace("\n", " "), "expected_kaf": expected, "range90_low": expected - half,
                 "range90_high": expected + half, "delivered_kaf": observed, "gap_kaf": observed - expected,
                 "gap_pct": 100 * (observed / expected - 1), "outside_range": abs(observed - expected) > half,
                 "r2_1987_2006": np.corrcoef(b.snow, b.q)[0, 1] ** 2, "years_base": len(b), "years_late": len(l)})
casc = pd.DataFrame(rows)
casc.round(2).to_csv(OUT / "cascade_expected_vs_delivered.csv", index=False, encoding="utf-8-sig")

evap_wy = (eb.evap.groupby(eb.index.year + (eb.index.month >= 10)).agg(["sum", "count"]))
evap_wy = (evap_wy["sum"] / 1000).where(evap_wy["count"] >= 360)
evap_late = evap_wy.loc[LATE_YRS[0]:LATE_YRS[1]].mean()
sm_late = casc.set_index("node").loc["SanMarcial", "delivered_kaf"]
print(casc[["node", "expected_kaf", "range90_low", "range90_high", "delivered_kaf", "gap_kaf", "gap_pct", "outside_range", "r2_1987_2006"]].round(1).to_string(index=False))
print(f"Elephant Butte evaporation WY{LATE_YRS[0]}–{LATE_YRS[1]}: {evap_late:.0f} kaf/yr = {100 * evap_late / sm_late:.0f}% of San Marcial")

eb["wy"] = eb.index.year + (eb.index.month >= 10)
g = eb.groupby("wy")

# reservoir balance check: storage change + releases + evaporation vs. what passed San Marcial
sep30 = eb.storage[(eb.index.month == 9) & (eb.index.day == 30)]
dS = sep30.diff()
dS.index = dS.index.year
bal = pd.DataFrame({"storage_change": dS / 1000, "release": g.release.sum() / 1000, "evap": g.evap.sum() / 1000,
                    "release_days": g.release.count(), "evap_days": g.evap.count()})
bal = bal[(bal.release_days >= 360) & (bal.evap_days >= 360)]
bal["reached_pool"] = bal.storage_change + bal.release + bal.evap
bal = bal.join(flows.SanMarcial.rename("san_marcial")).loc[1988:2026].dropna()
bal["pool_minus_san_marcial"] = bal.reached_pool - bal.san_marcial
bal.round(1).to_csv(CACHE / "elephant_butte_water_balance_wy.csv")
print(f"Balance check WY1988–2026: reached the pool minus San Marcial, mean {bal.pool_minus_san_marcial.mean():+.0f} kaf/yr, "
      f"positive in {int((bal.pool_minus_san_marcial > 0).sum())} of {len(bal)} years")

# ================================================================ chart 4: cascade
fig, ax = plt.subplots(figsize=(13.5, 7.4))
fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.24)
x = np.arange(len(casc))
w = 0.36
ax.bar(x - w / 2, casc.expected_kaf, w, facecolor=to_rgba(EARLY, 0.14), edgecolor=EARLY, linewidth=1.6, zorder=2)
ax.errorbar(x - w / 2, casc.expected_kaf, yerr=[casc.expected_kaf - casc.range90_low, casc.range90_high - casc.expected_kaf],
            fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
ax.bar(x + w / 2, casc.delivered_kaf, w, color=LATE, zorder=2)
for xi, r in zip(x, casc.itertuples()):
    top = max(r.range90_high, r.delivered_kaf)
    ax.annotate(f"{signed(r.gap_kaf, comma=True)} kaf/yr", (xi, top), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.6, color=INK2)
    ax.annotate(f"{signed(r.gap_pct)}%", (xi, top), xytext=(0, 18), textcoords="offset points",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
xe = len(casc) + 0.75
ax.bar([xe], [evap_late], w * 1.2, color=NAVY, zorder=2)
ax.annotate(f"{evap_late:.0f} kaf/yr\n≈{100 * evap_late / sm_late:.0f}% of what\nreached San Marcial", (xe, evap_late),
            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8.6, color=INK2, linespacing=1.3)
ax.axvline(len(casc) + 0.18, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.set_xticks(list(x) + [xe])
ax.set_xticklabels([lab for _, lab in NODES] + ["Evaporated from\nElephant Butte\n(Reclamation estimate)"], fontsize=9.2)
ax.tick_params(axis="x", length=0, pad=6)
ax.set_xlim(-0.6, xe + 0.6)
ax.set_ylim(0, casc[["range90_high", "delivered_kaf"]].max().max() * 1.12)
ax.set_ylabel("Thousand acre-feet per year (average, WY2007–2026)", fontsize=10)
ax.grid(axis="y", color=GRID, zorder=0)
fig.legend(handles=[Patch(facecolor=to_rgba(EARLY, 0.14), edgecolor=EARLY, linewidth=1.6, label="Expected from that year's snowpack (1987–2006 relationship)"),
                    Patch(color=LATE, label="Delivered (gaged)"),
                    Patch(color=NAVY, label="Evaporated (Reclamation estimate)")],
           loc="upper left", frameon=False, fontsize=9.2, ncol=3, bbox_to_anchor=(0.058, 0.845))
c = casc.set_index("node")
SHORT = {"COhead": "the Colorado headwaters", "Lobatos": "Lobatos", "Embudo": "Embudo", "Otowi": "Otowi",
         "Albuquerque": "Albuquerque", "SanMarcial": "San Marcial"}
beyond = [SHORT[k] for k in c.index[c.outside_range]]
fig.suptitle("Snowpack vs. Delivered Flow, Upper Rio Grande to Elephant Butte, WY2007–2026",
             x=0.012, ha="left", y=0.965, fontsize=13, fontweight="bold")
sub = (f"The Colorado headwaters still deliver what their snowpack predicts. Downstream the shortfall grows: "
       f"{-c.loc['Otowi', 'gap_pct']:.0f}% at Otowi, {-c.loc['Albuquerque', 'gap_pct']:.0f}% at Albuquerque, "
       f"{-c.loc['SanMarcial', 'gap_pct']:.0f}% at San Marcial.")
if beyond:
    sub += f"\nAt {' and '.join(beyond)} the gap is outside the range that year-to-year chance alone would produce (whiskers)."
fig.text(0.012, 0.918, sub, fontsize=10, color=INK, va="top", linespacing=1.5)
footnote(fig, "Expected: each gage's water-year flow fit against peak snow water equivalent (daily mean of Cumbres Trestle, Wolf Creek Summit and Beartown) "
              "for WY1987–2006, applied to WY2007–2026 snowpacks.\nWhiskers: where the 20-year average would fall 90% of the time if the "
              "1987–2006 relationship still held (assumes independent years). Gaged flows, not adjusted for reservoirs, diversions or\n"
              "San Juan–Chama imports (Otowi and below include them; median imports were about the same in both periods). San Marcial = "
              "floodway + Low Flow Conveyance Channel. Sources: NRCS SNOTEL,\nUSGS, Colorado DWR (2025–26 at the Colorado gages), "
              "Reclamation HydroData. WY2025–26 provisional.")
fig.savefig(OUT / "4_cascade_expected_vs_delivered.png")
plt.close(fig)

print("\nWrote", OUT / "4_cascade_expected_vs_delivered.png")
