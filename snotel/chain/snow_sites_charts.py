"""Draft charts for the four SNOTEL sites behind the snow indices:
  9_snowpack_through_the_season.png          typical snowpack by date, WY1987–2006 vs. WY2007–2026, with half-gone dates
  10_snotel_temperature_raw_vs_corrected.png annual temperature 1991–2025: raw SNOTEL, sensor jump removed, and NOAA

Run from the snotel folder:
    .venv/bin/python chain/snow_sites_charts.py
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sources import HERE, climdiv, snotel_daily

OUT = HERE.parent / "drafts"
EARLY, LATE = (1987, 2006), (2007, 2026)
SITES = [  # north to south: name, triplet, elevation (ft), NOAA climate division used as the temperature reference
    ("Beartown", "327:CO:SNTL", 11580, "CO5"),
    ("Wolf Creek Summit", "874:CO:SNTL", 10930, "CO5"),
    ("Cumbres Trestle", "431:CO:SNTL", 10030, "CO5"),
    ("Hopewell", "532:NM:SNTL", 10080, "NM2"),
]
DIV_NAME = {"CO5": "Colorado Rio Grande basin", "NM2": "New Mexico Northern Mountains"}

BAR, NAVY, ORANGE = "#b8c4cb", "#1f4a62", "#bc7b2f"
C_EARLY, C_LATE = "#1d6194", ORANGE
INK, INK2, GRID, RAW = "#1a1a1a", "#555555", "#e4e8ea", "#9aa6ae"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.facecolor": "white",
})


def footnote(fig, text):
    fig.text(0.012, 0.012, text, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.4)


def wy_day_to_date(day):
    t = pd.Timestamp("2001-10-01") + pd.Timedelta(days=int(round(day)))
    return f"{t:%b} {t.day}"


# ================================================================ melt timing
def season(swe, wy):
    """One water year of daily SWE (Oct 1 – Aug 31), indexed by day of water year; None if under 90% complete."""
    x = swe.loc[f"{wy - 1}-10-01":f"{wy}-08-31"]
    if x.count() < 0.9 * 335:
        return None
    return pd.Series(x.values, index=(x.index - pd.Timestamp(f"{wy - 1}-10-01")).days)


timing, curves = [], {}
for name, trip, elev, _ in SITES:
    raw = snotel_daily(trip, "WTEQ")
    smooth = raw.rolling(3, center=True, min_periods=1).median()   # knocks down one-day sensor spikes
    per_year = {}
    for wy in range(EARLY[0], LATE[1] + 1):
        x = season(smooth, wy)
        if x is None or x.max() < 2:
            continue
        per_year[wy] = x
        pk_day, pk = x.idxmax(), x.max()
        after = x.loc[pk_day:]
        timing.append({"station": name, "water_year": wy, "peak_swe_in": round(pk, 1), "peak_day": pk_day,
                       "half_gone_day": after[after <= 0.5 * pk].index.min(), "melt_out_day": after[after <= 0.1].index.min()})
    curves[name] = pd.DataFrame(per_year)
T = pd.DataFrame(timing)
for c in ["peak_day", "half_gone_day", "melt_out_day"]:
    T[c.replace("_day", "_date")] = [wy_day_to_date(v) if pd.notna(v) else "" for v in T[c]]
T.to_csv(OUT / "snotel_melt_timing_wy.csv", index=False, encoding="utf-8-sig")

summary = []
for name, *_ in SITES:
    t = T[T.station == name]
    e, l = t[t.water_year <= EARLY[1]], t[t.water_year >= LATE[0]]
    summary.append({"station": name, "peak_in": (e.peak_swe_in.median(), l.peak_swe_in.median()),
                    "peak": (e.peak_day.median(), l.peak_day.median()), "half": (e.half_gone_day.median(), l.half_gone_day.median()),
                    "out": (e.melt_out_day.median(), l.melt_out_day.median())})
    s = summary[-1]
    print(f"{name}: peak {s['peak_in'][0]:.1f} -> {s['peak_in'][1]:.1f} in; peak date {wy_day_to_date(s['peak'][0])} -> {wy_day_to_date(s['peak'][1])}; "
          f"half gone {wy_day_to_date(s['half'][0])} -> {wy_day_to_date(s['half'][1])}; melt-out {wy_day_to_date(s['out'][0])} -> {wy_day_to_date(s['out'][1])}")
shifts = [s["half"][1] - s["half"][0] for s in summary]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), sharex=True)
fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.12, hspace=0.42, wspace=0.16)
DAYS = np.arange(31, 305)  # Nov 1 – Jul 31
for ax, (name, trip, elev, _), s in zip(axes.flat, SITES, summary):
    c = curves[name]
    for (lo, hi), col, lab, key in [(EARLY, C_EARLY, f"WY{EARLY[0]}–{EARLY[1]}", 0), (LATE, C_LATE, f"WY{LATE[0]}–{LATE[1]}", 1)]:
        yrs = [y for y in c.columns if lo <= y <= hi]
        med = c[yrs].reindex(DAYS).median(axis=1)
        ax.plot(DAYS, med, color=col, lw=2.2, zorder=3, label=lab)
        hd = s["half"][key]
        ax.plot([hd, hd], [0, med.loc[int(round(hd))]], color=col, lw=1.1, ls=(0, (3, 2)), zorder=2)
        ax.scatter([hd], [med.loc[int(round(hd))]], s=36, color=col, edgecolor="white", linewidth=0.8, zorder=4)
    shift = s["half"][1] - s["half"][0]
    ax.text(0.015, 0.95, f"Peak: {s['peak_in'][0]:.1f} → {s['peak_in'][1]:.1f} in\n"
                         f"Half gone: {wy_day_to_date(s['half'][0])} → {wy_day_to_date(s['half'][1])} "
                         f"({abs(shift):.0f} days {'earlier' if shift < 0 else 'later'})",
            transform=ax.transAxes, va="top", fontsize=9, color=INK, linespacing=1.5)
    ax.set_title(f"{name} ({elev:,} ft)", fontsize=10.5, loc="left", fontweight="bold")
    top = max(c[[y for y in c.columns if EARLY[0] <= y <= LATE[1]]].reindex(DAYS).median(axis=1).max(), 1)
    ax.set_ylim(0, max(ax.get_ylim()[1], top) * 1.28)
    ax.set_ylabel("Snow water equivalent (in)", fontsize=9)
    ax.grid(color=GRID, zorder=0)
ticks = [31, 62, 92, 123, 151, 182, 212, 243, 273, 304]
for ax in axes[1]:
    ax.set_xticks(ticks)
    ax.set_xticklabels([wy_day_to_date(d)[:3] for d in ticks])
axes[0, 1].legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle("Snowpack Through the Season at Four High Sites, WY1987–2006 vs. WY2007–2026",
             x=0.012, ha="left", y=0.965, fontsize=13, fontweight="bold")
fig.text(0.012, 0.915, f"Typical snow water equivalent on each date. In recent years the snowpack peaks lower and is half gone "
                       f"{abs(max(shifts)):.0f}–{abs(min(shifts)):.0f} days earlier. Dots mark the half-gone date.",
         fontsize=10, color=INK, va="top")
footnote(fig, "Curves: the median snow water equivalent on each date across the years in each period. Half gone: the first day after each year's peak "
              "when snow water equivalent falls to half of that peak;\nthe dates shown are medians of those yearly dates. Part of the earlier melt comes "
              "from smaller snowpacks; comparing years with similar snowpacks, only Wolf Creek Summit melts clearly earlier.\n"
              "NRCS SNOTEL daily snow water equivalent (3-day median to remove one-day sensor spikes). WY2025–26 provisional.")
fig.savefig(OUT / "9_snowpack_through_the_season.png")
plt.close(fig)

# ================================================================ temperature: find and remove the sensor jump
# Window starts in 1991: Hopewell (May 1991) and Beartown (Apr 1990) shift sharply in their early records, before the
# network's sensor replacement, so earlier years aren't comparable. Inside the window, one jump per site is enough.
TEMP_YEARS = (1991, 2025)
_, t = climdiv("tmpcdv")
div = t.assign(date=pd.to_datetime(dict(year=t.year, month=t.month, day=1))).pivot(index="date", columns="div", values="v")


def pettitt(x):
    n = len(x)
    U = np.cumsum([np.sign(x[k] - x).sum() for k in range(n)])
    k = int(np.argmax(np.abs(U[:-1])))
    return k, 2 * np.exp(-6 * U[k] ** 2 / (n ** 3 + n ** 2))


temp_rows, results = [], {}
for name, trip, elev, dv in SITES:
    s = snotel_daily(trip, "TAVG")
    s = s.where(s.between(-40, 90))
    m = s.resample("MS").mean().where(s.resample("MS").count() >= 25)
    d = pd.DataFrame({"raw": m, "ref": div[dv]}).dropna().loc[f"{TEMP_YEARS[0]}-01-01":f"{TEMP_YEARS[1]}-12-31"]
    d["diff"] = d.raw - d.ref
    anom = (d["diff"] - d.groupby(d.index.month)["diff"].transform("mean")).values
    k, p = pettitt(anom)
    brk = d.index[k]
    before, after = d.iloc[:k + 1], d.iloc[k + 1:]
    step = after.groupby(after.index.month)["diff"].mean() - before.groupby(before.index.month)["diff"].mean()
    jump = p <= 1e-3 and min(len(before), len(after)) >= 24   # keep only statistically clear jumps
    d["corrected"] = d.raw - (np.where(d.index > brk, step.reindex(d.index.month).values, 0) if jump else 0)
    full = d.groupby(d.index.year).size() >= 11
    ann = d.groupby(d.index.year)[["raw", "corrected", "ref"]].mean()[full]
    trend = {c: 10 * stats.theilslopes(ann[c], ann.index).slope for c in ["raw", "corrected", "ref"]}
    results[name] = dict(ann=ann, trend=trend, jump=step.mean() if jump else None, jump_month=brk + pd.offsets.MonthBegin(1), p=p,
                         largest=step.mean())
    for yv, r in ann.iterrows():
        temp_rows.append({"station": name, "year": yv, "raw_snotel_f": round(r.raw, 2), "corrected_f": round(r.corrected, 2),
                          "noaa_division_f": round(r.ref, 2), "noaa_division": dv})
    print(f"{name}: " + (f"jump {step.mean():+.1f}°F starting {brk + pd.offsets.MonthBegin(1):%b %Y} (p ≈ {p:.0e})" if jump else "no clear jump")
          + f"; trends {TEMP_YEARS[0]}–{TEMP_YEARS[1]}: raw {trend['raw']:+.2f}, corrected {trend['corrected']:+.2f}, NOAA {trend['ref']:+.2f} °F/decade")
pd.DataFrame(temp_rows).to_csv(OUT / "snotel_temperature_annual.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), sharex=True)
fig.subplots_adjust(left=0.06, right=0.985, top=0.745, bottom=0.15, hspace=0.45, wspace=0.16)
for ax, (name, trip, elev, dv) in zip(axes.flat, SITES):
    r = results[name]
    a = r["ann"]
    noaa = a.ref - a.ref.mean() + a.corrected.mean()   # NOAA's year-to-year record, shifted to the site's average level
    ax.plot(a.index, a.raw, color=RAW, lw=1.2, marker="o", ms=3, zorder=2, label="Raw SNOTEL")
    ax.plot(a.index, a.corrected, color=NAVY, lw=2, marker="o", ms=3.5, zorder=3, label="Sensor jump removed")
    ax.plot(a.index, noaa, color=ORANGE, lw=1.4, ls=(0, (4, 2)), zorder=2, label="NOAA region (shifted to the site's level)")
    if r["jump"] is not None:
        jy = r["jump_month"].year + (r["jump_month"].month - 0.5) / 12
        ax.axvline(jy, color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=1)
        note = f"sensor jump {r['jump']:+.1f}°F from {r['jump_month']:%b %Y} (dotted line)"
    else:
        note = "no clear jump: raw and corrected lines overlap"
    ax.set_title(note, fontsize=8.8, loc="right", color=INK2)
    tr = r["trend"]
    ax.set_title(f"{name} ({elev:,} ft)", fontsize=10.5, loc="left", fontweight="bold")
    ax.text(0.015, 0.95, f"Trend, °F per decade:  raw {tr['raw']:+.2f}   corrected {tr['corrected']:+.2f}   NOAA {tr['ref']:+.2f}",
            transform=ax.transAxes, fontsize=9, color=INK, va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=5)
    lo, hi = min(a.raw.min(), a.corrected.min(), noaa.min()), max(a.raw.max(), a.corrected.max(), noaa.max())
    ax.set_ylim(lo - 1, hi + 2.2)
    ax.set_ylabel("Annual average, °F", fontsize=9)
    ax.grid(axis="y", color=GRID, zorder=0)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.828), ncol=3, frameon=False, fontsize=9.2)
jumped = [n for n, *_ in SITES if results[n]["jump"] is not None]
steady = [n for n, *_ in SITES if results[n]["jump"] is None]
ct = [results[n]["trend"]["corrected"] for n, *_ in SITES]
nt = [results[n]["trend"]["ref"] for n, *_ in SITES]
fig.suptitle(f"Temperature at Four High Snow Sites, {TEMP_YEARS[0]}–{TEMP_YEARS[1]}: Raw SNOTEL vs. Sensor Jump Removed",
             x=0.012, ha="left", y=0.965, fontsize=13, fontweight="bold")
jump_list = [f"{n} ({results[n]['jump']:+.1f}°F, {results[n]['jump_month']:%Y})" for n in jumped]
fig.text(0.012, 0.92, (", ".join(jump_list[:-1]) + " and " + jump_list[-1] if len(jump_list) > 1 else jump_list[0])
         + " jumped warmer, in step with the network's temperature-sensor replacement;\n"
         + (f"{' and '.join(steady)} {'shows' if len(steady) == 1 else 'show'} no clear jump. " if steady else "")
         + f"With the jumps removed, the four sites warmed about {min(ct):.2f}–{max(ct):.2f}°F per decade, "
         f"close to NOAA's regional record ({min(nt):.2f}–{max(nt):.2f}).",
         fontsize=10, color=INK, va="top", linespacing=1.5)
footnote(fig, "Jump: the largest shift in each site's monthly temperature relative to NOAA's homogenized climate-division record (Colorado Rio Grande basin; "
              "New Mexico Northern Mountains for Hopewell),\nkept only if statistically clear (p ≤ 0.001), then removed month by month after that date (Beartown's largest shift, +0.2°F, isn't). "
              f"Small shifts are hard to date precisely (Cumbres: 2007 here, 2012 in its full record). Starts in {TEMP_YEARS[0]}:\nHopewell and Beartown "
              "shift sharply around 1990–91, before the sensor replacement. Because the fix uses NOAA's record as the yardstick, it can't show whether the "
              "summits warmed faster than the region.\nAnnual averages of months with 25+ days of data; years with 11+ months. NRCS SNOTEL daily average "
              "temperature; NOAA nClimDiv.")
fig.savefig(OUT / "10_snotel_temperature_raw_vs_corrected.png")
plt.close(fig)
print("Wrote charts 9 and 10 to", OUT)
