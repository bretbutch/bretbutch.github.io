"""MRGCD diversions from the district's OneRain telemetry (history starts Dec 2023), pulled month by month.

Run from the snotel folder:  .venv/bin/python chain/mrgcd_annual.py
"""
import time
from pathlib import Path
import pandas as pd
import requests

s = requests.Session(); s.headers["User-Agent"] = "paperwater-river-pipeline/1.0 (+https://paperwater.net)"
s.get("https://mrgcd.onerain.com/", timeout=30)
SITES = {"MRGCD total": (195, 2), "Cochiti": (198, 2), "Angostura": (77, 3), "Isleta": (197, 2), "San Acacia": (131, 3)}

def pull_month(sid, dev, start):
    end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    p = {"method": "sensorDetails", "site_id": sid, "device_id": dev, "site": sid, "device": dev,
         "data_start": start.strftime("%Y-%m-%d %H:%M:%S"), "data_end": end.strftime("%Y-%m-%d %H:%M:%S"),
         "range": 2, "time_zone": "US/Mountain"}
    for attempt in range(4):
        r = s.get("https://mrgcd.onerain.com/export/flot/", params=p, timeout=180)
        try:
            pts = r.json()[0]["data"]
            return pd.Series([v for _, v in pts], index=pd.to_datetime([t for t, _ in pts], unit="ms"), dtype=float)
        except ValueError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{sid}/{dev} {start:%Y-%m}: HTTP {r.status_code} {r.text[:120]!r}")

daily = {}
for name, (sid, dev) in SITES.items():
    parts = [pull_month(sid, dev, m) for m in pd.date_range("2023-01-01", pd.Timestamp.today().normalize() - pd.offsets.MonthBegin(1), freq="MS")]
    x = pd.concat(parts).sort_index()
    x = x[~x.index.duplicated()].clip(lower=0)
    # time-weighted daily mean: 15-min bins, carry a reading forward up to 2 h across telemetry gaps
    daily[name] = x.resample("15min").mean().ffill(limit=8).resample("D").mean()
    print(f"{name}: {x.index.min():%Y-%m-%d} → {x.index.max():%Y-%m-%d}, {len(x)} readings", flush=True)

d = pd.DataFrame(daily)
(Path(__file__).resolve().parent / "cache").mkdir(exist_ok=True)
d.to_csv(Path(__file__).resolve().parent / "cache" / "mrgcd_daily_cfs.csv")
yr = d.groupby(d.index.year)
ann = (yr.sum() * 1.983471 / 1000).round(1)
cnt = yr.count()
print("\nCalendar-year diversion, thousand acre-feet (days with data in parentheses):")
print(pd.DataFrame({c: [f"{ann.loc[y, c]:.1f} ({cnt.loc[y, c]})" for y in ann.index] for c in ann.columns}, index=ann.index).to_string())
heads = ["Cochiti", "Angostura", "Isleta", "San Acacia"]
print("\nSum of the four headings vs. the district total sensor:")
print(pd.DataFrame({"four headings": ann[heads].sum(axis=1), "total sensor": ann["MRGCD total"]}).to_string())
