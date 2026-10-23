"""Shared data access for the Rio Grande chain analyses: gage flows, the three-site snow index, Reclamation
reservoir records and NOAA climate divisions. Downloads are cached in chain/cache/ for the day.
"""
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
CFSDAY_AF = 1.983471
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "paperwater-river-pipeline/1.0 (+https://paperwater.net)"

GAGES = {  # name: (USGS site, Colorado DWR abbrev used from 2025 on, where USGS daily values end)
    "DelNorte": ("08220000", "RIODELCO"), "Mogote": ("08246500", "CONMOGCO"), "Lobatos": ("08251500", "RIOLOBCO"),
    "Embudo": ("08279500", None), "LaPuente": ("08284100", None), "Otowi": ("08313000", None),
    "Albuquerque": ("08330000", None), "SanMarcialFloodway": ("08358400", None), "SanMarcialLFCC": ("08358300", None),
}
CLIMDIV = {"0505": "CO5", "2902": "NM2", "2905": "NM5", "2908": "NM8"}


def _cached(key, fetch):
    """Reuse today's download if there is one; otherwise fetch and save it."""
    path = CACHE / f"{key}.csv"
    if path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today():
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s = fetch().rename("value")
    s.to_csv(path)
    return s


def _usgs_daily(site):
    url = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
    params = {"monitoring_location_id": f"USGS-{site}", "parameter_code": "00060", "statistic_id": "00003",
              "time": "1986-10-01/..", "f": "json", "limit": 10000, "properties": "time,value"}
    rows = []
    while url:
        r = SESSION.get(url, params=params, timeout=180)
        r.raise_for_status()
        j = r.json()
        rows += [f["properties"] for f in j["features"]]
        url, params = next((l["href"] for l in j["links"] if l.get("rel") == "next"), None), None
    df = pd.DataFrame(rows)
    return pd.Series(pd.to_numeric(df.value, errors="coerce").values, index=pd.to_datetime(df.time)).sort_index()


def _dwr_daily(abbrev):
    r = SESSION.get("https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewatertsday/",
                    params={"format": "json", "abbrev": abbrev, "min-measDate": "01/01/2025", "pageSize": 50000}, timeout=180)
    r.raise_for_status()
    return pd.Series({pd.Timestamp(x["measDate"][:10]): x["value"] for x in r.json()["ResultList"]}, dtype=float)


def gage_daily(name):
    """Daily mean flow (cfs). Colorado gages switch to Colorado DWR's reviewed record where USGS daily values end."""
    site, abbrev = GAGES[name]
    q = _cached(f"usgs_{site}", lambda: _usgs_daily(site))
    if abbrev:
        d = _cached(f"dwr_{abbrev}", lambda: _dwr_daily(abbrev))
        q = pd.concat([q, d[d.index > q.index.max()]])
    return q


def water_years(q):
    """Water-year totals in thousand acre-feet; years with fewer than 360 days are left blank."""
    wy = q.index.year + (q.index.month >= 10)
    g = q.groupby(wy)
    return (g.sum() * CFSDAY_AF / 1000).where(g.count() >= 360)


def flows_wy(names):
    f = pd.DataFrame({n: water_years(gage_daily(n)) for n in names})
    if {"SanMarcialFloodway", "SanMarcialLFCC"} <= set(names):
        f["SanMarcial"] = f.SanMarcialFloodway + f.SanMarcialLFCC
    return f


def snow_index():
    """Peak of the daily mean SWE of Cumbres Trestle, Wolf Creek Summit and Beartown, by water year (inches)."""
    s = pd.read_csv(HERE.parent / "snotel_sites.csv", parse_dates=["date"]).pivot(index="date", columns="station", values="WTEQ")
    idx = s[["Cumbres Trestle", "Wolf Creek Summit", "Beartown"]].mean(axis=1, skipna=False)  # all three sites must report
    sd = pd.DataFrame({"swe": idx, "wy": idx.index.year + (idx.index.month >= 10)})
    ok = sd[sd.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6])].groupby("wy").swe.count() >= 250
    return sd.groupby("wy").swe.max().where(ok).rename("snow")


def snotel_daily(triplet, element):
    """NRCS SNOTEL daily values (e.g. 'WTEQ' snow water equivalent, 'TAVG' average air temperature) from WY1981 on."""
    def fetch():
        r = SESSION.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
                        params={"stationTriplets": triplet, "elements": element, "duration": "DAILY",
                                "beginDate": "1980-10-01", "endDate": date.today().isoformat()}, timeout=180)
        r.raise_for_status()
        vals = r.json()[0]["data"][0]["values"]
        return pd.Series({pd.Timestamp(v["date"]): v.get("value") for v in vals}, dtype=float)
    return _cached(f"snotel_{triplet.replace(':', '_')}_{element}", fetch)


def usbr_reservoir(site_id, datatype):
    """Reclamation HydroData daily reservoir series (e.g. 17 storage, 49 pool elevation, 25 evaporation, 43 release)."""
    def fetch():
        j = SESSION.get(f"https://www.usbr.gov/uc/water/hydrodata/reservoir_data/{site_id}/json/{datatype}.json", timeout=180).json()
        df = pd.DataFrame(j["data"], columns=["date", "v"])
        return pd.Series(pd.to_numeric(df.v, errors="coerce").values, index=pd.to_datetime(df.date))
    return _cached(f"usbr_{site_id}_{datatype}", fetch)


def climdiv(kind):
    """NOAA nClimDiv monthly values ('tmpcdv' temperature °F, 'pcpndv' precipitation in) for the divisions in CLIMDIV,
    from the newest file NOAA has posted. Returns (file name, long table with div, year, month, wy, v)."""
    base = "https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/"
    name = sorted(set(re.findall(rf"climdiv-{kind}-v[0-9.]+-\d{{8}}", SESSION.get(base, timeout=60).text)))[-1]
    path = CACHE / name
    if not path.exists():
        path.write_bytes(SESSION.get(base + name, timeout=300).content)
    rows = []
    for line in path.read_text().splitlines():
        if line[:4] in CLIMDIV:
            vals = [float(line[10 + 7 * i: 17 + 7 * i]) for i in range(12)]
            rows += [(CLIMDIV[line[:4]], int(line[6:10]), m + 1, v) for m, v in enumerate(vals)]
    df = pd.DataFrame(rows, columns=["div", "year", "month", "v"])
    df.loc[df.v < -9, "v"] = np.nan  # -99.90 / -9.99 mark months NOAA hasn't posted yet
    df["wy"] = df.year + (df.month >= 10)
    return name, df


def season_mean(df, months, divs):
    """Mean over the given months and divisions, by water year; blank unless every month is reported."""
    x = df[df.month.isin(months) & df["div"].isin(divs)]
    per_div = x.groupby(["wy", "div"]).v.agg(lambda v: v.mean() if v.notna().all() and len(v) == len(months) else np.nan).unstack()
    return per_div.mean(axis=1, skipna=False)


def season_total(df, months, divs):
    x = df[df.month.isin(months) & df["div"].isin(divs)]
    per_div = x.groupby(["wy", "div"]).v.agg(lambda v: v.sum() if v.notna().all() and len(v) == len(months) else np.nan).unstack()
    return per_div.mean(axis=1, skipna=False)
