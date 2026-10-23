"""Pull daily SNOTEL (WTEQ, PREC) and daily streamflow for the three upper Rio Grande pairs."""
import json, time, urllib.error, urllib.request
import pandas as pd

STATIONS = {"Cumbres Trestle": "431:CO:SNTL", "Wolf Creek Summit": "874:CO:SNTL", "Beartown": "327:CO:SNTL",
            "Hopewell": "532:NM:SNTL"}
# Each basin's snow index is a daily weighted average of its SNOTEL sites (weights sum to 1).
# Rio Chama: Cumbres stands for the upper Chama and Hopewell (just outside the drainage, in the Tusas) for the
# Rio Brazos side; the two halves each supply roughly half of La Puente's summer flow (2015–2026 gages).
PAIRS = {
    "Rio Chama": {"snotel": {"Cumbres Trestle": 0.5, "Hopewell": 0.5}, "usgs": "08284100", "dwr": None, "gage": "Rio Chama nr La Puente"},
    "Conejos":   {"snotel": {"Cumbres Trestle": 0.5, "Wolf Creek Summit": 0.5}, "usgs": "08246500", "dwr": "CONMOGCO", "gage": "Conejos nr Mogote"},
    "Rio Grande":{"snotel": {"Wolf Creek Summit": 0.5, "Beartown": 0.5}, "usgs": "08220000", "dwr": "RIODELCO", "gage": "Rio Grande nr Del Norte"},
}
START, END = "1979-10-01", "2026-09-30"

def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code not in (403, 429)) or i == tries - 1:
                raise
            time.sleep(10 * (i + 1))

def snotel(triplet):
    url = ("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data?"
           f"stationTriplets={triplet}&elements=WTEQ,PREC&duration=DAILY&beginDate={START}&endDate={END}")
    out = {}
    for el in json.loads(get(url))[0]["data"]:
        code = el["stationElement"]["elementCode"]
        out[code] = pd.Series({v["date"]: v.get("value") for v in el["values"]}, dtype=float)
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df

def usgs(site):
    """USGS daily mean discharge from the Water Data API (the older waterservices endpoint is being retired)."""
    url = ("https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items?f=json&limit=10000&properties=time,value"
           f"&monitoring_location_id=USGS-{site}&parameter_code=00060&statistic_id=00003&time={START}/{END}")
    rows = []
    while url:
        j = json.loads(get(url))
        rows += [f["properties"] for f in j["features"]]
        url = next((l["href"] for l in j["links"] if l.get("rel") == "next"), None)
    df = pd.DataFrame(rows)
    return pd.Series(pd.to_numeric(df.value, errors="coerce").values, index=pd.to_datetime(df.time)).sort_index()

def dwr(abbrev):
    r = json.loads(get("https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewatertsday/"
                       f"?format=json&abbrev={abbrev}&min-measDate=01/01/2025&pageSize=50000"))["ResultList"]
    return pd.Series({pd.Timestamp(x["measDate"][:10]): x["value"] for x in r}, dtype=float)

sites = {name: snotel(trip) for name, trip in STATIONS.items()}
pd.concat(sites, names=["station", "date"]).reset_index().to_csv("snotel_sites.csv", index=False)

def blend(weights):
    # skipna=False: a day counts only when every contributing site reported
    return sum(sites[n] * w for n, w in weights.items()).round(2)

frames = []
for basin, p in PAIRS.items():
    sn = blend(p["snotel"])
    q = usgs(p["usgs"])
    if p["dwr"]:
        d = dwr(p["dwr"])
        q = pd.concat([q, d[d.index > q.index.max()]])
    df = sn.join(q.rename("Q"), how="outer")
    df["basin"] = basin
    frames.append(df)
all_ = pd.concat(frames).rename_axis("date").reset_index()
all_.to_csv("daily.csv", index=False)
print(all_.groupby("basin").agg(first=("date", "min"), last=("date", "max"), n_swe=("WTEQ", "count"), n_q=("Q", "count")))

# Full La Puente record (WY1956 on) for the long-record chart in site_charts.py. Still the older service (its
# qualifier column flags provisional days); if it's down, keep the copy already on disk.
try:
    txt = get(f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=08284100&parameterCd=00060&startDT=1955-10-01&endDT={END}")
    with open("lapuente_dv.rdb", "w") as f:
        f.write(txt)
except urllib.error.HTTPError as e:
    print(f"La Puente long record not refreshed ({e}); keeping lapuente_dv.rdb from the last run")
