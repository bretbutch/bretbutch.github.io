#!/usr/bin/env python3
"""Pull current Rio Grande gage/reservoir readings and write data/river-data.json.

Sources (all public, no auth required):
  - MRGCD OneRain telemetry (mrgcd.onerain.com) — 5 diversion-division gages.
    Requires an anonymous session cookie, obtained by GETing the site root
    and following its redirect chain before calling the JSON export endpoint.
  - USGS NWIS instantaneous values API — 2 mainstem channel gages.
  - USBR HydroData gage_data API — 3 Colorado/state-line flow gages.
  - USBR HydroData reservoir_data API — Elephant Butte storage + release.

Each source is fetched independently; a failure on one does not prevent the
others from being written. Run directly (`python scripts/fetch_data.py`) or
via the daily GitHub Actions workflow.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "river-data.json"

MRGCD_BASE = "https://mrgcd.onerain.com"
MRGCD_SITES = {
    "mrgcd_total": (195, 2),
    "cochiti": (198, 2),
    "angostura": (77, 3),
    "isleta": (197, 2),
    "san_acacia": (131, 3),
}

USGS_SITES = {
    "otowi": "08313000",
    "albuquerque": "08330000",
    "isleta_lakes": "08330875",
    "san_marcial_floodway": "08358400",
    "san_marcial_lfcc": "08358300",
}

USBR_GAGE_SITES = {
    "del_norte": (2722, 19),
    "mogote": (2741, 19),
    "lobatos": (2723, 19),
}

USBR_RESERVOIR_PARAMS = {
    "elephant_butte_storage": (1119, 17),
    "elephant_butte_release": (1119, 43),
    "cochiti_storage": (2696, 17),
    "cochiti_release": (2696, 43),
}

# USBR reports reservoir "release volume" as acre-feet for the day, not an
# instantaneous cfs reading. Convert to an average cfs for the day so it's
# comparable to every other flow figure in this dataset. Confirmed against
# USBR's own live Cochiti Lake dashboard (usbr.gov/uc/water/hydrodata):
# 172.36 AF/day from this API matched its "currently: 87 cfs" readout
# (172.36 * 43560 / 86400 = 86.9 cfs).
AF_PER_DAY_TO_CFS = 43560.0 / 86400.0

USER_AGENT = "paperwater-river-pipeline/1.0 (+https://paperwater.net; contact butch@paperwater.net)"
TIMEOUT = 20


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def fetch_mrgcd(session: requests.Session) -> dict:
    out = {}
    try:
        session.get(MRGCD_BASE + "/", timeout=TIMEOUT)  # bootstrap guest session cookie
    except requests.RequestException as exc:
        print(f"MRGCD session bootstrap failed: {exc}", file=sys.stderr)
        return {key: {"error": str(exc)} for key in MRGCD_SITES}

    mountain = ZoneInfo("America/Denver")
    end = datetime.now(mountain)
    start = end - timedelta(days=2)

    for key, (site_id, device_id) in MRGCD_SITES.items():
        params = {
            "method": "sensorDetails",
            "site_id": site_id,
            "device_id": device_id,
            "site": site_id,
            "device": device_id,
            "data_start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "data_end": end.strftime("%Y-%m-%d %H:%M:%S"),
            "range": 2,
            "time_zone": "US/Mountain",
        }
        try:
            r = session.get(MRGCD_BASE + "/export/flot/", params=params, timeout=TIMEOUT)
            r.raise_for_status()
            series = r.json()[0]
            ts_ms, value = series["data"][-1]
            out[key] = {
                "value": value,
                "units": series.get("units", "cfs"),
                "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                "site_id": site_id,
                "device_id": device_id,
            }
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"MRGCD {key} ({site_id}/{device_id}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "site_id": site_id, "device_id": device_id}
    return out


def fetch_usgs(session: requests.Session) -> dict:
    out = {}
    for key, site in USGS_SITES.items():
        try:
            r = session.get(
                "https://waterservices.usgs.gov/nwis/iv/",
                params={"sites": site, "parameterCd": "00060", "format": "json"},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            time_series = r.json()["value"]["timeSeries"]
            if not time_series:
                raise ValueError("no timeSeries returned (site may be offline)")
            series = time_series[0]
            latest = series["values"][0]["value"][-1]
            out[key] = {
                "value": float(latest["value"]),
                "units": "cfs",
                "timestamp": latest["dateTime"],
                "site": site,
                "name": series["sourceInfo"]["siteName"],
            }
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"USGS {key} ({site}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "site": site}
    return out


def fetch_usbr_gage(session: requests.Session) -> dict:
    out = {}
    for key, (site_id, param_id) in USBR_GAGE_SITES.items():
        try:
            r = session.get(
                f"https://www.usbr.gov/uc/water/hydrodata/gage_data/{site_id}/json/{param_id}.json",
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            date, value = r.json()["data"][-1]
            out[key] = {
                "value": value,
                "units": "cfs",
                "date": date,
                "site_id": site_id,
                "param_id": param_id,
            }
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"USBR gage {key} ({site_id}/{param_id}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "site_id": site_id, "param_id": param_id}
    return out


def fetch_usbr_reservoir(session: requests.Session) -> dict:
    out = {}
    for key, (site_id, param_id) in USBR_RESERVOIR_PARAMS.items():
        try:
            r = session.get(
                f"https://www.usbr.gov/uc/water/hydrodata/reservoir_data/{site_id}/json/{param_id}.json",
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            payload = r.json()
            date, value = payload["data"][-1]
            columns = payload.get("columns")
            column_label = columns[1] if columns and len(columns) > 1 else None

            if column_label == "release volume":
                out[key] = {
                    "value": round(value * AF_PER_DAY_TO_CFS, 1),
                    "units": "cfs",
                    "note": "average cfs for the day, computed from USBR's acre-feet/day release volume",
                    "raw_af_per_day": value,
                    "date": date,
                    "site_id": site_id,
                    "param_id": param_id,
                }
            else:
                out[key] = {
                    "value": value,
                    "units": "AF",
                    "date": date,
                    "site_id": site_id,
                    "param_id": param_id,
                    "columns": columns,
                }
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"USBR reservoir {key} ({site_id}/{param_id}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "site_id": site_id, "param_id": param_id}
    return out


def main() -> None:
    session = make_session()

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mrgcd": fetch_mrgcd(session),
        "usgs": fetch_usgs(session),
        "usbr_gage": fetch_usbr_gage(session),
        "usbr_reservoir": fetch_usbr_reservoir(session),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
