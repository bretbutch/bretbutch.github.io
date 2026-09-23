#!/usr/bin/env python3
"""Pull current Rio Grande gage/reservoir readings and write data/river-data.json.

Sources (all public, no auth required):
  - MRGCD OneRain telemetry (mrgcd.onerain.com) — 5 diversion-division gages.
    Requires an anonymous session cookie, obtained by GETing the site root
    and following its redirect chain before calling the JSON export endpoint.
  - USGS NWIS instantaneous values API — mainstem channel gages, the Rio
    Chama below Abiquiu Dam and near its confluence with the Rio Grande,
    plus the three gages (channel + two canal intakes) that together
    account for all outflow from Cochiti Dam.
  - USBR HydroData gage_data API — 3 Colorado/state-line flow gages.
  - USBR HydroData reservoir_data API — Elephant Butte storage + release,
    Cochiti storage, Abiquiu storage, plus Elephant Butte's Delta Storage
    and Evaporation series, which are upserted by date into a separate
    rolling-window file (data/elephant_butte_series.json) rather than
    written as a single latest-reading snapshot — see
    fetch_elephant_butte_series for why.

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
    "embudo": "08279500",
    "abiquiu_channel": "08287000",
    "chama_chamita": "08290000",
    "otowi": "08313000",
    "albuquerque": "08330000",
    "isleta_lakes": "08330875",
    "san_marcial_floodway": "08358400",
    "san_marcial_lfcc": "08358300",
    "cochiti_channel": "08317400",
    "cochiti_east_side_canal": "08313500",
    "cochiti_sile_canal": "08314000",
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
    "abiquiu_storage": (2730, 17),
}

# Elephant Butte storage-change series: fetched and upserted separately from
# USBR_RESERVOIR_PARAMS above because it's an accumulating time series (we
# keep a rolling window so revisions to recent days are visible over time),
# not a single latest-reading snapshot like everything else in this file.
#
# Param IDs confirmed 2026-09-22 by probing reservoir_data/1119/json/{1..60}
# and reading each response's `columns` field:
#   17 = storage, 25 = evaporation, 29 = inflow, 30 = inflow volume,
#   42 = total release, 43 = release volume, 47 = delta storage, 49 = pool elevation.
# Deliberately NOT using 29/30 (inflow, inflow volume) — at this site those
# are themselves derived from the delta-storage/release accounting, not an
# independent gaged measurement, so using them here would be circular.
#
# Verified "delta storage" (param 47) against the storage series (param 17)
# itself: for every recent day, delta_storage[date] == storage[date] -
# storage[date - 1 day] exactly (e.g. 2026-09-17: 28961 - 28011 = 950,
# matching the API's own 950.0). Confirms units are acre-feet (same units
# as storage) and that a rising pool is a positive value.
EB_SITE_ID = 1119
EB_DELTA_STORAGE_PARAM = 47
EB_EVAPORATION_PARAM = 25
EB_SERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "elephant_butte_series.json"
EB_SERIES_WINDOW_DAYS = 45  # only keep a recent rolling window; USBR's own archive holds full history back to 1915
EB_REVISION_LOG_THRESHOLD_AF = 50

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


def fetch_elephant_butte_series(session: requests.Session) -> dict:
    """Fetch USBR's Delta Storage and Evaporation series for Elephant Butte,
    upsert by date into EB_SERIES_PATH, and trim to a rolling window.

    Unlike fetch_usbr_reservoir, this reads and rewrites its own on-disk
    state so that a later USBR revision to an already-stored date is
    detected (and logged) instead of silently overwritten with no record —
    the whole point being to observe how much USBR revises after the fact.
    """
    existing: dict = {}
    if EB_SERIES_PATH.exists():
        try:
            existing = json.loads(EB_SERIES_PATH.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Elephant Butte series: couldn't read existing {EB_SERIES_PATH}, starting fresh: {exc}", file=sys.stderr)
            existing = {}

    def fetch_param_tail(param_id: int) -> dict:
        r = session.get(
            f"https://www.usbr.gov/uc/water/hydrodata/reservoir_data/{EB_SITE_ID}/json/{param_id}.json",
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()["data"]
        # Only the tail is relevant to our rolling window; no need to hold
        # the full period-of-record response (40,000+ rows back to 1915).
        return {date: value for date, value in rows[-(EB_SERIES_WINDOW_DAYS * 2):]}

    try:
        delta_by_date = fetch_param_tail(EB_DELTA_STORAGE_PARAM)
        evap_by_date = fetch_param_tail(EB_EVAPORATION_PARAM)
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"Elephant Butte series fetch failed: {exc}", file=sys.stderr)
        return existing

    for date in set(delta_by_date) | set(evap_by_date):
        new_delta = delta_by_date.get(date)
        new_evap = evap_by_date.get(date)
        prior = existing.get(date, {})

        for label, new_val, key in (
            ("delta storage", new_delta, "delta_storage_af"),
            ("evaporation", new_evap, "evaporation_af"),
        ):
            prior_val = prior.get(key)
            if prior_val is not None and new_val is not None and abs(new_val - prior_val) > EB_REVISION_LOG_THRESHOLD_AF:
                print(
                    f"USBR revision: Elephant Butte {label} for {date} changed "
                    f"from {prior_val} to {new_val} AF ({new_val - prior_val:+.1f})",
                    file=sys.stderr,
                )

        existing[date] = {
            "delta_storage_af": new_delta if new_delta is not None else prior.get("delta_storage_af"),
            "evaporation_af": new_evap if new_evap is not None else prior.get("evaporation_af"),
        }

    # Trim to the rolling window so this file doesn't grow without bound.
    cutoff_dates = sorted(existing.keys())[-EB_SERIES_WINDOW_DAYS:]
    existing = {date: existing[date] for date in cutoff_dates}

    EB_SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EB_SERIES_PATH.open("w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
        f.write("\n")

    return existing


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

    fetch_elephant_butte_series(session)
    print(f"Wrote {EB_SERIES_PATH}")


if __name__ == "__main__":
    main()
