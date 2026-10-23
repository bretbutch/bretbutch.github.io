#!/usr/bin/env python3
"""Pull current Rio Grande gage/reservoir readings and write data/river-data.json.

Sources (all public, no auth required):
  - MRGCD OneRain telemetry (mrgcd.onerain.com) — 5 diversion-division gages.
    Requires an anonymous session cookie, obtained by GETing the site root
    and following its redirect chain before calling the JSON export endpoint.
  - USGS Water Data API (latest-continuous) — mainstem channel gages, the Rio
    Chama below Abiquiu Dam and near its confluence with the Rio Grande,
    plus the three gages (channel + two canal intakes) that together
    account for all outflow from Cochiti Dam.
  - Colorado Division of Water Resources telemetry API — the Del Norte,
    Mogote, and Lobatos gages, plus every telemetered San Luis Valley
    diversion between the index gages and the state line.
  - USBR HydroData reservoir_data API — Elephant Butte storage + release,
    Cochiti storage, Abiquiu storage, plus Elephant Butte's Delta Storage
    and Evaporation series, which are upserted by date into a separate
    rolling-window file (data/elephant_butte_series.json) rather than
    written as a single latest-reading snapshot — see
    fetch_elephant_butte_series for why.

Each source is fetched independently; a failure on one does not prevent the
others from being written. Run directly (`python scripts/fetch_data.py`) or
via the twice-daily GitHub Actions workflow.
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

USGS_LATEST_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/latest-continuous/items"
USGS_SITES = {
    "embudo": "08279500",
    "abiquiu_channel": "08287000",
    "chama_chamita": "08290000",
    "otowi": "08313000",
    "albuquerque": "08330000",
    "isleta_lakes": "08330875",
    "san_acacia_channel": "08354900",
    "san_marcial_floodway": "08358400",
    "san_marcial_lfcc": "08358300",
    "cochiti_channel": "08317400",
    "cochiti_east_side_canal": "08313500",
    "cochiti_sile_canal": "08314000",
    # Tributaries, each the active real-time gage nearest its confluence.
    "galisteo": "08317950",    # enters between Cochiti and Angostura
    "jemez": "08328950",       # below Jemez Canyon Dam; enters below Angostura
    "rio_puerco": "08353000",  # near Bernardo; enters between Isleta and San Acacia
}

# Colorado Division of Water Resources telemetry (15-minute readings). The
# three Compact index/delivery gages used to come from USBR's gage_data
# daily series, which runs a day behind — during the Oct 2026 Del Norte
# flood it showed 1,250 cfs while CDWR's own gage read 10,200 and rising.
# USGS also lists these stations (08220000, 08246500, 08251500) but stopped
# real-time reporting in 2003; CDWR operates them now.
CDWR_BASE = "https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations"
CDWR_INDEX_GAGES = {
    "del_norte": "RIODELCO",
    "mogote": "CONMOGCO",
    "lobatos": "RIOLOBCO",
}
CDWR_DIVISION = 3  # Rio Grande basin in Colorado

# San Luis Valley diversions counted on the page: every telemetered
# diversion structure on the Rio Grande below the Del Norte gage, or on the
# Conejos below the Mogote gage, and above Lobatos. Chosen by rule rather
# than a hardcoded list so headgates CDWR adds or brings back online are
# picked up automatically. Tributaries (Alamosa, La Jara, Culebra, etc.)
# are excluded because Del Norte/Mogote never measure that water; so are
# transbasin imports, the Closed Basin Project (groundwater pumped *into*
# the river), and drains — none of those are diversions off the mainstem.
# "Below a gage" = east of it (larger longitude); the valley drains east.
SLV_DIVERSION_REACHES = {
    20: ("RIO GRANDE", "RIODELCO"),
    22: ("CONEJOS RIVER", "CONMOGCO"),
}
SLV_DIVERSION_DOWNSTREAM_LIMIT = "RIOLOBCO"

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

# MRGCD and CDWR telemetry is normally only minutes old. If a reading is
# older than this at fetch time (sensor/telemetry outage), don't publish a
# frozen number that looks live but isn't — the JSON carries the actual
# value and age for anyone inspecting it directly, but river-schematic.html
# is expected to show "—" / leave it out of totals instead.
STALENESS_THRESHOLD_HOURS = 24

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
            reading_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - reading_time).total_seconds() / 3600
            stale = age_hours > STALENESS_THRESHOLD_HOURS
            print(f"MRGCD {key} ({site_id}/{device_id}): {age_hours:.2f}h old{' — STALE' if stale else ''}", file=sys.stderr)
            out[key] = {
                "value": value,
                "units": series.get("units", "cfs"),
                "timestamp": reading_time.isoformat(),
                "age_hours": round(age_hours, 2),
                "stale": stale,
                "site_id": site_id,
                "device_id": device_id,
            }
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"MRGCD {key} ({site_id}/{device_id}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "site_id": site_id, "device_id": device_id}
    return out


def fetch_usgs(session: requests.Session) -> dict:
    """Latest instantaneous discharge for every USGS_SITES gage, in one request.

    Uses USGS's Water Data OGC API; the legacy waterservices.usgs.gov NWIS
    service this used to call is permanently shut down 2027-02-22. Like the
    legacy service, "latest" here means the newest reading on record however
    old — so a reading older than STALENESS_THRESHOLD_HOURS is reported as an
    error (shown as "no reading") rather than as a current value.
    """
    loc_ids = {f"USGS-{site}": key for key, site in USGS_SITES.items()}
    try:
        r = session.get(
            USGS_LATEST_URL,
            params={
                "f": "json",
                "monitoring_location_id": ",".join(loc_ids),
                "parameter_code": "00060",
                "statistic_id": "00011",  # instantaneous
                "limit": len(loc_ids) * 2,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        latest = {f["properties"]["monitoring_location_id"]: f["properties"] for f in r.json()["features"]}
    except (requests.RequestException, KeyError, ValueError) as exc:
        print(f"USGS request failed: {exc}", file=sys.stderr)
        return {key: {"error": str(exc), "site": site} for key, site in USGS_SITES.items()}

    out = {}
    for loc_id, key in loc_ids.items():
        site = loc_id.removeprefix("USGS-")
        props = latest.get(loc_id)
        try:
            if props is None or props.get("value") is None:
                raise ValueError("no latest reading returned (site may be offline)")
            reading_time = datetime.fromisoformat(props["time"])
            age_hours = _age_hours(reading_time)
            if age_hours > STALENESS_THRESHOLD_HOURS:
                raise ValueError(f"stale reading ({age_hours:.0f}h old, {props['time']})")
            out[key] = {
                "value": float(props["value"]),
                "units": "cfs",
                "timestamp": reading_time.isoformat(),
                "age_hours": round(age_hours, 2),
                "approval_status": props.get("approval_status"),
                "site": site,
            }
        except (KeyError, ValueError) as exc:
            print(f"USGS {key} ({site}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "site": site}
    return out


def _age_hours(reading_time: datetime) -> float:
    return (datetime.now(timezone.utc) - reading_time).total_seconds() / 3600


def fetch_cdwr_index_gages(session: requests.Session) -> dict:
    """Latest non-blank discharge for the Colorado index/delivery gages.

    CDWR leaves discharge blank when it can't compute flow from stage —
    e.g. Del Norte on 2026-10-23, when stage kept rising past the point
    discharge stopped reporting, most likely above the top of the rating
    table. In that case the last real value is reported along with
    `later_readings_blank` and the time of the newest (blank) reading, so
    the page can say so instead of silently showing an older number.
    """
    mountain = ZoneInfo("America/Denver")
    start = (datetime.now(mountain) - timedelta(days=2)).strftime("%m/%d/%Y")
    out = {}
    for key, abbrev in CDWR_INDEX_GAGES.items():
        try:
            r = session.get(
                f"{CDWR_BASE}/telemetrytimeseriesraw/",
                params={"format": "json", "abbrev": abbrev, "parameter": "DISCHRG", "startDate": start},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            rows = r.json()["ResultList"]
            valid = [row for row in rows if row.get("measValue") is not None]
            if not valid:
                raise ValueError("no non-blank discharge readings in the last 2 days")

            # This endpoint's timestamps are naive Denver-local time.
            def local(ts: str) -> datetime:
                dt = datetime.fromisoformat(ts)
                return dt if dt.tzinfo else dt.replace(tzinfo=mountain)

            last = valid[-1]
            reading_time = local(last["measDateTime"])
            age_hours = _age_hours(reading_time)
            stale = age_hours > STALENESS_THRESHOLD_HOURS
            blank_after = rows[-1].get("measValue") is None
            print(
                f"CDWR {key} ({abbrev}): {last['measValue']} cfs, {age_hours:.2f}h old"
                f"{' — later readings blank' if blank_after else ''}{' — STALE' if stale else ''}",
                file=sys.stderr,
            )
            out[key] = {
                "value": last["measValue"],
                "units": "cfs",
                "timestamp": reading_time.isoformat(),
                "age_hours": round(age_hours, 2),
                "stale": stale,
                "later_readings_blank": blank_after,
                "latest_reading_timestamp": local(rows[-1]["measDateTime"]).isoformat(),
                "latest_reading_flag": rows[-1].get("flagB"),
                "abbrev": abbrev,
            }
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"CDWR {key} ({abbrev}) failed: {exc}", file=sys.stderr)
            out[key] = {"error": str(exc), "abbrev": abbrev}
    return out


def fetch_slv_diversions(session: requests.Session) -> dict:
    """Latest reading for every telemetered San Luis Valley diversion that
    takes water between the Del Norte/Mogote index gages and Lobatos — see
    SLV_DIVERSION_REACHES for the selection rule."""
    try:
        r = session.get(
            f"{CDWR_BASE}/telemetrystation/",
            params={"format": "json", "division": CDWR_DIVISION, "pageSize": 5000},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()["ResultList"]

        ref_abbrevs = {abbrev for _, abbrev in SLV_DIVERSION_REACHES.values()} | {SLV_DIVERSION_DOWNSTREAM_LIMIT}
        lon = {row["abbrev"]: row["longitude"] for row in rows if row.get("abbrev") in ref_abbrevs}
        missing = ref_abbrevs - lon.keys()
        if missing:
            raise ValueError(f"reference gage(s) missing from station list: {sorted(missing)}")
        limit_lon = lon[SLV_DIVERSION_DOWNSTREAM_LIMIT]

        picked: dict = {}
        for row in rows:
            param = str(row.get("parameter") or "")
            reach = SLV_DIVERSION_REACHES.get(row.get("waterDistrict"))
            if row.get("stationType") != "Diversion Structure" or not param.startswith("DISCHRG") or not reach:
                continue
            source, ref_abbrev = reach
            if row.get("waterSource") != source or row.get("longitude") is None:
                continue
            if not (lon[ref_abbrev] < row["longitude"] < limit_lon):
                continue
            # Some stations report under DISCHRG2 instead of DISCHRG; prefer DISCHRG if both exist.
            if row["abbrev"] in picked and picked[row["abbrev"]]["parameter"] == "DISCHRG":
                continue
            picked[row["abbrev"]] = row

        stations = []
        for row in picked.values():
            ts = row.get("measDateTime")
            reading_time = datetime.fromisoformat(ts) if ts else None  # this endpoint includes the UTC offset
            age_hours = _age_hours(reading_time) if reading_time else None
            stations.append({
                "abbrev": row["abbrev"],
                "name": row["stationName"],
                "water_district": row["waterDistrict"],
                "value": row.get("measValue"),
                "units": "cfs",
                "parameter": row["parameter"],
                "timestamp": reading_time.isoformat() if reading_time else None,
                "age_hours": round(age_hours, 2) if age_hours is not None else None,
                "stale": age_hours is None or age_hours > STALENESS_THRESHOLD_HOURS,
            })
        stations.sort(key=lambda s: -(s["value"] or 0))
        reporting = [s for s in stations if s["value"] is not None and not s["stale"]]
        print(
            f"CDWR SLV diversions: {len(stations)} stations matched, {len(reporting)} reporting, "
            f"{sum(s['value'] for s in reporting):.1f} cfs",
            file=sys.stderr,
        )
        return {"stations": stations}
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"CDWR SLV diversions failed: {exc}", file=sys.stderr)
        return {"error": str(exc)}


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
        "cdwr": fetch_cdwr_index_gages(session),
        "slv_diversions": fetch_slv_diversions(session),
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
