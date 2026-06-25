"""
=============================================================================
LIVE WEATHER DATA MODULE  –  live_weather.py
=============================================================================
Fetches real-time meteorological data from the Open-Meteo API
(https://open-meteo.com/) for use in the drone route simulation.

Open-Meteo is:
  • Free for non-commercial use – no API key required
  • Provides hourly wind speed at multiple pressure levels / altitudes
  • Returns JSON over plain HTTPS

Fallback strategy (3-tier):
  1. Live API call  →  parse and return real data
  2. Any network / HTTP / JSON error  →  log warning, return deterministic
     synthetic data shaped like the real response
  3. Unexpected exception  →  log, return fallback

All public functions return the *same* data structure regardless of which
tier fires, so the caller never needs to handle None or exceptions.

Public API
----------
fetch_wind_profile(lat, lon, n_points, route_length_km)
    → dict with keys:
        'source'        : str   – 'live' | 'fallback'
        'location'      : str   – human-readable label
        'fetch_time_s'  : float – seconds taken for the request (0.0 if fallback)
        'distances_km'  : np.ndarray (n_points,)  – route x-positions
        'wind_speeds'   : np.ndarray (n_points,)  – wind speed in m/s
        'wind_gusts'    : np.ndarray (n_points,)  – gust speed in m/s (may equal wind_speeds)
        'temperature_c' : float  – ambient temperature at route start (°C)
        'pressure_hpa'  : float  – surface pressure at route start (hPa)
        'humidity_pct'  : float  – relative humidity (%)
        'air_density'   : float  – derived air density (kg/m³)
        'description'   : str   – human-readable data summary
=============================================================================
"""

import time
import math
import warnings
import numpy as np

# ── optional imports (all in stdlib or already-installed) ────────────────────
try:
    import urllib.request
    import urllib.error
    import json as _json
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False

# ── constants ────────────────────────────────────────────────────────────────
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT  = 10          # seconds
R_DRY_AIR        = 287.058     # J/(kg·K)
CELSIUS_TO_K     = 273.15


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _air_density(temp_c: float, pressure_hpa: float, humidity_pct: float) -> float:
    """
    Derive air density (kg/m³) from temperature, pressure, and relative humidity
    using the ideal gas law with a vapour-pressure correction.

    Parameters
    ----------
    temp_c       : temperature in °C
    pressure_hpa : pressure in hPa (= mbar)
    humidity_pct : relative humidity in %  (0-100)

    Returns
    -------
    float : air density in kg/m³
    """
    T  = temp_c + CELSIUS_TO_K          # Kelvin
    P  = pressure_hpa * 100.0           # Pa
    # Saturation vapour pressure via Buck equation (hPa)
    e_s = 6.1121 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))
    e   = (humidity_pct / 100.0) * e_s * 100.0  # Pa
    # Density of moist air
    rho = (P - 0.378 * e) / (R_DRY_AIR * T)
    return max(rho, 0.8)   # physical lower bound


def _build_fallback(lat: float, lon: float,
                    n_points: int, route_length_km: float,
                    reason: str) -> dict:
    """
    Return a deterministic synthetic dataset shaped identically to the live
    response.  Uses a sine-based wind profile seeded by the route coordinates
    so different locations produce different (but reproducible) curves.
    """
    warnings.warn(
        f"[live_weather] Using fallback data. Reason: {reason}",
        RuntimeWarning, stacklevel=3
    )
    rng_seed = int(abs(lat * 100) + abs(lon * 100)) % (2**31)
    rng = np.random.default_rng(rng_seed)

    distances = np.linspace(0.0, route_length_km, n_points)
    # Representative midlatitude baseline: 6-10 m/s with spatial variation
    base_wind  = 6.0 + 4.0 * abs(math.sin(math.radians(lat)))
    wind_speeds = (
        base_wind
        + 2.0 * np.sin(distances * math.pi / route_length_km)
        + rng.normal(0.0, 0.4, n_points)
    ).clip(0.5, 25.0)
    wind_gusts  = (wind_speeds * 1.3 + rng.uniform(0, 1, n_points)).clip(0.5, 35.0)

    temp_c       = 20.0 - 0.006 * 200.0    # ISA lapse: 200 m asl assumed
    pressure_hpa = 1013.25 * ((1 - 0.0000225577 * 200) ** 5.25588)
    humidity_pct = 55.0
    rho          = _air_density(temp_c, pressure_hpa, humidity_pct)

    return {
        'source'        : 'fallback',
        'location'      : f"({lat:.4f}°, {lon:.4f}°)  [offline fallback]",
        'fetch_time_s'  : 0.0,
        'distances_km'  : distances,
        'wind_speeds'   : wind_speeds,
        'wind_gusts'    : wind_gusts,
        'temperature_c' : temp_c,
        'pressure_hpa'  : pressure_hpa,
        'humidity_pct'  : humidity_pct,
        'air_density'   : rho,
        'description'   : (
            f"FALLBACK – {reason}\n"
            f"  Synthetic wind: mean={wind_speeds.mean():.2f} m/s, "
            f"max={wind_speeds.max():.2f} m/s\n"
            f"  Air density: {rho:.4f} kg/m³"
        ),
    }


def _distribute_wind_along_route(hourly_wind: list, hourly_gusts: list,
                                  n_points: int, route_length_km: float) -> tuple:
    """
    Convert hourly wind forecast (typically 24 values) into a spatial wind
    profile along the route.

    Strategy:
      • Take the next 12 forecast hours as representative of the flight window.
      • Treat each hour as covering (route_length_km / 12) km of the route.
      • Interpolate to n_points with a cubic spline.
      • Add a small high-frequency perturbation (±0.3 m/s) to simulate
        spatial turbulence, seeded deterministically from the first value.
    """
    from scipy.interpolate import CubicSpline

    segment = min(12, len(hourly_wind))
    w  = np.array(hourly_wind[:segment],  dtype=float)
    g  = np.array(hourly_gusts[:segment], dtype=float)

    knot_x = np.linspace(0.0, route_length_km, segment)
    fine_x = np.linspace(0.0, route_length_km, n_points)

    cs_w = CubicSpline(knot_x, w, bc_type='not-a-knot')
    cs_g = CubicSpline(knot_x, g, bc_type='not-a-knot')

    # Spatial perturbation — deterministic but route-specific
    rng = np.random.default_rng(int(w[0] * 1000) % (2**31))
    noise_w = rng.normal(0.0, 0.3, n_points)
    noise_g = rng.normal(0.0, 0.2, n_points)

    wind_profile = np.clip(cs_w(fine_x) + noise_w, 0.5, 30.0)
    gust_profile = np.clip(cs_g(fine_x) + noise_g, wind_profile, 40.0)

    return wind_profile, gust_profile


# =============================================================================
# PUBLIC API
# =============================================================================

def fetch_wind_profile(
    lat: float           = 36.8969,   # Mersin, Turkey default
    lon: float           = 34.7313,
    n_points: int        = 12,
    route_length_km: float = 20.0,
    verbose: bool        = True,
) -> dict:
    """
    Fetch a real-time wind profile along a drone delivery route from
    the Open-Meteo free weather API.

    Parameters
    ----------
    lat, lon         : route start coordinates (decimal degrees)
    n_points         : number of spatial sample points along the route
    route_length_km  : total route length (km)
    verbose          : if True, print status messages

    Returns
    -------
    dict – see module docstring for full key list.
           'source' is 'live' on success, 'fallback' on any error.
    """
    if not _HTTP_AVAILABLE:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               "urllib not available in this environment")

    # ── Build API URL ─────────────────────────────────────────────────────────
    params = (
        f"latitude={lat:.6f}"
        f"&longitude={lon:.6f}"
        f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m,"
        f"surface_pressure,relative_humidity_2m"
        f"&wind_speed_unit=ms"           # metres per second
        f"&forecast_days=1"
        f"&timezone=auto"
    )
    url = f"{OPEN_METEO_BASE}?{params}"

    if verbose:
        print(f"  [live_weather] Querying Open-Meteo API ...")
        print(f"  [live_weather] URL: {url[:90]}...")

    # ── HTTP request with full error isolation ────────────────────────────────
    t_start = time.perf_counter()
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "DroneProject/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw  = resp.read().decode("utf-8")
        elapsed = time.perf_counter() - t_start

    except urllib.error.URLError as exc:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               f"Network error: {exc.reason}")
    except OSError as exc:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               f"OS/socket error: {exc}")
    except Exception as exc:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               f"Unexpected HTTP error: {exc}")

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        data = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               f"JSON parse error: {exc}")

    # ── Validate response structure ───────────────────────────────────────────
    try:
        hourly  = data["hourly"]
        # Mandatory fields
        wind_h  = hourly["wind_speed_10m"]          # list[float]
        gust_h  = hourly["wind_gusts_10m"]
        temp_h  = hourly["temperature_2m"]
        pres_h  = hourly["surface_pressure"]
        hum_h   = hourly["relative_humidity_2m"]

        if not wind_h:
            raise ValueError("Empty wind_speed_10m array")

    except (KeyError, ValueError) as exc:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               f"Unexpected API response structure: {exc}")

    # ── Derive spatial wind profile ───────────────────────────────────────────
    try:
        wind_profile, gust_profile = _distribute_wind_along_route(
            wind_h, gust_h, n_points, route_length_km
        )
    except Exception as exc:
        return _build_fallback(lat, lon, n_points, route_length_km,
                               f"Wind distribution error: {exc}")

    # ── Current-hour scalars (first non-None value) ──────────────────────────
    def first_valid(lst, default):
        for v in lst:
            if v is not None:
                return float(v)
        return float(default)

    temp_c       = first_valid(temp_h,  20.0)
    pressure_hpa = first_valid(pres_h,  1013.25)
    humidity_pct = first_valid(hum_h,   55.0)
    rho          = _air_density(temp_c, pressure_hpa, humidity_pct)

    location_label = data.get("timezone", f"({lat:.4f}°N, {lon:.4f}°E)")

    if verbose:
        print(f"  [live_weather] ✓ Live data received in {elapsed:.2f}s")
        print(f"  [live_weather]   Location   : {location_label}")
        print(f"  [live_weather]   Temperature: {temp_c:.1f} °C")
        print(f"  [live_weather]   Pressure   : {pressure_hpa:.1f} hPa")
        print(f"  [live_weather]   Humidity   : {humidity_pct:.0f} %")
        print(f"  [live_weather]   Air density: {rho:.4f} kg/m³")
        print(f"  [live_weather]   Wind range : "
              f"{wind_profile.min():.2f} – {wind_profile.max():.2f} m/s "
              f"(mean {wind_profile.mean():.2f} m/s)")

    return {
        'source'        : 'live',
        'location'      : location_label,
        'fetch_time_s'  : elapsed,
        'distances_km'  : np.linspace(0.0, route_length_km, n_points),
        'wind_speeds'   : wind_profile,
        'wind_gusts'    : gust_profile,
        'temperature_c' : temp_c,
        'pressure_hpa'  : pressure_hpa,
        'humidity_pct'  : humidity_pct,
        'air_density'   : rho,
        'description'   : (
            f"LIVE  – Open-Meteo ({elapsed:.2f}s)\n"
            f"  Location   : {location_label}\n"
            f"  Temperature: {temp_c:.1f} °C  |  Pressure: {pressure_hpa:.1f} hPa  "
            f"|  Humidity: {humidity_pct:.0f}%\n"
            f"  Wind: mean={wind_profile.mean():.2f} m/s, "
            f"max={wind_profile.max():.2f} m/s\n"
            f"  Air density: {rho:.4f} kg/m³"
        ),
    }


# =============================================================================
# SELF-TEST  (run this file directly to verify)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  live_weather.py  –  self-test")
    print("=" * 60)

    # Mersin, Turkey (close to the university)
    result = fetch_wind_profile(lat=36.8969, lon=34.7313, n_points=12,
                                route_length_km=20.0, verbose=True)

    print(f"\n  Source       : {result['source']}")
    print(f"  Description  :\n{result['description']}")
    print(f"\n  distances_km : {result['distances_km']}")
    print(f"  wind_speeds  : {np.round(result['wind_speeds'], 2)}")
    print(f"  wind_gusts   : {np.round(result['wind_gusts'], 2)}")
    print("\n[PASS]  Module loaded and executed successfully.")