"""
=============================================================================
DRONE DELIVERY ROUTE OPTIMIZATION USING NUMERICAL METHODS
=============================================================================
Course  : 155-4007 Numerical Methods in Engineering
Topic   : Drone Delivery Route Optimization Using Numerical Methods
Language: Python 3

Description:
    This project applies comprehensive numerical methods to the problem of
    optimizing drone delivery routes. It covers error analysis, root finding,
    interpolation, numerical differentiation, numerical integration, linear
    systems, LU decomposition, optimization, ODE solving, stability analysis,
    visualization, and comparative case studies — all within the context of
    real-world drone logistics.

Topics Covered:
    1.  Error Analysis and Floating-Point Precision
    2.  Solving Equations (Root Finding)
    3.  Interpolation Techniques
    4.  Numerical Differentiation
    5.  Numerical Integration
    6.  Solving Linear Systems
    7.  LU Decomposition for Efficient Calculations
    8.  Optimization Techniques
    9.  Ordinary Differential Equations (ODEs) Solving
    10. Performance Analysis, Numerical Stability, and Error Handling
    11. Visualization and Documentation
    12. Comparative Analysis and Case Study
=============================================================================
"""

import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import odeint, solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import time
import warnings
import os

from realtime_weather import fetch_wind_profile

warnings.filterwarnings('ignore')

# Output folder for plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SEPARATOR = "=" * 70


def section_header(number, title):
    print(f"\n{SEPARATOR}")
    print(f"  TOPIC {number}: {title}")
    print(SEPARATOR)


# =============================================================================
# TOPIC 1: ERROR ANALYSIS AND FLOATING-POINT PRECISION
# =============================================================================

def topic1_error_analysis():
    section_header(1, "ERROR ANALYSIS AND FLOATING-POINT PRECISION")

    print("\n[1.1] Machine Epsilon and Floating-Point Limits")
    machine_eps = np.finfo(float).eps
    print(f"  Machine epsilon (float64): {machine_eps:.6e}")
    print(f"  Max float64 value        : {np.finfo(float).max:.6e}")
    print(f"  Min positive float64     : {np.finfo(float).tiny:.6e}")

    # Demonstrate catastrophic cancellation
    print("\n[1.2] Catastrophic Cancellation Demonstration")
    a = 1.0
    b = 1e-15
    exact = b  # a + b - a = b
    computed = (a + b) - a
    rel_error = abs(computed - exact) / abs(exact) if exact != 0 else 0
    print(f"  True value  (b)          : {exact:.6e}")
    print(f"  Computed (a+b)-a         : {computed:.6e}")
    print(f"  Relative error           : {rel_error:.4f} ({rel_error*100:.2f}%)")

    # Iterative error accumulation in summing distances
    print("\n[1.3] Iterative Error Accumulation in Route Distance Summing")
    # Simulate 1000-step route; add small distances iteratively
    n_steps = 1000
    true_dist = 100.0  # km
    step_dist = true_dist / n_steps  # each segment
    errors = []
    cumulative = 0.0
    for i in range(n_steps):
        cumulative += step_dist
        errors.append(abs(cumulative - step_dist * (i + 1)))

    # Kahan compensated summation
    kahan_sum = 0.0
    compensation = 0.0
    for i in range(n_steps):
        y = step_dist - compensation
        t = kahan_sum + y
        compensation = (t - kahan_sum) - y
        kahan_sum = t

    naive_final_error = abs(cumulative - true_dist)
    kahan_final_error = abs(kahan_sum - true_dist)
    print(f"  Naive summation error    : {naive_final_error:.4e} km")
    print(f"  Kahan summation error    : {kahan_final_error:.4e} km")
    print(f"  Kahan improvement factor : {naive_final_error / (kahan_final_error + 1e-20):.1f}x")

    # Rounding error in GPS coordinates
    print("\n[1.4] GPS Coordinate Rounding Error Analysis")
    lat_true = 36.123456789  # degrees
    precisions = [1, 2, 3, 4, 5, 6]
    print(f"  {'Decimal Places':<20} {'Rounded Lat':<20} {'Error (m)':<15}")
    print(f"  {'-'*55}")
    for p in precisions:
        rounded = round(lat_true, p)
        # 1 degree lat ≈ 111,000 m
        error_m = abs(lat_true - rounded) * 111000
        print(f"  {p:<20} {rounded:<20.{p}f} {error_m:<15.4f}")

    # Plot error accumulation
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Topic 1: Floating-Point Error Analysis", fontsize=14, fontweight='bold')

    axes[0].plot(errors, color='crimson', linewidth=1)
    axes[0].set_title("Iterative Summation Errors (Naive)")
    axes[0].set_xlabel("Step Number")
    axes[0].set_ylabel("Cumulative Absolute Error (km)")
    axes[0].grid(True, alpha=0.4)

    prec_arr = np.array(precisions)
    errors_m = [abs(lat_true - round(lat_true, p)) * 111000 for p in precisions]
    axes[1].semilogy(prec_arr, errors_m, 'bo-', linewidth=2, markersize=8)
    axes[1].set_title("GPS Coordinate Rounding Error vs Decimal Precision")
    axes[1].set_xlabel("Decimal Places in Latitude")
    axes[1].set_ylabel("Position Error (meters, log scale)")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic1_error_analysis.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic1_error_analysis.png]")

    return {
        'machine_eps': machine_eps,
        'naive_error': naive_final_error,
        'kahan_error': kahan_final_error
    }


# =============================================================================
# TOPIC 2: SOLVING EQUATIONS (ROOT FINDING)
# =============================================================================

def topic2_root_finding():
    section_header(2, "SOLVING EQUATIONS (ROOT FINDING)")

    # Physical problem: Find the optimal hover altitude where drag force = thrust
    # Drag model: F_drag(h) = 0.5 * rho(h) * Cd * A * v^2 - F_thrust = 0
    # Air density: rho(h) = rho0 * exp(-h/H)  (barometric formula)

    rho0 = 1.225      # kg/m^3 sea-level air density
    H    = 8500.0     # scale height (m)
    Cd   = 0.3        # drag coefficient
    A    = 0.05       # drone cross-section (m^2)
    v    = 15.0       # drone cruise speed (m/s)
    F_thrust = 1.0    # Net aerodynamic force threshold (N)
    # At sea level: F_drag = 0.5*1.225*0.3*0.05*15^2 ≈ 2.07 N > 1.0 N → root exists

    def drag_equation(h):
        rho = rho0 * np.exp(-h / H)
        return 0.5 * rho * Cd * A * v**2 - F_thrust

    print("\n[2.1] Problem: Find altitude where aerodynamic drag = net thrust")
    print(f"  Drag threshold = {F_thrust} N, Speed = {v} m/s, Cd = {Cd}")
    print(f"  Sea-level drag ≈ {0.5*rho0*Cd*A*v**2:.4f} N  → root exists where drag falls to threshold")

    # --- Bisection Method ---
    def bisection(f, a, b, tol=1e-8, max_iter=100):
        if f(a) * f(b) > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        history = []
        for i in range(max_iter):
            c = (a + b) / 2.0
            fc = f(c)
            history.append((i + 1, c, abs(fc)))
            if abs(fc) < tol or (b - a) / 2.0 < tol:
                break
            if f(a) * fc < 0:
                b = c
            else:
                a = c
        return c, history

    # --- Newton-Raphson Method ---
    def newton_raphson(f, df, x0, tol=1e-8, max_iter=100):
        x = x0
        history = []
        for i in range(max_iter):
            fx = f(x)
            history.append((i + 1, x, abs(fx)))
            if abs(fx) < tol:
                break
            dfx = df(x)
            if abs(dfx) < 1e-15:
                raise ZeroDivisionError("Derivative too small")
            x = x - fx / dfx
        return x, history

    # --- Secant Method ---
    def secant(f, x0, x1, tol=1e-8, max_iter=100):
        history = []
        for i in range(max_iter):
            fx0, fx1 = f(x0), f(x1)
            history.append((i + 1, x1, abs(fx1)))
            if abs(fx1) < tol:
                break
            if abs(fx1 - fx0) < 1e-15:
                raise ZeroDivisionError("Denominator too small in secant")
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            x0, x1 = x1, x2
        return x1, history

    # Analytical derivative of drag equation
    def d_drag_equation(h):
        return -0.5 * rho0 * np.exp(-h / H) * Cd * A * v**2 / H

    print("\n[2.2] Applying Root-Finding Methods")
    results = {}

    try:
        root_bis, hist_bis = bisection(drag_equation, 0, 12000, tol=1e-8)
        results['bisection'] = {'root': root_bis, 'iters': len(hist_bis), 'history': hist_bis}
        print(f"  Bisection    : root = {root_bis:.4f} m, iterations = {len(hist_bis)}")
    except Exception as e:
        print(f"  Bisection error: {e}")

    try:
        root_nr, hist_nr = newton_raphson(drag_equation, d_drag_equation, x0=3000, tol=1e-8)
        results['newton'] = {'root': root_nr, 'iters': len(hist_nr), 'history': hist_nr}
        print(f"  Newton-Raphson: root = {root_nr:.4f} m, iterations = {len(hist_nr)}")
    except Exception as e:
        print(f"  Newton-Raphson error: {e}")

    try:
        root_sec, hist_sec = secant(drag_equation, 1000, 6000, tol=1e-8)
        results['secant'] = {'root': root_sec, 'iters': len(hist_sec), 'history': hist_sec}
        print(f"  Secant       : root = {root_sec:.4f} m, iterations = {len(hist_sec)}")
    except Exception as e:
        print(f"  Secant error: {e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Topic 2: Root-Finding for Drone Hover Altitude", fontsize=14, fontweight='bold')

    h_vals = np.linspace(0, 25000, 500)
    axes[0].plot(h_vals, drag_equation(h_vals), 'b-', linewidth=2, label='Drag - Thrust')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=1)
    if 'bisection' in results:
        axes[0].axvline(results['bisection']['root'], color='red', linestyle=':', label=f"Root ≈ {results['bisection']['root']:.0f} m")
    axes[0].set_xlabel("Altitude (m)")
    axes[0].set_ylabel("F_drag - F_thrust (N)")
    axes[0].set_title("Drag - Thrust vs Altitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    colors = {'bisection': 'crimson', 'newton': 'blue', 'secant': 'green'}
    for method, color in colors.items():
        if method in results:
            iters = [h[0] for h in results[method]['history']]
            errs  = [h[2] for h in results[method]['history']]
            axes[1].semilogy(iters, errs, color=color, marker='o', markersize=3,
                             linewidth=1.5, label=method.capitalize())
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("|f(x)| (log scale)")
    axes[1].set_title("Convergence Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic2_root_finding.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic2_root_finding.png]")

    return results


# =============================================================================
# TOPIC 3: INTERPOLATION TECHNIQUES
# =============================================================================

def topic3_interpolation(weather: dict):
    """
    Interpolation techniques applied to drone route data.

    Parameters
    ----------
    weather : dict returned by live_weather.fetch_wind_profile()
              Keys used: 'distances_km', 'wind_speeds', 'source', 'location'
    """
    section_header(3, "INTERPOLATION TECHNIQUES")

    # ── Terrain data (fixed waypoints from survey / map data) ─────────────
    waypoint_dist  = np.array([0, 2, 5, 8, 11, 14, 17, 20], dtype=float)   # km
    terrain_height = np.array([150, 180, 320, 410, 290, 200, 175, 160], dtype=float)  # m

    x_fine = np.linspace(0, 20, 500)

    print("\n[3.1] Linear Interpolation of Terrain Height Profile")
    lin_interp = interp1d(waypoint_dist, terrain_height, kind='linear')
    y_linear = lin_interp(x_fine)
    print(f"  Data points             : {len(waypoint_dist)}")
    print(f"  Linear interp at x=6 km: {lin_interp(6):.2f} m")

    print("\n[3.2] Cubic Spline Interpolation of Terrain Height Profile")
    cs = CubicSpline(waypoint_dist, terrain_height)
    y_cubic = cs(x_fine)
    print(f"  Cubic spline at x=6 km : {cs(6):.2f} m")
    print(f"  Max terrain height (cubic): {y_cubic.max():.2f} m")

    # ── Live / fallback wind data ─────────────────────────────────────────
    print("\n[3.3] Wind Speed Interpolation Over Route")
    data_label  = "Live Open-Meteo" if weather['source'] == 'live' else "Fallback (offline)"
    print(f"  Data source : {data_label}  |  Location: {weather['location']}")

    dist_sensors = weather['distances_km']   # shape (12,)
    wind_vals    = weather['wind_speeds']     # shape (12,)  real m/s

    cs_wind      = CubicSpline(dist_sensors, wind_vals, bc_type='not-a-knot')
    x_wind_fine  = np.linspace(0, 20, 500)
    y_wind_spline = cs_wind(x_wind_fine)

    # Also build a linear interpolant for comparison
    lin_wind_interp = interp1d(dist_sensors, wind_vals, kind='linear',
                               fill_value='extrapolate')
    mid_points = (dist_sensors[:-1] + dist_sensors[1:]) / 2
    linear_err = np.mean(np.abs(lin_wind_interp(mid_points) - cs_wind(mid_points)))
    print(f"  Wind range  : {wind_vals.min():.2f} – {wind_vals.max():.2f} m/s "
          f"(mean {wind_vals.mean():.2f} m/s)")
    print(f"  Mean |linear − cubic| at midpoints: {linear_err:.4f} m/s")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    src_tag = f"[{data_label}]"
    fig.suptitle(f"Topic 3: Interpolation Techniques for Drone Route  {src_tag}",
                 fontsize=13, fontweight='bold')

    axes[0].plot(waypoint_dist, terrain_height, 'ko', markersize=8,
                 label='Waypoints (survey)', zorder=5)
    axes[0].plot(x_fine, y_linear, 'r--', linewidth=1.5, label='Linear')
    axes[0].plot(x_fine, y_cubic,  'b-',  linewidth=2,   label='Cubic Spline')
    axes[0].fill_between(x_fine, 0, y_cubic, alpha=0.1, color='blue')
    axes[0].set_xlabel("Horizontal Distance (km)")
    axes[0].set_ylabel("Terrain Height (m)")
    axes[0].set_title("Terrain Profile: Linear vs Cubic Spline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(dist_sensors, wind_vals, 'rs', markersize=7,
                 label=f'{data_label} measurements', zorder=5)
    axes[1].plot(x_wind_fine, lin_wind_interp(x_wind_fine), 'g--',
                 linewidth=1.2, label='Linear interp', alpha=0.7)
    axes[1].plot(x_wind_fine, y_wind_spline, 'b-', linewidth=2, label='Cubic Spline')
    axes[1].set_xlabel("Route Distance (km)")
    axes[1].set_ylabel("Wind Speed (m/s)")
    axes[1].set_title(f"Wind Speed Interpolation  ({data_label})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic3_interpolation.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic3_interpolation.png]")

    return cs, cs_wind


# =============================================================================
# TOPIC 4: NUMERICAL DIFFERENTIATION
# =============================================================================

def topic4_differentiation(cs_terrain):
    section_header(4, "NUMERICAL DIFFERENTIATION")

    # Compute slope of terrain (gradient) using finite differences
    # This determines if a drone must climb or descend — affects energy

    x = np.linspace(0, 20, 200)
    h_vals = cs_terrain(x)

    print("\n[4.1] Forward, Backward, Central Difference Schemes")
    dx = x[1] - x[0]

    # Forward difference
    dh_forward = np.zeros_like(h_vals)
    for i in range(len(h_vals) - 1):
        dh_forward[i] = (h_vals[i + 1] - h_vals[i]) / dx
    dh_forward[-1] = dh_forward[-2]  # boundary

    # Backward difference
    dh_backward = np.zeros_like(h_vals)
    dh_backward[0] = dh_forward[0]
    for i in range(1, len(h_vals)):
        dh_backward[i] = (h_vals[i] - h_vals[i - 1]) / dx

    # Central difference
    dh_central = np.zeros_like(h_vals)
    dh_central[0] = dh_forward[0]
    dh_central[-1] = dh_backward[-1]
    for i in range(1, len(h_vals) - 1):
        dh_central[i] = (h_vals[i + 1] - h_vals[i - 1]) / (2 * dx)

    # Analytical derivative via CubicSpline
    dh_exact = cs_terrain(x, 1)  # first derivative

    # Error analysis
    err_forward  = np.max(np.abs(dh_forward  - dh_exact))
    err_backward = np.max(np.abs(dh_backward - dh_exact))
    err_central  = np.max(np.abs(dh_central  - dh_exact))

    print(f"  Step size dx                  : {dx:.4f} km")
    print(f"  Max error - Forward  difference: {err_forward:.4f} m/km")
    print(f"  Max error - Backward difference: {err_backward:.4f} m/km")
    print(f"  Max error - Central  difference: {err_central:.6f} m/km")
    print(f"  Central is ~{err_forward/err_central:.0f}x more accurate than forward")

    print("\n[4.2] Effect of Step Size on Accuracy (h refinement study)")
    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    errors_step = []
    ref_x = 10.0  # reference point at 10 km
    exact_deriv = cs_terrain(ref_x, 1)
    for hh in step_sizes:
        approx = (cs_terrain(ref_x + hh) - cs_terrain(ref_x - hh)) / (2 * hh)
        errors_step.append(abs(approx - exact_deriv))
        print(f"  h = {hh:.3f} km | central diff error = {abs(approx - exact_deriv):.6f} m/km")

    # Identify max slope (most dangerous climbing segment)
    max_slope_idx = np.argmax(np.abs(dh_central))
    print(f"\n  Max terrain slope: {dh_central[max_slope_idx]:.2f} m/km at x={x[max_slope_idx]:.2f} km")
    print(f"  Drone must handle this grade — critical for battery planning")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Topic 4: Numerical Differentiation for Terrain Slope Analysis",
                 fontsize=14, fontweight='bold')

    axes[0, 0].plot(x, h_vals, 'b-', linewidth=2)
    axes[0, 0].set_title("Terrain Elevation Profile")
    axes[0, 0].set_xlabel("Distance (km)")
    axes[0, 0].set_ylabel("Height (m)")
    axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(x, dh_exact,    'k-',  linewidth=2, label='Exact (Spline)', zorder=5)
    axes[0, 1].plot(x, dh_forward,  'r--', linewidth=1.5, label='Forward')
    axes[0, 1].plot(x, dh_backward, 'g--', linewidth=1.5, label='Backward')
    axes[0, 1].plot(x, dh_central,  'b-',  linewidth=1.5, label='Central')
    axes[0, 1].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[0, 1].set_title("Terrain Slope (dh/dx)")
    axes[0, 1].set_xlabel("Distance (km)")
    axes[0, 1].set_ylabel("Slope (m/km)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].plot(x, np.abs(dh_forward - dh_exact),  'r-', label='Forward Error')
    axes[1, 0].plot(x, np.abs(dh_backward - dh_exact), 'g-', label='Backward Error')
    axes[1, 0].plot(x, np.abs(dh_central - dh_exact),  'b-', label='Central Error')
    axes[1, 0].set_title("Pointwise Derivative Error")
    axes[1, 0].set_xlabel("Distance (km)")
    axes[1, 0].set_ylabel("|Error| (m/km)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].loglog(step_sizes, errors_step, 'mo-', linewidth=2, markersize=8)
    # Reference slope lines
    h_ref = np.array(step_sizes)
    axes[1, 1].loglog(h_ref, 0.01 * h_ref**2, 'k--', linewidth=1, label='O(h²) reference')
    axes[1, 1].set_title("Step Size vs Error (Central Difference)")
    axes[1, 1].set_xlabel("Step Size h (km)")
    axes[1, 1].set_ylabel("|Error| (m/km)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.4, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic4_differentiation.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic4_differentiation.png]")

    return dh_central, x


# =============================================================================
# TOPIC 5: NUMERICAL INTEGRATION
# =============================================================================

def topic5_integration(cs_terrain, cs_wind):
    section_header(5, "NUMERICAL INTEGRATION")

    # Physical problem: Compute total energy consumption for a drone traversing
    # a route with varying terrain height and wind speed.
    # Energy = integral of (Power) dt  =>  E = integral of P(x)/v dx  (along route)
    # Power model: P(x) = P_base + k_climb * max(dh/dx, 0) + k_wind * wind(x)^2

    route_length = 20.0  # km
    P_base   = 200.0     # W baseline power
    k_climb  = 50.0      # W per m/km of climb
    k_wind   = 2.0       # W per (m/s)^2 of wind

    def power_function(x_km):
        slope  = cs_terrain(x_km, 1)        # m/km
        wind   = cs_wind(x_km)              # m/s
        climb_power = k_climb * np.maximum(slope, 0)
        wind_power  = k_wind * wind**2
        return P_base + climb_power + wind_power

    x = np.linspace(0, route_length, 1000)
    P_vals = power_function(x)
    dx_km  = x[1] - x[0]
    # Convert km to m and divide by speed (15 m/s) to get seconds
    speed_mps = 15.0
    # energy integral: E = ∫ P(x)/v dx  (x in m)
    # dx in km → ×1000 for meters; divide by speed for time → P*dx_m/v
    P_over_v = P_vals / speed_mps  # W·km/ms... keep consistent units: J/m

    print("\n[5.1] Trapezoidal Rule")
    # E in Joules = ∫ P(x) [W] dx [m] / v [m/s] … already P_over_v [W/ms*km] -> use km*1000
    n_trap = len(x)
    dx_m = dx_km * 1000
    E_trap = np.trapezoid(P_over_v, dx=dx_m)
    print(f"  Total energy (Trapezoidal) : {E_trap:.2f} J  =  {E_trap/3600:.4f} Wh")

    print("\n[5.2] Simpson's Rule")
    # Use scipy.integrate.simpson
    E_simpson = integrate.simpson(P_over_v, dx=dx_m)
    print(f"  Total energy (Simpson's)   : {E_simpson:.2f} J  =  {E_simpson/3600:.4f} Wh")

    print("\n[5.3] Scipy quad (adaptive Gaussian quadrature)")
    def P_integrand(x_km):
        return power_function(x_km) * 1000 / speed_mps  # J/km * km = J
    E_quad, quad_err = integrate.quad(P_integrand, 0, route_length)
    print(f"  Total energy (quad)        : {E_quad:.2f} J  =  {E_quad/3600:.4f} Wh")
    print(f"  quad error estimate        : {quad_err:.4e} J")

    # Error comparison
    ref_E = E_quad
    err_trap    = abs(E_trap - ref_E) / ref_E * 100
    err_simpson = abs(E_simpson - ref_E) / ref_E * 100
    print(f"\n  Trapezoidal error vs quad  : {err_trap:.6f}%")
    print(f"  Simpson's error vs quad    : {err_simpson:.8f}%")

    print("\n[5.4] Convergence: Effect of Number of Intervals")
    n_values = [10, 20, 50, 100, 200, 500, 1000]
    trap_errors, simp_errors = [], []
    for n in n_values:
        x_n = np.linspace(0, route_length, n)
        P_n = power_function(x_n) * 1000 / speed_mps
        dx_n = x_n[1] - x_n[0]
        E_t = np.trapezoid(P_n, dx=dx_n)
        E_s = integrate.simpson(P_n, dx=dx_n)
        trap_errors.append(abs(E_t - ref_E) / ref_E)
        simp_errors.append(abs(E_s - ref_E) / ref_E)
        print(f"  n={n:5d} | Trap error={abs(E_t-ref_E)/ref_E:.6f} | Simp error={abs(E_s-ref_E)/ref_E:.8f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Topic 5: Numerical Integration – Drone Energy Consumption",
                 fontsize=14, fontweight='bold')

    axes[0, 0].plot(x, P_vals, 'purple', linewidth=2)
    axes[0, 0].fill_between(x, 0, P_vals, alpha=0.2, color='purple')
    axes[0, 0].set_title("Power Consumption Along Route")
    axes[0, 0].set_xlabel("Distance (km)")
    axes[0, 0].set_ylabel("Power (W)")
    axes[0, 0].grid(True, alpha=0.4)

    # Trapezoidal illustration on coarse grid
    x_coarse = np.linspace(0, route_length, 12)
    P_coarse  = power_function(x_coarse)
    axes[0, 1].plot(x, P_vals, 'b-', linewidth=1.5, label='P(x)', alpha=0.7)
    axes[0, 1].bar(x_coarse[:-1], P_coarse[:-1], width=np.diff(x_coarse),
                   align='edge', alpha=0.4, color='orange', edgecolor='darkorange', label='Trapezoids')
    axes[0, 1].set_title("Trapezoidal Rule Visualization (n=12)")
    axes[0, 1].set_xlabel("Distance (km)")
    axes[0, 1].set_ylabel("Power (W)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].loglog(n_values, trap_errors, 'r-o', linewidth=2, markersize=7, label="Trapezoidal")
    axes[1, 0].loglog(n_values, simp_errors, 'b-s', linewidth=2, markersize=7, label="Simpson's")
    n_arr = np.array(n_values, dtype=float)
    axes[1, 0].loglog(n_arr, 0.1 / n_arr**2, 'k--', linewidth=1, label='O(n⁻²)')
    axes[1, 0].loglog(n_arr, 0.01 / n_arr**4, 'g--', linewidth=1, label='O(n⁻⁴)')
    axes[1, 0].set_title("Integration Convergence Rate")
    axes[1, 0].set_xlabel("Number of Intervals n")
    axes[1, 0].set_ylabel("Relative Error")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.4, which='both')

    methods = ['Trapezoidal', "Simpson's", 'Quad (ref)']
    energies = [E_trap / 3600, E_simpson / 3600, E_quad / 3600]
    colors_bar = ['tomato', 'steelblue', 'forestgreen']
    bars = axes[1, 1].bar(methods, energies, color=colors_bar, edgecolor='black', width=0.5)
    axes[1, 1].set_title("Energy Estimate Comparison")
    axes[1, 1].set_ylabel("Energy (Wh)")
    axes[1, 1].set_ylim(min(energies) * 0.999, max(energies) * 1.001)
    for bar, val in zip(bars, energies):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    axes[1, 1].grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic5_integration.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic5_integration.png]")

    return E_quad


# =============================================================================
# TOPIC 6: SOLVING LINEAR SYSTEMS
# =============================================================================

def topic6_linear_systems():
    section_header(6, "SOLVING LINEAR SYSTEMS")

    # Problem: Solve drone fleet allocation across delivery zones
    # Each zone has demand, and each warehouse has capacity
    # The allocation satisfies demand constraints via a linear system A*x = b
    # We use a realistic 6×6 system

    print("\n[6.1] Drone Fleet Allocation Problem (6 zones, 6 warehouses)")
    # A[i][j] = fraction of warehouse j's capacity allocated to zone i
    np.random.seed(7)
    n = 6
    # Create a diagonally dominant matrix for guaranteed solvability
    A = np.random.uniform(0.1, 1.0, (n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0   # diagonal dominance

    # b = demand vector (number of deliveries per hour)
    b = np.array([30.0, 45.0, 20.0, 55.0, 35.0, 25.0])

    print(f"  System size: {n}×{n}")
    print(f"  Condition number: {np.linalg.cond(A):.4f}")

    # Direct solve using numpy
    t0 = time.perf_counter()
    x_direct = np.linalg.solve(A, b)
    t_direct = time.perf_counter() - t0
    residual_direct = np.linalg.norm(A @ x_direct - b)
    print(f"\n  numpy.linalg.solve:")
    print(f"    Time        : {t_direct * 1e6:.2f} μs")
    print(f"    Residual ‖Ax-b‖: {residual_direct:.4e}")
    print(f"    Solution x  : {x_direct}")

    # Verify via matrix inverse (educational, not recommended for production)
    A_inv = np.linalg.inv(A)
    x_inv = A_inv @ b
    residual_inv = np.linalg.norm(A @ x_inv - b)
    print(f"\n  Via matrix inverse:")
    print(f"    Residual ‖Ax-b‖: {residual_inv:.4e}")

    # Scale up and compare performance
    print("\n[6.2] Performance vs Matrix Size")
    sizes = [10, 50, 100, 200, 500]
    times_direct = []
    cond_numbers = []
    for sz in sizes:
        A_big = np.random.uniform(0.1, 1.0, (sz, sz))
        for i in range(sz):
            A_big[i, i] = np.sum(np.abs(A_big[i, :])) + 1.0
        b_big = np.random.rand(sz) * 100
        t0 = time.perf_counter()
        _ = np.linalg.solve(A_big, b_big)
        t_elapsed = time.perf_counter() - t0
        times_direct.append(t_elapsed * 1000)  # ms
        cond_numbers.append(np.linalg.cond(A_big))
        print(f"  n={sz:4d} | solve time={t_elapsed*1000:.3f} ms | cond={np.linalg.cond(A_big):.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Topic 6: Linear Systems – Drone Fleet Allocation", fontsize=14, fontweight='bold')

    # Bar plot of solution (allocation per warehouse)
    warehouse_labels = [f'W{i+1}' for i in range(n)]
    axes[0].bar(warehouse_labels, x_direct, color='steelblue', edgecolor='black')
    axes[0].set_title("Drone Allocation per Warehouse")
    axes[0].set_xlabel("Warehouse")
    axes[0].set_ylabel("Allocation (drones/hour)")
    axes[0].grid(True, alpha=0.4, axis='y')

    # Residual heatmap
    im = axes[1].imshow(A, cmap='viridis', aspect='auto')
    axes[1].set_title("Coefficient Matrix A (Heatmap)")
    axes[1].set_xlabel("Column (Warehouse)")
    axes[1].set_ylabel("Row (Zone)")
    plt.colorbar(im, ax=axes[1])

    # Solve time vs size
    axes[2].plot(sizes, times_direct, 'ro-', linewidth=2, markersize=8)
    axes[2].set_title("Solve Time vs Matrix Size")
    axes[2].set_xlabel("Matrix Size n")
    axes[2].set_ylabel("Time (ms)")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic6_linear_systems.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic6_linear_systems.png]")

    return A, b


# =============================================================================
# TOPIC 7: LU DECOMPOSITION FOR EFFICIENT CALCULATIONS
# =============================================================================

def topic7_lu_decomposition(A, b):
    section_header(7, "LU DECOMPOSITION FOR EFFICIENT CALCULATIONS")

    print("\n[7.1] LU Factorization for Multi-Scenario Fleet Planning")
    print("  Scenario: Same drone allocation matrix, 10 different demand scenarios")
    print("  LU factorizes A once, then solves each RHS cheaply")

    # Factorize once
    t0 = time.perf_counter()
    lu, piv = la.lu_factor(A)
    t_factor = time.perf_counter() - t0
    print(f"\n  LU factorization time : {t_factor * 1e6:.2f} μs")

    # Multiple right-hand side vectors (different demand scenarios)
    np.random.seed(12)
    n_scenarios = 10
    B = np.random.uniform(20, 60, (len(b), n_scenarios))

    # Solve with LU (reuse factorization)
    t0 = time.perf_counter()
    solutions_lu = np.array([la.lu_solve((lu, piv), B[:, j]) for j in range(n_scenarios)])
    t_lu = time.perf_counter() - t0

    # Solve directly (each time)
    t0 = time.perf_counter()
    solutions_direct = np.array([np.linalg.solve(A, B[:, j]) for j in range(n_scenarios)])
    t_direct = time.perf_counter() - t0

    print(f"  LU solve ({n_scenarios} RHS)       : {t_lu * 1e6:.2f} μs total")
    print(f"  Direct solve ({n_scenarios} times)  : {t_direct * 1e6:.2f} μs total")
    if t_lu > 0:
        speedup = t_direct / t_lu
        print(f"  Speedup factor         : {speedup:.2f}x")

    # Verify solutions match
    max_diff = np.max(np.abs(solutions_lu - solutions_direct))
    print(f"  Max solution difference: {max_diff:.4e}")

    print("\n[7.2] Scaling Study: LU vs Direct for Repeated Solves")
    matrix_sizes = [20, 50, 100, 200]
    n_rhs_list = [5, 10, 20, 50]
    print(f"  {'Size':<8} {'n_RHS':<8} {'Direct(ms)':<14} {'LU(ms)':<14} {'Speedup':<10}")
    print(f"  {'-'*54}")
    for sz in [50, 100]:
        Abig = np.random.uniform(0.1, 1.0, (sz, sz))
        for i in range(sz):
            Abig[i, i] = np.sum(np.abs(Abig[i, :])) + 1.0
        lu_b, piv_b = la.lu_factor(Abig)
        for n_rhs in [10, 50]:
            Bb = np.random.rand(sz, n_rhs)
            t0 = time.perf_counter()
            for j in range(n_rhs):
                np.linalg.solve(Abig, Bb[:, j])
            td = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            for j in range(n_rhs):
                la.lu_solve((lu_b, piv_b), Bb[:, j])
            tl = (time.perf_counter() - t0) * 1000
            print(f"  {sz:<8} {n_rhs:<8} {td:<14.3f} {tl:<14.3f} {td/tl:<10.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Topic 7: LU Decomposition – Efficient Multi-Scenario Solving",
                 fontsize=14, fontweight='bold')

    scenarios = [f'S{i+1}' for i in range(n_scenarios)]
    x_pos = np.arange(n_scenarios)
    width = 0.35
    # Plot first warehouse allocation across scenarios
    axes[0].bar(x_pos - width/2, solutions_lu[:, 0], width, label='LU solve', color='steelblue')
    axes[0].bar(x_pos + width/2, solutions_direct[:, 0], width, label='Direct', color='tomato', alpha=0.7)
    axes[0].set_title("Warehouse 1 Allocation per Scenario")
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Allocation")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(scenarios, fontsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.4, axis='y')

    # L and U heatmaps
    P_mat, L_mat, U_mat = la.lu(A)
    im_l = axes[1].imshow(np.abs(L_mat), cmap='Blues', aspect='auto')
    axes[1].set_title("L Matrix (|values|)")
    plt.colorbar(im_l, ax=axes[1])

    im_u = axes[2].imshow(np.abs(U_mat), cmap='Oranges', aspect='auto')
    axes[2].set_title("U Matrix (|values|)")
    plt.colorbar(im_u, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic7_lu_decomposition.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic7_lu_decomposition.png]")


# =============================================================================
# TOPIC 8: OPTIMIZATION TECHNIQUES
# =============================================================================

def topic8_optimization():
    section_header(8, "OPTIMIZATION TECHNIQUES")

    print("\n[8.1] Scalar Optimization: Optimal Drone Speed for Minimum Energy")
    # Energy model: E(v) = (P_drag(v) + P_motor) * time
    # P_drag = 0.5*rho*Cd*A*v^3/eta (propulsive power ~ v^3 in some models)
    # Flight time = distance / v
    # => E(v) = (a*v^2 + b/v) * distance  [simplified model]
    a_coeff = 0.5    # drag coefficient term (W·s²/m²)  — realistic aerodynamic drag
    b_coeff = 500.0  # induced power term (W·m)
    distance = 10000  # meters

    def energy(v):
        if v <= 0:
            return 1e12
        return (a_coeff * v**2 + b_coeff / v) * distance / v

    def energy_np(v):
        return (a_coeff * v**2 + b_coeff / v) * distance / v

    result = opt.minimize_scalar(energy, bounds=(2, 50), method='bounded')
    v_opt = result.x
    E_opt = result.fun
    print(f"  Optimal speed   : {v_opt:.4f} m/s  ({v_opt*3.6:.2f} km/h)")
    print(f"  Minimum energy  : {E_opt:.2f} J  ({E_opt/3600:.4f} Wh)")

    # Analytical optimum: dE/dv = 0 => v_opt = (b/(2*a))^(1/3)
    v_analytical = (b_coeff / (2 * a_coeff))**(1/3)
    print(f"  Analytical v_opt: {v_analytical:.4f} m/s  (error: {abs(v_opt - v_analytical):.6f})")

    print("\n[8.2] Multivariable Optimization: Optimal Waypoint Placement")
    # Minimize total route distance with terrain clearance penalty
    # 3 intermediate waypoints between (0,0) and (10,10) on a 2D plane
    # with hills at (3,3) and (7,7) → must route around them
    start = np.array([0.0, 0.0])
    end   = np.array([10.0, 10.0])
    hills = [(3.0, 3.0, 2.0), (7.0, 7.0, 1.5)]  # (cx, cy, radius)

    def route_cost(params):
        # params: [x1,y1, x2,y2, x3,y3]
        p1 = np.array([params[0], params[1]])
        p2 = np.array([params[2], params[3]])
        p3 = np.array([params[4], params[5]])
        pts = [start, p1, p2, p3, end]
        distance_total = sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(4))
        # Penalty for being inside hills
        penalty = 0.0
        for cx, cy, r in hills:
            for pt in [p1, p2, p3]:
                d = np.sqrt((pt[0]-cx)**2 + (pt[1]-cy)**2)
                if d < r:
                    penalty += 100 * (r - d)**2
        return distance_total + penalty

    x0 = np.array([2.5, 2.5, 5.0, 5.0, 7.5, 7.5])
    result_mv = opt.minimize(route_cost, x0, method='Nelder-Mead',
                              options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
    p_opt = result_mv.x
    print(f"  Optimal waypoints:")
    print(f"    WP1 = ({p_opt[0]:.3f}, {p_opt[1]:.3f})")
    print(f"    WP2 = ({p_opt[2]:.3f}, {p_opt[3]:.3f})")
    print(f"    WP3 = ({p_opt[4]:.3f}, {p_opt[5]:.3f})")
    print(f"  Minimum route cost: {result_mv.fun:.4f} km")
    print(f"  Optimizer converged: {result_mv.success}")

    # Compare methods
    print("\n[8.3] Comparing Optimization Algorithms")
    methods_mv = ['Nelder-Mead', 'Powell', 'COBYLA']
    for m in methods_mv:
        try:
            res = opt.minimize(route_cost, x0, method=m,
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            print(f"  {m:<15}: cost={res.fun:.4f}, iters={res.nit}, success={res.success}")
        except Exception as e:
            print(f"  {m:<15}: FAILED – {e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Topic 8: Optimization – Speed and Route Planning", fontsize=14, fontweight='bold')

    v_range = np.linspace(3, 50, 300)
    E_range = energy_np(v_range)
    axes[0].plot(v_range, E_range / 3600, 'b-', linewidth=2, label='E(v)')
    axes[0].axvline(v_opt, color='red', linestyle='--', linewidth=2, label=f'v_opt={v_opt:.2f} m/s')
    axes[0].axvline(v_analytical, color='green', linestyle=':', linewidth=2, label=f'Analytical={v_analytical:.2f} m/s')
    axes[0].set_xlabel("Speed (m/s)")
    axes[0].set_ylabel("Energy (Wh)")
    axes[0].set_title("Energy vs Speed – Find Optimal Cruise Speed")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Route optimization map
    grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    cost_map = np.zeros_like(grid_x)
    for cx, cy, r in hills:
        dist_hill = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
        cost_map += np.exp(-dist_hill**2 / (2 * r**2)) * 5
    axes[1].contourf(grid_x, grid_y, cost_map, levels=20, cmap='YlOrRd', alpha=0.6)
    # Draw hill circles
    for cx, cy, r in hills:
        circle = plt.Circle((cx, cy), r, fill=False, color='darkred', linewidth=2)
        axes[1].add_patch(circle)
    # Draw optimized route
    route_pts = np.array([start,
                           [p_opt[0], p_opt[1]],
                           [p_opt[2], p_opt[3]],
                           [p_opt[4], p_opt[5]],
                           end])
    axes[1].plot(route_pts[:, 0], route_pts[:, 1], 'b-o', linewidth=2.5,
                 markersize=8, label='Optimized Route', zorder=5)
    # Draw naive straight route
    axes[1].plot([start[0], end[0]], [start[1], end[1]], 'k--',
                 linewidth=1.5, label='Straight Line', alpha=0.6)
    axes[1].plot(*start, 'gs', markersize=12, label='Start', zorder=6)
    axes[1].plot(*end,   'r^', markersize=12, label='End',   zorder=6)
    axes[1].set_title("Route Optimization Around Obstacles")
    axes[1].set_xlabel("X (km)")
    axes[1].set_ylabel("Y (km)")
    axes[1].legend(fontsize=8)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic8_optimization.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic8_optimization.png]")

    return v_opt, p_opt


# =============================================================================
# TOPIC 9: ORDINARY DIFFERENTIAL EQUATIONS (ODEs) SOLVING
# =============================================================================

def topic9_odes():
    section_header(9, "ORDINARY DIFFERENTIAL EQUATIONS (ODEs) SOLVING")

    print("\n[9.1] Drone Flight Dynamics – Full 2D ODE Model")
    # State: [x, y, vx, vy]  (position and velocity in 2D plane)
    # dx/dt = vx,  dy/dt = vy
    # dvx/dt = (F_thrust_x - F_drag_x) / m
    # dvy/dt = (F_thrust_y - F_drag_y - m*g) / m
    m    = 2.5     # kg
    g    = 9.81    # m/s^2
    Cd   = 0.3
    A    = 0.05    # m^2
    rho0 = 1.225
    H_sc = 8500.0

    # Thrust profile: climb for 20s, cruise for 60s, descend for 20s
    def thrust(t):
        if t < 20:
            return np.array([10.0, 20.0])   # climb phase
        elif t < 80:
            return np.array([15.0, 0.0])    # cruise
        else:
            return np.array([5.0, -15.0])   # descend

    def drone_dynamics(t, state):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2) + 1e-10
        rho = rho0 * np.exp(-max(y, 0) / H_sc)
        F_drag = 0.5 * rho * Cd * A * v_mag**2
        F_d_x = -F_drag * vx / v_mag
        F_d_y = -F_drag * vy / v_mag
        T = thrust(t)
        ax = (T[0] + F_d_x) / m
        ay = (T[1] + F_d_y - m * g) / m
        return [vx, vy, ax, ay]

    state0 = [0.0, 100.0, 0.0, 0.0]  # start at 100m altitude
    t_span = (0, 100)
    t_eval = np.linspace(0, 100, 500)

    print("  Solving with RK45 (default adaptive solver)")
    t0 = time.perf_counter()
    sol_rk45 = solve_ivp(drone_dynamics, t_span, state0, method='RK45',
                          t_eval=t_eval, rtol=1e-6, atol=1e-9)
    t_rk45 = time.perf_counter() - t0
    print(f"    RK45 solve time : {t_rk45*1000:.2f} ms, steps={len(sol_rk45.t)}")
    print(f"    Final position  : x={sol_rk45.y[0, -1]:.1f} m, y={sol_rk45.y[1, -1]:.1f} m")

    print("\n  Solving with RK23")
    t0 = time.perf_counter()
    sol_rk23 = solve_ivp(drone_dynamics, t_span, state0, method='RK23',
                          t_eval=t_eval, rtol=1e-4, atol=1e-7)
    t_rk23 = time.perf_counter() - t0
    print(f"    RK23 solve time : {t_rk23*1000:.2f} ms")

    print("\n  Solving with DOP853")
    t0 = time.perf_counter()
    sol_dop853 = solve_ivp(drone_dynamics, t_span, state0, method='DOP853',
                            t_eval=t_eval, rtol=1e-8, atol=1e-12)
    t_dop853 = time.perf_counter() - t0
    print(f"    DOP853 solve time: {t_dop853*1000:.2f} ms")

    # Error comparison (use DOP853 as reference)
    ref_y = sol_dop853.y[1]  # altitude
    err_rk45 = np.max(np.abs(sol_rk45.y[1] - ref_y))
    err_rk23 = np.max(np.abs(sol_rk23.y[1] - ref_y))
    print(f"\n  Max altitude error vs DOP853:")
    print(f"    RK45: {err_rk45:.4e} m")
    print(f"    RK23: {err_rk23:.4e} m")

    print("\n[9.2] Battery Discharge ODE")
    # dC/dt = -I(t) / 3600   (C in Ah, I = current in A)
    # I(t) depends on power demand: I = P(v(t)) / V_battery
    V_bat = 22.2  # volts (6S LiPo)
    C0 = 5.0      # Ah initial capacity

    def battery_ode(t, C):
        # Power at time t based on current flight phase
        if t < 20:
            P = 350.0  # climb
        elif t < 80:
            P = 200.0  # cruise
        else:
            P = 120.0  # descend
        I = P / V_bat
        return [-I / 3600]

    sol_bat = solve_ivp(battery_ode, (0, 100), [C0], method='RK45',
                         t_eval=t_eval, dense_output=True)
    C_remaining = sol_bat.y[0]
    print(f"  Initial capacity : {C0:.2f} Ah")
    print(f"  Final capacity   : {C_remaining[-1]:.4f} Ah")
    print(f"  Energy used      : {(C0 - C_remaining[-1]) * V_bat:.2f} Wh")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Topic 9: ODE Solving – Drone Flight Dynamics & Battery",
                 fontsize=14, fontweight='bold')

    t = sol_rk45.t

    axes[0, 0].plot(sol_rk45.y[0], sol_rk45.y[1], 'b-', linewidth=2, label='RK45')
    axes[0, 0].plot(sol_rk23.y[0], sol_rk23.y[1], 'r--', linewidth=1.5, label='RK23')
    axes[0, 0].plot(sol_dop853.y[0], sol_dop853.y[1], 'g:', linewidth=2, label='DOP853')
    axes[0, 0].set_title("Drone Trajectory (x-y plane)")
    axes[0, 0].set_xlabel("Horizontal Position (m)")
    axes[0, 0].set_ylabel("Altitude (m)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(t, sol_rk45.y[1], 'b-', linewidth=2, label='RK45')
    axes[0, 1].plot(t, sol_rk23.y[1], 'r--', linewidth=1.5, label='RK23')
    axes[0, 1].set_title("Altitude vs Time")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Altitude (m)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    speed = np.sqrt(sol_rk45.y[2]**2 + sol_rk45.y[3]**2)
    axes[0, 2].plot(t, speed, 'purple', linewidth=2)
    axes[0, 2].set_title("Speed vs Time")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Speed (m/s)")
    axes[0, 2].grid(True, alpha=0.4)

    axes[1, 0].plot(t, sol_bat.y[0], 'darkorange', linewidth=2)
    axes[1, 0].axhline(C0 * 0.2, color='red', linestyle='--', label='20% reserve')
    axes[1, 0].set_title("Battery Capacity vs Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Capacity (Ah)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].semilogy(t[1:], np.abs(sol_rk45.y[1, 1:] - ref_y[1:]) + 1e-15, 'b-', label='RK45 err')
    axes[1, 1].semilogy(t[1:], np.abs(sol_rk23.y[1, 1:] - ref_y[1:]) + 1e-15, 'r--', label='RK23 err')
    axes[1, 1].set_title("Altitude Error vs DOP853 (log scale)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("|Error| (m)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.4)

    solver_names = ['RK45', 'RK23', 'DOP853']
    solver_times = [t_rk45 * 1000, t_rk23 * 1000, t_dop853 * 1000]
    axes[1, 2].bar(solver_names, solver_times, color=['steelblue', 'tomato', 'seagreen'],
                   edgecolor='black')
    axes[1, 2].set_title("ODE Solver Speed Comparison")
    axes[1, 2].set_ylabel("Solve Time (ms)")
    axes[1, 2].grid(True, alpha=0.4, axis='y')
    for i, (name, val) in enumerate(zip(solver_names, solver_times)):
        axes[1, 2].text(i, val, f'{val:.2f}ms', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic9_odes.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic9_odes.png]")

    return sol_rk45


# =============================================================================
# TOPIC 10: PERFORMANCE ANALYSIS, NUMERICAL STABILITY, AND ERROR HANDLING
# =============================================================================

def topic10_performance_stability():
    section_header(10, "PERFORMANCE ANALYSIS, NUMERICAL STABILITY & ERROR HANDLING")

    print("\n[10.1] Timing Benchmark: Key Numerical Operations")
    operations = {}

    # Matrix operations
    for n in [100, 500]:
        A = np.random.rand(n, n)
        A += n * np.eye(n)
        b_vec = np.random.rand(n)
        t0 = time.perf_counter()
        for _ in range(20):
            np.linalg.solve(A, b_vec)
        t_avg = (time.perf_counter() - t0) / 20 * 1000
        operations[f'linalg.solve n={n}'] = t_avg
        print(f"  linalg.solve (n={n})      : {t_avg:.4f} ms avg over 20 runs")

    # Integration timing
    def f_test(x):
        return np.sin(x) * np.exp(-x)
    t0 = time.perf_counter()
    for _ in range(1000):
        integrate.quad(f_test, 0, 10)
    t_quad = (time.perf_counter() - t0) / 1000 * 1000
    operations['quad integration'] = t_quad
    print(f"  quad integration          : {t_quad:.4f} ms avg over 1000 runs")

    # Root finding timing
    def f_root(x):
        return x**3 - 2*x - 5
    t0 = time.perf_counter()
    for _ in range(1000):
        opt.brentq(f_root, 2, 3)
    t_root = (time.perf_counter() - t0) / 1000 * 1000
    operations['brentq root finding'] = t_root
    print(f"  brentq root finding       : {t_root:.6f} ms avg over 1000 runs")

    print("\n[10.2] Numerical Stability: Ill-Conditioned System")
    # Hilbert matrix: famously ill-conditioned
    for n in [5, 8, 10, 12]:
        try:
            H = np.array([[1.0 / (i + j - 1) for j in range(1, n + 1)]
                          for i in range(1, n + 1)])
            b_h = np.ones(n)
            x_h = np.linalg.solve(H, b_h)
            residual = np.linalg.norm(H @ x_h - b_h)
            cond = np.linalg.cond(H)
            print(f"  Hilbert n={n:2d}: cond={cond:.2e}, residual={residual:.2e}")
        except np.linalg.LinAlgError as e:
            print(f"  Hilbert n={n:2d}: SINGULAR – {e}")

    print("\n[10.3] Robust Error Handling with try-except")
    test_cases = [
        ("Valid root finding",    lambda: opt.brentq(lambda x: x**2 - 4, 1, 3)),
        ("No root in interval",   lambda: opt.brentq(lambda x: x**2 + 1, -1, 1)),
        ("Singular matrix solve", lambda: np.linalg.solve(np.zeros((3, 3)), [1, 2, 3])),
        ("Division by zero",      lambda: 1.0 / 0.0),
        ("Valid integration",     lambda: integrate.quad(np.sin, 0, np.pi)),
    ]
    for name, operation in test_cases:
        try:
            result = operation()
            if isinstance(result, tuple):
                print(f"  ✓ {name}: result={result[0]:.4f}")
            else:
                print(f"  ✓ {name}: result={result:.4f}")
        except ValueError as e:
            print(f"  ✗ {name}: ValueError – {e}")
        except np.linalg.LinAlgError as e:
            print(f"  ✗ {name}: LinAlgError – {e}")
        except ZeroDivisionError as e:
            print(f"  ✗ {name}: ZeroDivisionError – {e}")
        except Exception as e:
            print(f"  ✗ {name}: Exception – {e}")

    print("\n[10.4] Adaptive Step Size Demonstration (ODE)")
    def stiff_ode(t, y):
        return [-1000 * y[0] + 3000 - 2000 * np.exp(-t)]

    sol_stiff_adaptive = solve_ivp(stiff_ode, (0, 1), [0.0], method='Radau',
                                    dense_output=True)
    sol_stiff_explicit = solve_ivp(stiff_ode, (0, 1), [0.0], method='RK45',
                                    dense_output=True)
    print(f"  Radau (implicit, stiff-suited): nsteps={len(sol_stiff_adaptive.t)}, "
          f"final y={sol_stiff_adaptive.y[0, -1]:.6f}")
    print(f"  RK45  (explicit, not stiff):   nsteps={len(sol_stiff_explicit.t)}, "
          f"final y={sol_stiff_explicit.y[0, -1]:.6f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Topic 10: Performance Analysis, Stability & Error Handling",
                 fontsize=14, fontweight='bold')

    # Timing bar chart
    op_names = list(operations.keys())
    op_times = list(operations.values())
    axes[0, 0].barh(op_names, op_times, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_title("Numerical Operation Benchmark")
    axes[0, 0].grid(True, alpha=0.4, axis='x')

    # Condition number vs Hilbert matrix size
    sizes_h = range(2, 14)
    conds_h = []
    for sz in sizes_h:
        H = np.array([[1.0 / (i + j - 1) for j in range(1, sz + 1)] for i in range(1, sz + 1)])
        conds_h.append(np.linalg.cond(H))
    axes[0, 1].semilogy(list(sizes_h), conds_h, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].axhline(1.0 / np.finfo(float).eps, color='red', linestyle='--',
                        label='Machine precision limit', linewidth=1.5)
    axes[0, 1].set_title("Hilbert Matrix Condition Number vs Size")
    axes[0, 1].set_xlabel("Matrix Size")
    axes[0, 1].set_ylabel("Condition Number (log scale)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.4)

    # Stiff ODE comparison
    t_fine = np.linspace(0, 1, 300)
    axes[1, 0].plot(sol_stiff_adaptive.t, sol_stiff_adaptive.y[0], 'b-', linewidth=2, label='Radau')
    axes[1, 0].plot(sol_stiff_explicit.t, sol_stiff_explicit.y[0], 'r--', linewidth=1.5, label='RK45')
    axes[1, 0].set_title("Stiff ODE: Radau vs RK45")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("y(t)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.4)

    # Adaptive step sizes
    steps_radau = np.diff(sol_stiff_adaptive.t)
    steps_rk45  = np.diff(sol_stiff_explicit.t)
    axes[1, 1].semilogy(sol_stiff_adaptive.t[1:], steps_radau, 'b-',
                         label=f'Radau (n={len(sol_stiff_adaptive.t)})', linewidth=1.5)
    axes[1, 1].semilogy(sol_stiff_explicit.t[1:], steps_rk45, 'r-',
                         label=f'RK45 (n={len(sol_stiff_explicit.t)})', linewidth=1, alpha=0.7)
    axes[1, 1].set_title("Adaptive Step Sizes (log scale)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Step Size (s)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic10_performance.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic10_performance.png]")


# =============================================================================
# TOPIC 11: VISUALIZATION AND DOCUMENTATION
# =============================================================================

def topic11_visualization(sol_ode, v_opt, weather: dict):
    """
    Full system dashboard.

    Parameters
    ----------
    weather : dict from live_weather.fetch_wind_profile()
    """
    section_header(11, "VISUALIZATION AND DOCUMENTATION")

    print("\n[11.1] Full System Dashboard – Drone Delivery Summary")
    # Terrain & waypoints
    wp_dist  = np.array([0, 2, 5, 8, 11, 14, 17, 20], dtype=float)
    wp_height= np.array([150, 180, 320, 410, 290, 200, 175, 160], dtype=float)
    cs_t     = CubicSpline(wp_dist, wp_height)
    x_route  = np.linspace(0, 20, 500)
    h_route  = cs_t(x_route)
    slope_route = cs_t(x_route, 1)

    # Wind speed — live or fallback
    data_label = "Live Open-Meteo" if weather['source'] == 'live' else "Fallback (offline)"
    dist_sensors = weather['distances_km']
    wind_vals    = weather['wind_speeds']
    cs_wind      = CubicSpline(dist_sensors, wind_vals, bc_type='not-a-knot')
    wind_route   = cs_wind(x_route)

    # Power consumption
    P_base = 200; k_climb = 50; k_wind = 2
    P_route = P_base + k_climb * np.maximum(slope_route, 0) + k_wind * wind_route**2

    fig = plt.figure(figsize=(16, 12))
    src_tag = f"  [{data_label}]"
    fig.suptitle(f"DRONE DELIVERY ROUTE OPTIMIZATION — Full System Dashboard{src_tag}",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Terrain profile
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(x_route, 0, h_route, alpha=0.3, color='brown')
    ax1.plot(x_route, h_route, 'brown', linewidth=2, label='Terrain (Cubic Spline)')
    ax1.plot(wp_dist, wp_height, 'ko', markersize=8, label='Waypoints', zorder=5)
    # Drone safe altitude (terrain + clearance)
    clearance = 50
    ax1.fill_between(x_route, h_route, h_route + clearance, alpha=0.2, color='skyblue')
    ax1.plot(x_route, h_route + clearance, 'b--', linewidth=1.5, label=f'Drone path (+{clearance}m)')
    ax1.set_title("Terrain Profile & Drone Flight Path")
    ax1.set_xlabel("Route Distance (km)")
    ax1.set_ylabel("Altitude (m)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4)

    # 2. Wind speed
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(x_route, wind_route, 'teal', linewidth=2)
    ax2.fill_between(x_route, 0, wind_route, alpha=0.2, color='teal')
    ax2.set_title(f"Wind Speed ({data_label})")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Wind (m/s)")
    ax2.grid(True, alpha=0.4)

    # 3. Power consumption
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(x_route, P_route, 'purple', linewidth=2, label='Total Power')
    ax3.axhline(P_base, color='gray', linestyle='--', linewidth=1, label=f'Baseline {P_base}W')
    ax3.fill_between(x_route, P_base, P_route, where=P_route > P_base,
                     alpha=0.3, color='orange', label='Extra power')
    ax3.set_title("Power Consumption Along Route")
    ax3.set_xlabel("Route Distance (km)")
    ax3.set_ylabel("Power (W)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.4)

    # 4. Terrain slope
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(x_route, slope_route, 'darkorange', linewidth=2)
    ax4.axhline(0, color='k', linewidth=1)
    ax4.fill_between(x_route, 0, slope_route, where=slope_route > 0,
                     alpha=0.3, color='red', label='Climb')
    ax4.fill_between(x_route, 0, slope_route, where=slope_route < 0,
                     alpha=0.3, color='green', label='Descend')
    ax4.set_title("Terrain Slope (dh/dx)")
    ax4.set_xlabel("Distance (km)")
    ax4.set_ylabel("Slope (m/km)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.4)

    # 5. ODE: Drone altitude from Topic 9
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(sol_ode.t, sol_ode.y[1], 'b-', linewidth=2, label='Altitude (ODE)')
    ax5_twin = ax5.twinx()
    speed_ode = np.sqrt(sol_ode.y[2]**2 + sol_ode.y[3]**2)
    ax5_twin.plot(sol_ode.t, speed_ode, 'r--', linewidth=1.5, label='Speed')
    ax5_twin.set_ylabel("Speed (m/s)", color='red')
    ax5.set_title("Dynamic Flight: Altitude & Speed vs Time")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Altitude (m)", color='blue')
    ax5.legend(loc='upper left', fontsize=8)
    ax5_twin.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.4)

    # 6. Summary stats box
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    summary_text = [
        "SUMMARY STATISTICS",
        "─" * 26,
        f"Data source   : {data_label}",
        f"Location      : {weather['location'][:24]}",
        f"Temp / RH     : {weather['temperature_c']:.1f}°C / {weather['humidity_pct']:.0f}%",
        f"Air density   : {weather['air_density']:.4f} kg/m³",
        "─" * 26,
        f"Route length  : 20 km",
        f"Max terrain   : {h_route.max():.0f} m",
        f"Max slope     : {slope_route.max():.1f} m/km",
        f"Avg wind      : {wind_route.mean():.1f} m/s",
        f"Avg power     : {P_route.mean():.1f} W",
        f"Max power     : {P_route.max():.1f} W",
        f"Optimal speed : {v_opt:.2f} m/s",
        f"Optimal speed : {v_opt*3.6:.1f} km/h",
    ]
    ax6.text(0.05, 0.95, "\n".join(summary_text),
             transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(PLOT_DIR, 'topic11_dashboard.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic11_dashboard.png]")
    print("  Full system visualization dashboard generated successfully.")


# =============================================================================
# TOPIC 12: COMPARATIVE ANALYSIS AND CASE STUDY
# =============================================================================

def topic12_comparative_analysis(weather: dict):
    """
    Comparative analysis and case study.

    Parameters
    ----------
    weather : dict from live_weather.fetch_wind_profile()
    """
    section_header(12, "COMPARATIVE ANALYSIS AND CASE STUDY")

    print("\n[12.1] Case Study: Total Route Energy – Multiple Integration Methods")
    data_label = "Live Open-Meteo" if weather['source'] == 'live' else "Fallback (offline)"
    print(f"  Wind data source: {data_label}  |  {weather['location']}")

    wp_dist  = np.array([0, 2, 5, 8, 11, 14, 17, 20], dtype=float)
    wp_height= np.array([150, 180, 320, 410, 290, 200, 175, 160], dtype=float)
    cs_t     = CubicSpline(wp_dist, wp_height)

    dist_sensors = weather['distances_km']
    wind_vals    = weather['wind_speeds']
    cs_wind      = CubicSpline(dist_sensors, wind_vals, bc_type='not-a-knot')

    P_base = 200; k_climb = 50; k_wind = 2; v_speed = 15.0

    def power_func(x_km):
        slope = cs_t(x_km, 1)
        wind  = cs_wind(x_km)
        return P_base + k_climb * np.maximum(slope, 0) + k_wind * wind**2

    def integrand(x_km):
        return power_func(x_km) * 1000 / v_speed  # J

    # Reference
    E_ref, _ = integrate.quad(integrand, 0, 20)

    print(f"\n  Reference energy (quad): {E_ref:.4f} J  =  {E_ref/3600:.6f} Wh")
    print(f"\n  {'Method':<25} {'Energy (J)':<16} {'Error (%)':<14} {'Time (μs)'}")
    print(f"  {'-'*70}")

    methods_comp = {}
    for n in [20, 100, 500]:
        x_n  = np.linspace(0, 20, n)
        P_n  = np.array([integrand(x) for x in x_n])
        dx_n = x_n[1] - x_n[0]

        t0 = time.perf_counter(); E_t = np.trapezoid(P_n, dx=dx_n); tt = (time.perf_counter()-t0)*1e6
        err_t = abs(E_t - E_ref) / E_ref * 100
        methods_comp[f'Trapz n={n}'] = (E_t, err_t, tt)
        print(f"  {'Trapezoidal n='+str(n):<25} {E_t:<16.4f} {err_t:<14.6f} {tt:.2f}")

        t0 = time.perf_counter(); E_s = integrate.simpson(P_n, dx=dx_n); ts = (time.perf_counter()-t0)*1e6
        err_s = abs(E_s - E_ref) / E_ref * 100
        methods_comp[f'Simpson n={n}'] = (E_s, err_s, ts)
        print(f"  {'Simpsons n='+str(n):<25} {E_s:<16.4f} {err_s:<14.8f} {ts:.2f}")

    t0 = time.perf_counter(); E_q, _ = integrate.quad(integrand, 0, 20); tq=(time.perf_counter()-t0)*1e6
    methods_comp['quad (adaptive)'] = (E_q, 0.0, tq)
    print(f"  {'Quad (adaptive)':<25} {E_q:<16.4f} {'0 (ref)':<14} {tq:.2f}")

    print("\n[12.2] Root-Finding Comparative Summary")
    def test_function(x):
        return x**3 - 2*x**2 - 5

    methods_rf = {
        'Bisection'      : lambda: opt.bisect(test_function, 2, 4, xtol=1e-10),
        'Brentq'         : lambda: opt.brentq(test_function, 2, 4, xtol=1e-10),
        'Newton-Raphson' : lambda: opt.newton(test_function, x0=3.0, tol=1e-10),
        'Secant'         : lambda: opt.newton(test_function, x0=3.0, x1=3.5, tol=1e-10),
    }
    print(f"\n  {'Method':<20} {'Root':<18} {'Time (μs)':<14} {'Residual'}")
    print(f"  {'-'*60}")
    rf_results = {}
    for name, fn in methods_rf.items():
        try:
            t0 = time.perf_counter()
            root = fn()
            elapsed = (time.perf_counter() - t0) * 1e6
            residual = abs(test_function(root))
            rf_results[name] = (root, elapsed, residual)
            print(f"  {name:<20} {root:<18.10f} {elapsed:<14.4f} {residual:.2e}")
        except Exception as e:
            print(f"  {name:<20} FAILED: {e}")

    print("\n[12.3] ODE Solver Comparison Summary")
    def ode_system(t, y):
        return [-0.5 * y[0] + np.sin(t)]

    solvers = ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF']
    print(f"\n  {'Solver':<12} {'Steps':<10} {'Final y':<14} {'Time (ms)':<14} {'Accuracy'}")
    print(f"  {'-'*60}")
    y_ref = None
    ode_results = {}
    for solver in solvers:
        try:
            t0 = time.perf_counter()
            sol = solve_ivp(ode_system, (0, 10), [1.0], method=solver,
                            rtol=1e-8, atol=1e-10, dense_output=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if y_ref is None:
                y_ref = sol.y[0, -1]
            acc = abs(sol.y[0, -1] - y_ref)
            ode_results[solver] = (len(sol.t), sol.y[0, -1], elapsed_ms, acc)
            print(f"  {solver:<12} {len(sol.t):<10} {sol.y[0,-1]:<14.8f} {elapsed_ms:<14.4f} {acc:.2e}")
        except Exception as e:
            print(f"  {solver:<12} FAILED: {e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Topic 12: Comparative Analysis & Case Study", fontsize=14, fontweight='bold')

    # Integration accuracy vs method
    labels_i = list(methods_comp.keys())
    errors_i = [v[1] for v in methods_comp.values()]
    times_i  = [v[2] for v in methods_comp.values()]
    colors_i = ['tomato'] * 3 + ['steelblue'] * 3 + ['forestgreen']
    bars = axes[0, 0].bar(labels_i, [max(e, 1e-10) for e in errors_i],
                           color=colors_i, edgecolor='black')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title("Integration: Error Comparison (log scale)")
    axes[0, 0].set_ylabel("Relative Error (%)")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.4, axis='y')

    axes[0, 1].bar(labels_i, times_i, color=colors_i, edgecolor='black')
    axes[0, 1].set_title("Integration: Speed Comparison")
    axes[0, 1].set_ylabel("Time (μs)")
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.4, axis='y')

    # Root finding comparison
    if rf_results:
        rf_names = list(rf_results.keys())
        rf_times = [v[1] for v in rf_results.values()]
        rf_resids = [v[2] + 1e-20 for v in rf_results.values()]
        x_rf = np.arange(len(rf_names))
        axes[1, 0].bar(x_rf, rf_resids, color='mediumpurple', edgecolor='black')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title("Root-Finding: Residual |f(root)| (log)")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_xticks(x_rf)
        axes[1, 0].set_xticklabels(rf_names, rotation=20, fontsize=9)
        axes[1, 0].grid(True, alpha=0.4, axis='y')

    # ODE solver steps vs time scatter
    if ode_results:
        solver_names_ode = list(ode_results.keys())
        steps_ode = [v[0] for v in ode_results.values()]
        times_ode = [v[2] for v in ode_results.values()]
        scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(solver_names_ode)))
        for i, (name, nstep, tms) in enumerate(zip(solver_names_ode, steps_ode, times_ode)):
            axes[1, 1].scatter(nstep, tms, s=120, color=scatter_colors[i],
                               label=name, zorder=5, edgecolors='black')
        axes[1, 1].set_title("ODE Solvers: Steps vs Time")
        axes[1, 1].set_xlabel("Number of Steps")
        axes[1, 1].set_ylabel("Solve Time (ms)")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'topic12_comparative.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: topic12_comparative.png]")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(SEPARATOR)
    print("  DRONE DELIVERY ROUTE OPTIMIZATION USING NUMERICAL METHODS")
    print("  Course: 155-4007 Numerical Methods in Engineering")
    print("  Python Version: " + sys.version.split()[0])
    print(SEPARATOR)

    # ── Fetch live weather data ONCE (shared across all topics) ──────────────
    print("\n  Fetching real-time weather data from Open-Meteo API ...")
    print("  Route: Mersin, Turkey  (lat=36.8969, lon=34.7313, length=20 km)\n")
    weather = fetch_wind_profile(
        lat=36.8969, lon=34.7313,
        n_points=12, route_length_km=20.0,
        verbose=True
    )
    print(f"\n  Weather data source : {weather['source'].upper()}")
    print(f"  {weather['description']}\n")

    t_start = time.perf_counter()

    # --- Run all 12 topics ---
    results = {}
    results['weather'] = weather    # store for summary

    # Topic 1: Error Analysis
    results['t1'] = topic1_error_analysis()

    # Topic 2: Root Finding
    results['t2'] = topic2_root_finding()

    # Topic 3: Interpolation  (uses live weather)
    cs_terrain, cs_wind = topic3_interpolation(weather)

    # Topic 4: Differentiation
    results['t4_dh'], results['t4_x'] = topic4_differentiation(cs_terrain)

    # Topic 5: Integration
    results['t5_E'] = topic5_integration(cs_terrain, cs_wind)

    # Topic 6: Linear Systems
    A_mat, b_vec = topic6_linear_systems()

    # Topic 7: LU Decomposition
    topic7_lu_decomposition(A_mat, b_vec)

    # Topic 8: Optimization
    results['v_opt'], results['wp_opt'] = topic8_optimization()

    # Topic 9: ODEs
    results['sol_ode'] = topic9_odes()

    # Topic 10: Performance & Stability
    topic10_performance_stability()

    # Topic 11: Visualization Dashboard  (uses live weather)
    topic11_visualization(results['sol_ode'], results['v_opt'], weather)

    # Topic 12: Comparative Analysis  (uses live weather)
    topic12_comparative_analysis(weather)

    t_total = time.perf_counter() - t_start

    print(f"\n{SEPARATOR}")
    print(f"  ALL 12 TOPICS COMPLETED SUCCESSFULLY")
    print(f"  Total runtime: {t_total:.2f} seconds")
    print(f"  Plots saved to: {os.path.abspath(PLOT_DIR)}")
    print(SEPARATOR)

    # Final Summary
    w = results['weather']
    print("\n  KEY RESULTS SUMMARY")
    print(f"  Weather source           : {w['source'].upper()} ({w['location']})")
    print(f"  Temperature / Humidity   : {w['temperature_c']:.1f} °C  /  {w['humidity_pct']:.0f} %")
    print(f"  Air density (live)       : {w['air_density']:.4f} kg/m³")
    print(f"  Wind (mean along route)  : {w['wind_speeds'].mean():.2f} m/s")
    print(f"  Machine epsilon          : {results['t1']['machine_eps']:.4e}")
    print(f"  Hover altitude (root)    : {results['t2']['bisection']['root']:.2f} m")
    print(f"  Total route energy       : {results['t5_E']/3600:.4f} Wh")
    print(f"  Optimal cruise speed     : {results['v_opt']:.4f} m/s ({results['v_opt']*3.6:.2f} km/h)")

    return results


if __name__ == "__main__":
    main()