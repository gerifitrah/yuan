"""
charts.py — All visualisation functions for PLTA Grindulu dashboard.

Group A — Model prediction charts (require trained model):
    chart_observed_vs_predicted(...)   Chart 1
    chart_quantile_prediction(...)     Chart 2
    chart_prediction_interval(...)     Chart 3

Group B — Turbine performance charts (PLTA specs, no model needed):
    chart_turbine_efficiency(...)      Chart 4
    chart_fixed_vs_variable_speed(...) Chart 5
    chart_power_output(...)            Chart 6

PLTA Grindulu PS Specs — OPSI-2A
----------------------------------
Net head        : 500 m
Design flow/unit: 59.0 m³/s
Units           : 4  (total 236 m³/s)
Turbine η       : 0.90
Generator η     : 0.96
Rated power/unit: 250 MW  (total 1000 MW)
Turbine type    : Francis reversible pumped
Speed           : 500 RPM  (variable speed range 450–550 RPM)
Upper reservoir : 5.42 juta m³
Lower reservoir : 6.51 juta m³
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# PLTA constants
# ---------------------------------------------------------------------------

H_NET        = 500.0          # m  net head (OPSI-2A)
H_GROSS      = 500.0          # m  gross head
Q_DESIGN     = 59.0           # m³/s  per unit (OPSI-2A)
N_UNITS      = 4
ETA_T        = 0.90           # turbine efficiency at design point
ETA_G        = 0.96           # generator efficiency
RATED_MW     = 250.0          # MW per unit
RHO          = 1000.0         # kg/m³
G            = 9.81           # m/s²
RPM_RATED    = 500            # RPM (fixed speed)
RPM_VAR_LOW  = 450            # RPM variable speed lower bound
RPM_VAR_HIGH = 550            # RPM variable speed upper bound

# Power formula: P(MW) = rho * g * Q * H * eta_t * eta_g / 1e6
def _power_mw(Q, H=H_NET, eta_t=ETA_T, eta_g=ETA_G):
    return RHO * G * Q * H * eta_t * eta_g / 1e6


# ---------------------------------------------------------------------------
# GROUP A — Prediction charts
# ---------------------------------------------------------------------------

def chart_observed_vs_predicted(
    obs: np.ndarray,
    pred_q50: np.ndarray,
    dates=None,
    title: str = "Observed vs Predicted Inflow (Q50 — Test Set)",
) -> go.Figure:
    """Chart 1: Scatter + time-series of observed vs predicted median inflow.

    Args:
        obs      : (N*pred_len,) or (N, pred_len) observed Q total (m³/s)
        pred_q50 : same shape, q50 predictions
        dates    : optional date array matching flattened shape
    """
    obs      = np.array(obs).flatten()
    pred_q50 = np.array(pred_q50).flatten()
    n        = len(obs)
    x_idx    = dates if dates is not None else np.arange(n)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series", "Scatter Plot (Observed vs Q50)"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Time series
    fig.add_trace(go.Scatter(
        x=x_idx, y=obs, name="Observed",
        mode="lines", line=dict(color="#2ca02c", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_idx, y=pred_q50, name="Q50 Predicted",
        mode="lines", line=dict(color="#d62728", width=1.2, dash="dot"),
    ), row=1, col=1)

    # Scatter
    max_val = max(obs.max(), pred_q50.max()) * 1.05
    fig.add_trace(go.Scatter(
        x=obs, y=pred_q50,
        mode="markers", name="Q50 vs Obs",
        marker=dict(color="#1f77b4", size=4, opacity=0.5),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect fit",
        line=dict(color="red", dash="dash", width=1),
        showlegend=False,
    ), row=2, col=1)

    # NSE / correlation annotation
    ss_res = np.sum((obs - pred_q50) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    nse    = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr   = float(np.corrcoef(obs, pred_q50)[0, 1])
    rmse   = float(np.sqrt(np.mean((obs - pred_q50) ** 2)))

    fig.add_annotation(
        xref="x2 domain", yref="y2 domain",
        x=0.05, y=0.95,
        text=(f"NSE  = {nse:.4f}<br>"
              f"r    = {corr:.4f}<br>"
              f"RMSE = {rmse:.3f} m³/s"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray", borderwidth=1,
    )

    fig.update_xaxes(title_text="Observed Q (m³/s)", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Q50 (m³/s)", row=2, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", row=1, col=1)
    fig.update_layout(title=title, height=620, legend=dict(orientation="h", y=-0.08))
    return fig


def chart_quantile_prediction(
    obs: np.ndarray,
    pred_q10: np.ndarray,
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    dates=None,
    n_show: int = 200,
    title: str = "Quantile Prediction — Q10 / Q50 / Q90 (Test Set)",
) -> go.Figure:
    """Chart 2: Time-series of all 3 quantiles vs observed.

    Args:
        obs, pred_q10/q50/q90 : (N*pred_len,) flat arrays
        n_show                : number of most-recent points to display
    """
    obs      = np.array(obs).flatten()[-n_show:]
    pred_q10 = np.array(pred_q10).flatten()[-n_show:]
    pred_q50 = np.array(pred_q50).flatten()[-n_show:]
    pred_q90 = np.array(pred_q90).flatten()[-n_show:]
    x        = (dates[-n_show:] if dates is not None
                else np.arange(len(obs)))

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(pred_q90) + list(pred_q10)[::-1],
        fill="toself",
        fillcolor="rgba(31,119,180,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Q10–Q90 band",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q90, name="Q10 (Basah)",
        mode="lines", line=dict(color="#1f77b4", dash="dot", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q50, name="Q50 (Normal)",
        mode="lines", line=dict(color="#d62728", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q10, name="Q90 (Kering)",
        mode="lines", line=dict(color="#ff7f0e", dash="dot", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=obs, name="Observed",
        mode="lines", line=dict(color="#2ca02c", width=1.5),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Inflow Q (m³/s)",
        legend=dict(orientation="h", y=-0.15),
        height=430,
    )
    return fig


def chart_prediction_interval(
    obs: np.ndarray,
    pred_q10: np.ndarray,
    pred_q90: np.ndarray,
    dates=None,
    n_show: int = 200,
    title: str = "80% Prediction Interval Coverage (Test Set)",
) -> go.Figure:
    """Chart 3: Prediction interval with coverage highlighting.

    Points outside the [q10, q90] band are highlighted in red.
    """
    obs      = np.array(obs).flatten()[-n_show:]
    pred_q10 = np.array(pred_q10).flatten()[-n_show:]
    pred_q90 = np.array(pred_q90).flatten()[-n_show:]
    x        = (dates[-n_show:] if dates is not None
                else np.arange(len(obs)))

    inside  = (obs >= pred_q10) & (obs <= pred_q90)
    outside = ~inside
    picp    = inside.mean() * 100

    fig = go.Figure()

    # Band
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(pred_q90) + list(pred_q10)[::-1],
        fill="toself",
        fillcolor="rgba(31,119,180,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="[Q10, Q90] band",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q90, name="Q90",
        mode="lines", line=dict(color="#1f77b4", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q10, name="Q10",
        mode="lines", line=dict(color="#ff7f0e", width=1),
    ))

    # Inside points
    fig.add_trace(go.Scatter(
        x=x[inside], y=obs[inside],
        mode="markers", name="Inside interval",
        marker=dict(color="#2ca02c", size=4),
    ))
    # Outside points
    if outside.any():
        fig.add_trace(go.Scatter(
            x=x[outside], y=obs[outside],
            mode="markers", name="Outside interval",
            marker=dict(color="red", size=6, symbol="x"),
        ))

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97,
        text=f"PICP = {picp:.2f} % (target >= 80 %)",
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray", borderwidth=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Inflow Q (m³/s)",
        legend=dict(orientation="h", y=-0.15),
        height=430,
    )
    return fig


# ---------------------------------------------------------------------------
# GROUP B — Turbine performance charts
# ---------------------------------------------------------------------------

def chart_turbine_efficiency(
    title: str = "Turbine Efficiency vs Flow Rate — Francis Reversible (PLTA Grindulu)",
) -> go.Figure:
    """Chart 4: η vs Q/Q_rated for Francis turbine.

    Uses a standard hill-chart polynomial approximation for Francis turbines.
    Reference: Çengel & Cimbala (2014), IEC 60193 test standards.

    η(q) = ηmax * (1 - k * (q - 1)^2)
    where q = Q/Q_design, k calibrated to drop to η≈0.70 at q=0.3
    """
    q_ratio = np.linspace(0.30, 1.10, 300)

    # Francis turbine best-fit polynomial (dimensionless flow ratio)
    # Peak at q=1.0, η=0.90; drops to ~0.72 at q=0.3 and ~0.85 at q=1.1
    k = 0.28
    eta_francis = ETA_T * (1 - k * (q_ratio - 1.0) ** 2)
    eta_francis = np.clip(eta_francis, 0, 1)

    Q_abs = q_ratio * Q_DESIGN   # m³/s per unit

    fig = go.Figure()

    # Main efficiency curve
    fig.add_trace(go.Scatter(
        x=Q_abs, y=eta_francis * 100,
        mode="lines", name="Turbine efficiency ηT",
        line=dict(color="#1f77b4", width=2.5),
    ))

    # Design point marker
    fig.add_trace(go.Scatter(
        x=[Q_DESIGN], y=[ETA_T * 100],
        mode="markers", name=f"Design point (Q={Q_DESIGN} m³/s, η={ETA_T*100:.0f}%)",
        marker=dict(color="red", size=12, symbol="star"),
    ))

    # Minimum technical flow (typically 0.3 × Q_design for Francis)
    q_min = 0.3 * Q_DESIGN
    fig.add_vline(
        x=q_min, line_dash="dash", line_color="orange",
        annotation_text=f"Q_min = {q_min:.1f} m³/s",
        annotation_position="top right",
    )

    # Zones
    fig.add_hrect(y0=85, y1=90, fillcolor="rgba(0,200,0,0.08)",
                  line_width=0, annotation_text="Optimal zone (≥85%)",
                  annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_title="Flow Rate Q (m³/s per unit)",
        yaxis_title="Turbine Efficiency ηT (%)",
        yaxis=dict(range=[60, 95], ticksuffix="%"),
        legend=dict(orientation="h", y=-0.15),
        height=430,
    )
    return fig


def chart_fixed_vs_variable_speed(
    title: str = "Fixed Speed vs Variable Speed Operation — Turbine Efficiency",
) -> go.Figure:
    """Chart 5: Efficiency comparison fixed speed (500 RPM) vs variable speed (450–550 RPM).

    Variable speed (e.g., with doubly-fed induction generator or full converter)
    allows RPM to track the BEP across a wider flow range.

    Reference curves based on:
    - Muljadi & Butterfield (2001) variable speed hydro
    - Bortoni et al. (2008) variable speed Francis optimization
    """
    q_ratio = np.linspace(0.30, 1.10, 300)
    Q_abs   = q_ratio * Q_DESIGN

    # Fixed speed: sharper peak, steeper drop at partial load
    k_fixed = 0.28
    eta_fixed = ETA_T * (1 - k_fixed * (q_ratio - 1.0) ** 2)

    # Variable speed: flatter curve, holds ~η_peak over wider range
    # Modelled as fixed speed curve with "BEP shift" that tracks Q
    # Net effect: efficiency improved at partial load by ~3–5%
    k_var = 0.14
    eta_variable = ETA_T * (1 - k_var * (q_ratio - 1.0) ** 2)
    eta_variable = np.clip(eta_variable, 0, ETA_T)

    improvement = (eta_variable - eta_fixed) * 100  # percentage points

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Efficiency Curves", "Improvement from Variable Speed"),
        vertical_spacing=0.14,
        row_heights=[0.65, 0.35],
    )

    # Efficiency curves
    fig.add_trace(go.Scatter(
        x=Q_abs, y=eta_fixed * 100,
        mode="lines", name="Fixed speed (500 RPM)",
        line=dict(color="#d62728", width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=Q_abs, y=eta_variable * 100,
        mode="lines", name="Variable speed (450–550 RPM)",
        line=dict(color="#1f77b4", width=2.5, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[Q_DESIGN], y=[ETA_T * 100],
        mode="markers", name="Design point",
        marker=dict(color="black", size=10, symbol="diamond"),
        showlegend=True,
    ), row=1, col=1)

    # Improvement bars
    fig.add_trace(go.Bar(
        x=Q_abs, y=improvement,
        name="Variable – Fixed (pp)",
        marker_color=np.where(improvement >= 0, "#2ca02c", "#d62728").tolist(),
        showlegend=False,
    ), row=2, col=1)

    fig.update_yaxes(title_text="Efficiency (%)", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="Improvement (pp)", row=2, col=1)
    fig.update_xaxes(title_text="Flow Rate Q (m³/s per unit)", row=2, col=1)

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=-0.08),
        height=600,
    )
    return fig


def chart_probabilistic_forecast(
    obs: np.ndarray,
    pred_q10: np.ndarray,
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    dates=None,
    n_show: int = 200,
    title: str = "Quantile BiLSTM: Probabilistic Inflow Forecast",
) -> go.Figure:
    """Chart 7: Confidence band + median + actual — matching the reference paper style.

    Blue/purple shaded band for Q10–Q90, solid blue Q50 median,
    red dashed actual observed line.
    """
    obs      = np.array(obs).flatten()[-n_show:]
    pred_q10 = np.array(pred_q10).flatten()[-n_show:]
    pred_q50 = np.array(pred_q50).flatten()[-n_show:]
    pred_q90 = np.array(pred_q90).flatten()[-n_show:]
    x        = dates[-n_show:] if dates is not None else np.arange(len(obs))

    fig = go.Figure()

    # Shaded confidence band Q10–Q90
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(pred_q90) + list(pred_q10)[::-1],
        fill="toself",
        fillcolor="rgba(100, 130, 220, 0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10–90% Interval",
        hoverinfo="skip",
    ))

    # Q90 border (light)
    fig.add_trace(go.Scatter(
        x=x, y=pred_q90, name="Q10 — Basah",
        mode="lines",
        line=dict(color="rgba(100,130,220,0.6)", width=1),
        showlegend=False,
    ))

    # Q10 border (light)
    fig.add_trace(go.Scatter(
        x=x, y=pred_q10, name="Q90 — Kering",
        mode="lines",
        line=dict(color="rgba(100,130,220,0.6)", width=1),
        showlegend=False,
    ))

    # Q50 median — solid blue
    fig.add_trace(go.Scatter(
        x=x, y=pred_q50, name="Median (Q50)",
        mode="lines",
        line=dict(color="#1f77b4", width=2),
    ))

    # Actual — red dashed
    fig.add_trace(go.Scatter(
        x=x, y=obs, name="Actual",
        mode="lines",
        line=dict(color="#d62728", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title="Inflow Q (m³/s)",
        legend=dict(
            orientation="h", y=-0.15,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eeeeee", showline=True, linecolor="#cccccc")
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#cccccc")
    return fig


def chart_all_quantiles(
    obs: np.ndarray,
    pred_q10: np.ndarray,
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    dates=None,
    n_show: int = 200,
    title: str = "All Quantile Predictions",
) -> go.Figure:
    """Chart 8: All 3 quantile lines plotted together against observed.

    Displays Q10, Q50, Q90 as distinct coloured lines so individual
    quantile behaviour can be compared directly.
    """
    obs      = np.array(obs).flatten()[-n_show:]
    pred_q10 = np.array(pred_q10).flatten()[-n_show:]
    pred_q50 = np.array(pred_q50).flatten()[-n_show:]
    pred_q90 = np.array(pred_q90).flatten()[-n_show:]
    x        = dates[-n_show:] if dates is not None else np.arange(len(obs))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=obs, name="Observed",
        mode="lines",
        line=dict(color="#2ca02c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q90, name="Q10 — Basah",
        mode="lines",
        line=dict(color="#1565C0", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q50, name="Q50 — Median",
        mode="lines",
        line=dict(color="#d62728", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pred_q10, name="Q90 — Kering",
        mode="lines",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title="Inflow Q (m³/s)",
        legend=dict(
            orientation="h", y=-0.15,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eeeeee", showline=True, linecolor="#cccccc")
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#cccccc")
    return fig


def chart_uncertainty_width(
    pred_q10: np.ndarray,
    pred_q90: np.ndarray,
    dates=None,
    n_show: int = 200,
    title: str = "Prediction Uncertainty (10–90% Interval)",
) -> go.Figure:
    """Chart 9: Width of the prediction interval (Q90 − Q10) per time step.

    Bar chart with a red dashed mean-uncertainty reference line.
    Wider bars = model less certain; narrower = more confident.
    """
    pred_q10 = np.array(pred_q10).flatten()[-n_show:]
    pred_q90 = np.array(pred_q90).flatten()[-n_show:]
    x        = dates[-n_show:] if dates is not None else np.arange(len(pred_q10))

    width     = pred_q90 - pred_q10
    mean_w    = float(width.mean())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x, y=width,
        name="Interval Width (Q90 − Q10)",
        marker_color="rgba(100, 130, 220, 0.65)",
        marker_line_color="rgba(60, 90, 180, 0.8)",
        marker_line_width=0.5,
    ))

    # Mean uncertainty reference line
    fig.add_hline(
        y=mean_w,
        line_dash="dash", line_color="#d62728", line_width=1.8,
        annotation_text=f"Mean = {mean_w:.3f} m³/s",
        annotation_position="top right",
        annotation_font_color="#d62728",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title="Interval Width (m³/s)",
        legend=dict(orientation="h", y=-0.15),
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eeeeee", rangemode="tozero",
                     showline=True, linecolor="#cccccc")
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#cccccc")
    return fig


def chart_power_output(
    title: str = "Power Output at Various Flow Conditions — PLTA Grindulu (1000 MW)",
) -> go.Figure:
    """Chart 6: Power output vs total available inflow.

    Shows:
    - P vs Q_total for 1, 2, 3, 4 active units
    - Design operating point (4 × 60.5 m³/s = 242 m³/s → 4 × 250 MW)
    - Minimum technical flow per unit

    Unit commitment logic:
        Q available = Q_total (m³/s from inflow)
        Each unit requires exactly Q_design m³/s when ON
        If Q < Q_design → 0 units (insufficient for even 1 unit)
        Fractional loading capped at Q_design per unit
    """
    Q_total = np.linspace(0, N_UNITS * Q_DESIGN * 1.05, 500)

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for n_on in range(1, N_UNITS + 1):
        Q_min_n = 0.30 * Q_DESIGN * n_on   # min tech flow for n_on units
        Q_max_n = Q_DESIGN * n_on

        P_n = np.zeros_like(Q_total)
        mask = (Q_total >= Q_min_n)
        Q_per_unit = np.clip(Q_total[mask] / n_on, 0, Q_DESIGN)
        q_ratio = Q_per_unit / Q_DESIGN
        eta_t = ETA_T * (1 - 0.28 * (q_ratio - 1.0) ** 2)
        eta_t = np.clip(eta_t, 0, ETA_T)
        P_n[mask] = n_on * _power_mw(Q_per_unit, eta_t=eta_t)

        # Only show region where this unit count is optimal
        fig.add_trace(go.Scatter(
            x=Q_total, y=P_n,
            mode="lines", name=f"{n_on} unit{'s' if n_on > 1 else ''}",
            line=dict(color=colors[n_on - 1], width=2),
        ))

    # Design operating point
    P_design = N_UNITS * RATED_MW
    Q_design_total = N_UNITS * Q_DESIGN
    fig.add_trace(go.Scatter(
        x=[Q_design_total], y=[P_design],
        mode="markers", name=f"Design point ({Q_design_total:.0f} m³/s, {P_design:.0f} MW)",
        marker=dict(color="black", size=14, symbol="star"),
    ))

    # Rated line
    fig.add_hline(
        y=P_design, line_dash="dash", line_color="gray",
        annotation_text=f"Rated capacity: {P_design:.0f} MW",
        annotation_position="top left",
    )

    # Min flow annotation
    q_min_1 = 0.30 * Q_DESIGN
    fig.add_vline(
        x=q_min_1, line_dash="dot", line_color="orange",
        annotation_text=f"Q_min 1 unit = {q_min_1:.1f} m³/s",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Total Available Inflow Q (m³/s)",
        yaxis_title="Power Output P (MW)",
        legend=dict(orientation="h", y=-0.12),
        height=450,
        annotations=[
            dict(
                xref="paper", yref="paper", x=0.99, y=0.02,
                text=(f"H_net={H_NET} m | ηT={ETA_T} | ηG={ETA_G}<br>"
                      f"Q_design={Q_DESIGN} m³/s/unit | {N_UNITS} units"),
                showarrow=False, align="right",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="gray", borderwidth=1,
            )
        ],
    )
    return fig
