"""
app.py -- Panel user interface for the directional-noise interferometry simulator.

Run with:   panel serve app.py --show
All numerics live in simulation.py; this file only wires widgets to plots.

Two kinds of controls:
  * Medium / geometry (station spacing, heterogeneity): need new FD runs (~15 s),
    triggered by the "Run solver" button and cached.
  * Noise field and frequency band: only re-weight stored FD results, so the plots
    update instantly while dragging the sliders.
"""
from functools import lru_cache
import numpy as np
import panel as pn
from matplotlib.figure import Figure
import simulation as S

pn.extension(sizing_mode="stretch_width")


@lru_cache(maxsize=8)
def experiment(r_km, het_pct):
    """Two FD impulse runs for a station pair (cached per medium/geometry)."""
    m = S.Model()
    if het_pct > 0:
        m = S.Model(c=S.random_medium(m, rel_std=het_pct / 100, corr_len=0.1, seed=1))
    return S.PairExperiment(m, r=r_km)


# ---------------------------------------------------------------- widgets
w_r = pn.widgets.Select(name="Station spacing r (km)", options=[1, 2, 3, 4, 5], value=3)
w_het = pn.widgets.Select(name="Velocity heterogeneity (std, %)", options=[0, 5, 10], value=0)
w_run = pn.widgets.Button(name="Run solver", button_type="primary")
w_th = pn.widgets.FloatSlider(name="Dominant noise direction θ₀ (deg from A→B axis)",
                              start=0, end=180, step=5, value=90)
w_k = pn.widgets.FloatSlider(name="Concentration κ (0 = isotropic)", start=0, end=8, step=0.25, value=2)
w_eps = pn.widgets.FloatSlider(name="Isotropic background ε", start=0, end=1, step=0.05, value=0)
w_fc = pn.widgets.FloatSlider(name="Band centre frequency f_c (Hz)", start=0.8, end=2.8,
                              step=0.1, value=S.F0)
status = pn.pane.Markdown("")
state = {"exp": experiment(w_r.value, w_het.value)}


def rerun(_):
    status.object = "*Running FD solver (2 impulse runs)...*"
    state["exp"] = experiment(w_r.value, w_het.value)
    status.object = f"Solver done: r = {state['exp'].r:.0f} km, heterogeneity {w_het.value}%."
    w_th.param.trigger("value")   # refresh plots


w_run.on_click(rerun)


# ---------------------------------------------------------------- plots
def source_map(exp, th, k, eps):
    fig = Figure(figsize=(4.6, 4.2)); ax = fig.subplots()
    B = S.source_density(exp.theta, th, k, eps)
    sc = ax.scatter(*exp.src_xy.T, c=B * 2 * np.pi, s=12, cmap="viridis", vmin=0)
    ax.plot(*np.c_[exp.xA, exp.xB], "r^", ms=9)
    ax.annotate("A", exp.xA, xytext=(-14, 6), textcoords="offset points", color="r")
    ax.annotate("B", exp.xB, xytext=(6, 6), textcoords="offset points", color="r")
    R = np.linalg.norm(exp.src_xy[0])
    ax.annotate("", xy=(0, 0), xytext=(0.75 * R * np.cos(th), 0.75 * R * np.sin(th)),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))   # noise propagation
    ax.set(aspect="equal", xlabel="x (km)", ylabel="y (km)", title="Noise sources (arrow: dominant flow)")
    fig.colorbar(sc, ax=ax, label="relative intensity B·2π", shrink=0.8)
    fig.tight_layout(); return fig


def correlation_plot(exp, C, Cref, fc):
    fig = Figure(figsize=(7.5, 3.6)); ax = fig.subplots()
    n = np.abs(Cref).max()
    ax.plot(exp.lags, Cref / n, color="0.6", lw=2, label="isotropic noise (reference)")
    ax.plot(exp.lags, C / n, "k", lw=1.2, label="directional noise")
    t = exp.r / exp.model.c0
    for s in (-1, 1):
        ax.axvline(s * t, color="g", ls=":")
        ax.axvspan(s * t - 1.5 / fc, s * t + 1.5 / fc, color="g", alpha=0.06)
    ax.set(xlim=(-2.2 * t - 3 / fc, 2.2 * t + 3 / fc), xlabel="lag (s)   (+: A→B, −: B→A)",
           ylabel="C_AB (normalised to isotropic peak)", title="Expected noise cross-correlation")
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout(); return fig


def anisotropy_plot(exp, th, k, eps, fc):
    az, dv, fit = exp.azimuth_sweep(th, k, eps, fc)
    fig = Figure(figsize=(4.2, 4.2)); ax = fig.add_subplot(projection="polar")
    # symmetric branch is 180-degree periodic: mirror to a full circle
    a2 = np.r_[az, az + np.pi, az[:1]]
    v2 = exp.model.c0 * (1 + np.r_[dv, dv, dv[:1]] / 100)          # apparent velocity [km/s]
    ax.plot(a2, v2, "b", lw=2, label="apparent velocity")
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, exp.model.c0), "k:", lw=1, label="true velocity")
    pad = max(0.3 * np.ptp(v2), 0.01)
    lo, hi = min(v2.min(), exp.model.c0) - pad, max(v2.max(), exp.model.c0) + pad
    ax.set_rlim(lo, hi); ax.set_rticks(np.round(np.linspace(lo + pad, hi - pad, 3), 3))
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"Apparent velocity (km/s) vs pair azimuth\nfake 2θ anisotropy A₂ = {fit['A2']:.2f}%, "
                 f"fast axis {np.degrees(fit['fast_az']):.0f}°", fontsize=9)
    ax.legend(loc="lower left", fontsize=7, bbox_to_anchor=(-0.15, -0.15))
    fig.tight_layout(); return fig


def view(th_deg, k, eps, fc):
    exp = state["exp"]
    th = np.radians(th_deg)
    C, Cref = exp.correlate(exp.weights(th, k, eps), fc), exp.correlate(exp.weights(), fc)
    m = exp.measure(C, Cref, fc)
    p = exp.weaver_prediction(th, k, eps, fc)
    lam = exp.model.c0 / fc
    table = pn.pane.Markdown(f"""
| quantity | measured (FD) | stationary-phase theory |
|---|---|---|
| travel-time bias, causal (A→B) | {1e3 * m['dt_causal']:+.1f} ms | {1e3 * p['dt_causal']:+.1f} ms |
| travel-time bias, acausal (B→A) | {1e3 * m['dt_acausal']:+.1f} ms | {1e3 * p['dt_acausal']:+.1f} ms |
| travel-time bias, symmetric | {1e3 * m['dt_sym']:+.1f} ms | {1e3 * p['dt_sym']:+.1f} ms |
| **apparent velocity error δc/c** | **{m['dv_sym']:+.2f} %** | {p['dv_sym']:+.2f} % |
| causal/acausal energy | {m['asym_db']:+.1f} dB | |

True travel time r/c = {exp.r / exp.model.c0:.2f} s; wavelength {lam:.2f} km; spacing r/λ = {exp.r / lam:.2f}.
""")
    return pn.Column(
        pn.Row(pn.pane.Matplotlib(source_map(exp, th, k, eps), tight=True, format="svg"),
               pn.pane.Matplotlib(anisotropy_plot(exp, th, k, eps, fc), tight=True, format="svg")),
        pn.pane.Matplotlib(correlation_plot(exp, C, Cref, fc), tight=True, format="svg"),
        table)


controls = pn.Column(
    "### Medium & stations", w_r, w_het, w_run, status,
    "### Noise field (instant)", w_th, w_k, w_eps,
    "### Frequency band", w_fc,
    pn.pane.Markdown(f"Basin scale: scalar 2D wave equation, c = {1000 * S.C0:.0f} m/s, FD leapfrog with "
                     f"4th-order Laplacian (dx = {1000 * S.DX:.0f} m). Noise sources on a "
                     f"{S.RING_R:.0f} km ring. Correlations via reciprocity."),
)
pn.template.FastListTemplate(
    title="Directional ambient noise: travel-time bias in seismic interferometry",
    sidebar=[controls], main=[pn.bind(view, w_th, w_k, w_eps, w_fc)],
).servable()
