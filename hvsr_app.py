"""
hvsr_app.py -- Panel UI: HVSR-style site response of a dipping soft layer from ambient noise.

Run with:   panel serve hvsr_app.py --show
All numerics live in hvsr.py; this file only wires widgets to plots.

  * Model and noise controls need a solver run ("Run solver"): one run with the soft layer
    and one bare-rock reference run with the same noise (about 15 s on the default grid).
    The wavefield is drawn live while the solver runs, then replayed as a looping GIF.
    Results are cached in memory and on disk (.cache/).
  * The station slider (which H/V curve and amplification prediction to show) and the
    smoothing control only re-process the stored records (instant).
"""
import asyncio, base64, io, pickle, time
from pathlib import Path
import numpy as np
import panel as pn
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
import hvsr as H

pn.extension(sizing_mode="stretch_width")

CACHE_DIR = Path(__file__).parent / ".cache"
GRIDS = {"fast (dx = 5 m)": 5.0, "fine (dx = 2.5 m, ~8x slower)": 2.5}
FRAMES = np.round(np.arange(1.0, 7.001, 0.05), 3)       # animation frame times [s]
REC_X = np.arange(-150.0, 150.1, 10.0)                  # surface receivers


def run_model(model, dx, T, n_surface, progress=None, on_snapshot=None):
    """Site + reference runs for one model, cached in memory (all sessions) and on disk."""
    key = (f"hvsr_v2_dx{dx}_T{T}_s{n_surface}_v{model.vs1:.0f}-{model.vs2:.0f}_r{model.rho1:.0f}-{model.rho2:.0f}"
           f"_h{model.h0:.1f}_d{model.dip:.1f}_q{model.q1:.0f}")
    mem = pn.state.cache.setdefault("hvsr", {})
    if key in mem:
        return mem[key]
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        out = pickle.loads(path.read_bytes())
    else:
        grid = H.default_grid(dx)
        src = H.noise_sources(n_surface=n_surface)
        p1 = (lambda q: progress(0.5 * q)) if progress else None
        p2 = (lambda q: progress(0.5 + 0.5 * q)) if progress else None
        site = H.simulate_noise(model, grid, T, REC_X, src=src, progress=p1, snapshots=FRAMES,
                                on_snapshot=on_snapshot)
        ref = H.simulate_noise(model, grid, T, REC_X, src=src, homogeneous=True, progress=p2)
        out = dict(model=model, src=src, site=site, ref=ref, grid=grid)
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(out))
    mem[key] = out
    return out


# ---------------------------------------------------------------- widgets
w_grid = pn.widgets.Select(name="Grid", options=list(GRIDS), value=list(GRIDS)[0])
w_vs1 = pn.widgets.IntSlider(name="Soft layer Vs₁ (m/s)", start=150, end=500, step=10, value=250)
w_vs2 = pn.widgets.IntSlider(name="Rock Vs₂ (m/s)", start=400, end=2000, step=50, value=1000)
w_rho = pn.widgets.FloatSlider(name="Density ratio ρ₂/ρ₁ (ρ₁ = 1800 kg/m³)", start=1.0, end=1.5, step=0.05, value=1.3)
w_h0 = pn.widgets.IntSlider(name="Layer thickness at x = 0 (m)", start=10, end=60, step=1, value=30)
w_dip = pn.widgets.FloatSlider(name="Interface dip (deg, + = thickens to the right)", start=-15, end=15, step=1, value=5)
w_q = pn.widgets.IntSlider(name="Soft-layer Q (0 = no damping)", start=0, end=100, step=5, value=20)
w_surf = pn.widgets.IntSlider(name="Extra surface-wave sources (beyond the line ends)", start=0, end=12, step=2, value=0)
w_T = pn.widgets.Select(name="Noise record length", options={"30 s": 30.0, "60 s": 60.0, "120 s": 120.0}, value=60.0)
w_run = pn.widgets.Button(name="Run solver", button_type="primary")
bar = pn.indicators.Progress(name="solver", value=0, max=100, visible=False)
status = pn.pane.Markdown("")
contrast_note = pn.pane.Markdown("")

w_ko = pn.widgets.IntSlider(name="Konno-Ohmachi smoothing b", start=10, end=80, step=5, value=25)
refresh = pn.widgets.IntInput(value=0, visible=False)
state = {"run": None}


def current_model():
    return H.Model(vs1=w_vs1.value, vs2=w_vs2.value, rho1=1800.0, rho2=1800.0 * w_rho.value,
                   h0=float(w_h0.value), dip=float(w_dip.value), q1=float(w_q.value))


def show_contrast(*_):
    m = current_model()
    contrast_note.object = (f"Impedance contrast ρ₂Vs₂/ρ₁Vs₁ = **{m.contrast:.1f}**; "
                            f"1D f₀ at x = 0: Vs₁/4H = **{m.f0_1d(0):.2f} Hz**")


for w in (w_vs1, w_vs2, w_rho, w_h0):
    w.param.watch(show_contrast, "value")
show_contrast()


# ---------------------------------------------------------------- wavefield animation
def wave_figure(frame, t, run_or_model, xs, zs, lim, src=None, live=False):
    m = run_or_model
    fig = Figure(figsize=(12, 3.6)); ax = fig.subplots()
    ax.imshow(frame, extent=[xs[0], xs[-1], zs[-1], zs[0]], cmap="seismic", vmin=-lim, vmax=lim,
              aspect="auto", interpolation="bilinear")
    xx = np.linspace(xs[0], xs[-1], 300)
    ax.plot(xx, m.thickness(xx), "k", lw=1.2)
    ax.plot(REC_X, np.full_like(REC_X, 2.0), "gv", ms=5, clip_on=False)
    if src is not None:
        ax.plot(src[0], src[1], "k*", ms=6)
    ax.text(xs[0] + 8, 6, f"soft layer Vs₁ = {m.vs1:.0f} m/s", fontsize=8, va="top")
    ax.text(xs[0] + 8, zs[-1] - 8, f"rock Vs₂ = {m.vs2:.0f} m/s", fontsize=8)
    what = "solver running" if live else "replay"
    ax.set(xlim=(xs[0], xs[-1]), ylim=(zs[-1], 0), xlabel="x (m)", ylabel="depth (m)",
           title=f"Ambient noise from {'' if src is None else len(src[0])} random sources (stars) [{what}], t = {t:.2f} s")
    fig.tight_layout(); return fig


def png_html(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="width:100%">'


def make_gif(run):
    """All frames -> one looping animated GIF (plays in the browser with no server traffic)."""
    s = run["site"]
    lim = float(np.percentile(np.abs(np.stack(list(s["snaps"].values()))), 99.5))
    fig = wave_figure(s["snaps"][FRAMES[0]], FRAMES[0], run["model"], s["x_snap"], s["z_snap"], lim, run["src"])
    ax = fig.axes[0]; im = ax.images[0]
    canvas = FigureCanvasAgg(fig)
    frames = []
    for t in FRAMES:
        im.set_data(s["snaps"][t])
        ax.set_title(ax.get_title().split(", t =")[0] + f", t = {t:.2f} s")
        canvas.draw()
        rgb = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
        frames.append(rgb.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return f'<img src="data:image/gif;base64,{base64.b64encode(buf.getvalue()).decode()}" style="width:100%">'


wave_pane = pn.pane.HTML("", height=300, sizing_mode="stretch_width")
movie_note = pn.pane.Markdown("")


async def rerun(_=None):
    model, dx, T, ns = current_model(), GRIDS[w_grid.value], w_T.value, w_surf.value
    w_run.disabled, bar.visible, bar.value = True, True, 0
    status.object = "*Running the SH solver: soft-layer model, then the bare-rock reference...*"
    t0 = time.time()
    src = H.noise_sources(n_surface=ns)
    last = [0.0]

    def progress(q):
        bar.value = int(100 * q)

    def live(t, frame):
        if time.time() - last[0] < 0.4:
            return
        last[0] = time.time()
        g = H.default_grid(dx)
        keep = np.abs(g.x) <= g.half_width + 1e-9
        sd = max(1, int(round(4.0 / dx)))
        xs, zs = g.x[keep][::sd], g.z[(g.z >= 0) & (g.z <= g.depth + 1e-9)][::sd]
        lim = max(np.percentile(np.abs(frame), 99.5), 1e-30)
        wave_pane.object = png_html(wave_figure(frame, t, model, xs, zs, lim, src, live=True))

    try:
        state["run"] = await asyncio.to_thread(run_model, model, dx, T, ns, progress, live)
        status.object = f"Solver ready ({time.time() - t0:.1f} s): {T:.0f} s of noise, dx = {dx} m."
    except Exception as err:
        status.object = f"**Solver failed:** {err!r}"
        raise
    finally:
        w_run.disabled, bar.visible = False, False
    refresh.value += 1
    movie_note.object = "*Rendering the animation...*"
    wave_pane.object = await asyncio.to_thread(make_gif, state["run"])
    movie_note.object = ("Horizontal (SH) ground motion from 60 random noise sources in the rock; "
                         "the soft layer traps and amplifies it. Loops in the browser.")


w_run.on_click(rerun)
pn.state.onload(rerun)


# ---------------------------------------------------------------- analysis plots
def ratios(b):
    """Spectral ratios and f0 picks for the current run, cached per smoothing setting."""
    run = state["run"]
    key = (id(run), b)
    if state.get("ratio_key") != key:
        f, R = H.spectral_ratio(run["site"], run["ref"], b=b)
        state["ratio"], state["ratio_key"] = (f, R) + H.pick_f0(f, R), key
    return state["ratio"]


def overview(b, station, *_):
    j = station - 1
    run = state["run"]
    if run is None:
        return pn.pane.Markdown("### Running the solver for the first time...\nThe plots appear when it finishes.")
    m, x = run["model"], run["site"]["rec_x"]
    f, R, f0, A0 = ratios(b)
    xx = np.linspace(x[0], x[-1], 300)
    inband = (m.f0_1d(x) > 0.8) & (m.f0_1d(x) < 8.0)

    fig1 = Figure(figsize=(6.4, 4)); ax = fig1.subplots()
    sel = (f >= 0.6) & (f <= 9)
    pc = ax.pcolormesh(x, f[sel], R[sel], shading="nearest", cmap="viridis", vmin=1,
                       vmax=float(np.percentile(R[sel], 99)))
    ax.plot(xx, m.f0_1d(xx), "w--", lw=1.5, label="1D: Vs₁ / 4H(x)")
    ax.plot(x, f0, "o", mfc="r", mec="w", ms=5, label="picked f₀")
    ax.axvline(x[j], color="w", lw=1, alpha=0.8)
    ax.set(yscale="log", ylim=(0.6, 9), xlabel="receiver x (m)", ylabel="frequency (Hz)",
           title="H/V proxy (site / rock spectral ratio) along the line")
    ax.legend(fontsize=8, loc="upper right"); fig1.colorbar(pc, ax=ax, label="ratio")
    fig1.tight_layout()

    fig2 = Figure(figsize=(5.2, 4)); ax = fig2.subplots(2, 1, sharex=True)
    ax[0].plot(xx, m.f0_1d(xx), "k", label="1D Vs₁ / 4H(x)")
    ax[0].plot(x[inband], f0[inband], "o", color="C3", ms=4, label="picked f₀")
    ax[0].set(ylabel="f₀ (Hz)", yscale="log", title="f₀ and the thickness it implies"); ax[0].legend(fontsize=8)
    ax[1].plot(xx, m.thickness(xx), "k", label="true H(x)")
    ax[1].plot(x[inband], m.vs1 / (4 * f0[inband]), "o", color="C3", ms=4, label="Vs₁ / 4f₀")
    for a in ax:
        a.axvline(x[j], color="0.5", lw=1, ls=":")
    ax[1].set(xlabel="receiver x (m)", ylabel="H (m)"); ax[1].invert_yaxis(); ax[1].legend(fontsize=8)
    fig2.tight_layout()
    return pn.Row(pn.pane.Matplotlib(fig1, tight=True, format="svg"),
                  pn.pane.Matplotlib(fig2, tight=True, format="svg"))


def tile(name, value, unit, note, color):
    return pn.pane.HTML(
        f"""<div style="border:1px solid {color};border-left:6px solid {color};border-radius:6px;padding:8px 12px;
        height:92px;box-sizing:border-box">
        <div style="font-size:12px;opacity:0.75">{name}</div>
        <div style="font-size:26px;font-weight:600;line-height:1.2">{value}<span style="font-size:14px;font-weight:400"> {unit}</span></div>
        <div style="font-size:12px;opacity:0.75">{note}</div></div>""", width=250, sizing_mode="fixed")


def station_view(b, station, *_):
    j = station - 1
    run = state["run"]
    if run is None:
        return pn.pane.Markdown("")
    m, x = run["model"], run["site"]["rec_x"]
    f, R, f0, A0 = ratios(b)
    sel = (f >= 0.6) & (f <= 9)
    Hj = float(m.thickness(x[j]))
    mj = H.Model(**{**m.__dict__, "h0": Hj, "dip": 0.0})
    ff = np.linspace(0.3, 9, 3000)
    tf = mj.transfer_1d(ff)
    lo = ff < 1.5 * mj.f0_1d(0)                      # around the fundamental
    A_true, f_true = float(tf[lo].max()), float(ff[lo][np.argmax(tf[lo])])

    fig3 = Figure(figsize=(6.4, 3.8)); ax = fig3.subplots()
    ax.plot(f[sel], R[sel, j], "C0", lw=2, label="H/V proxy from the noise")
    ax.plot(ff, tf, "k--", lw=1, label=f"true 1D amplification (H = {Hj:.1f} m)")
    ax.axvline(f0[j], color="C3", lw=1, label=f"picked f₀ = {f0[j]:.2f} Hz")
    ax.plot(f0[j], A0[j], "o", color="C3")
    ax.set(xscale="log", xlim=(0.6, 9), ylim=(0, max(1.15 * R[sel, j].max(), 1.1 * A_true)),
           xlabel="frequency (Hz)", ylabel="amplification", title=f"Station {j + 1}, x = {x[j]:.0f} m")
    ax.legend(fontsize=8); fig3.tight_layout()

    # ---- prediction from H/V alone (what a field survey would know: f0, peak, and Vs1)
    H_pred = m.vs1 / (4 * f0[j])
    T0 = 1 / f0[j]
    floors = max(1, int(round(10 * T0)))            # rule of thumb: building period ~ 0.1 s per storey
    v_pred = H.vs30(m.vs1, H_pred, m.vs2)
    v_true = H.vs30(m.vs1, Hj, m.vs2)
    risk = "high" if A0[j] >= 4 else "moderate" if A0[j] >= 2 else "low"
    color = {"high": "#d62728", "moderate": "#ff7f0e", "low": "#2ca02c"}[risk]
    tiles = pn.FlexBox(
        tile("Resonance frequency f₀", f"{f0[j]:.2f}", "Hz", f"true 1D {f_true:.2f} Hz", "#1f77b4"),
        tile("Predicted peak amplification", f"×{A0[j]:.1f}", "", f"true 1D ×{A_true:.1f}; {risk} amplification", color),
        tile("Depth to rock, Vs₁/4f₀", f"{H_pred:.0f}", "m", f"true {Hj:.1f} m ({100 * (H_pred / Hj - 1):+.0f} %)", "#9467bd"),
        tile("Buildings at risk of resonance", f"~{floors}", "storeys", f"natural period ≈ T₀ = {T0:.2f} s", color),
        tile("Vs30 from H/V depth", f"{v_pred:.0f}", "m/s", f"class {H.site_class(v_pred)[0]}; true {v_true:.0f} m/s, class {H.site_class(v_true)[0]}", "#8c564b"),
        flex_wrap="wrap", gap="8px", sizing_mode="stretch_width")
    note = pn.pane.Markdown(
        "How the prediction works: f₀ and the peak height come straight from the H/V curve; the depth uses "
        "f₀ = Vs₁/4H with the known soft-layer speed; the building rule of thumb is a natural period of "
        "about 0.1 s per storey, so buildings with T ≈ T₀ shake in resonance with the ground. The *true* "
        "values use the real local thickness in the analytic 1D model. In field HVSR the peak height is "
        "only a rough (often low) estimate of the amplification; here the proxy is a site/rock ratio, so the "
        "SH simplification makes it a direct one.", styles={"font-size": "12px"})
    card = pn.Card(tiles, note, title=f"Site amplification predicted from H/V at station {j + 1}",
                   collapsible=False, sizing_mode="stretch_width")
    return pn.Row(pn.pane.Matplotlib(fig3, tight=True, format="svg", width=520, sizing_mode="fixed"), card)


w_station = pn.widgets.IntSlider(
    name=f"Station whose H/V curve is shown (1 = x {REC_X[0]:.0f} m ... {len(REC_X)} = x +{REC_X[-1]:.0f} m, every {REC_X[1] - REC_X[0]:.0f} m)",
    start=1, end=len(REC_X), value=len(REC_X) // 2 + 1, sizing_mode="stretch_width")
summary = pn.pane.Markdown(
    "**Simplification:** this is an SH (antiplane) model with only a horizontal component, so the "
    "'H/V' shown is the site-to-rock spectral ratio (surface over soft layer ÷ surface of bare rock, "
    "same noise). Field HVSR divides by the vertical component instead; both peak near the same f₀ "
    "for a strong impedance contrast.", styles={"font-size": "12px"})


controls = pn.Column(
    "### Model (needs a solver run)", w_grid, w_vs1, w_vs2, w_rho, w_h0, w_dip, w_q, contrast_note,
    "### Noise", w_surf, w_T, w_run, bar, status,
    "### Analysis (instant)", w_ko,
    pn.pane.Markdown("2D depth slice, SH wave equation ρu_tt = ∇·(μ∇u) with a free surface, solved by "
                     "FD leapfrog with a conservative variable-μ operator. 60 noise sources in the rock fire "
                     "band-limited random noise simultaneously, so waves arrive from all directions."),
)
pn.template.FastListTemplate(
    title="HVSR-style site response of a dipping soft layer",
    sidebar=[controls],
    main=[pn.Column(wave_pane, movie_note),
          pn.bind(overview, w_ko, w_station, refresh),
          pn.Column("### H/V curve at one station (pick it after the run)", w_station,
                    pn.bind(station_view, w_ko, w_station, refresh), summary)],
).servable()
