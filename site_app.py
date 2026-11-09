"""
site_app.py -- Panel UI for passive Love-wave site characterisation.

Run with:   panel serve site_app.py --show
All numerics live in love_waves.py; this file only wires widgets to plots.

Two kinds of controls:
  * Site model and grid: need one new FD run (a few seconds on the fast grid, ~30-60 s on
    the standard grid), started with "Run solver". The wavefield is drawn live while the
    solver runs, then replayed as a looping animation (play/pause/scrub with the player).
    Results are cached in memory and on disk (.cache/), so each model is only solved once.
  * Noise sources, array and picking: only re-weight the stored Green's functions, so the
    plots update instantly.
"""
import asyncio, pickle, time
from pathlib import Path
import numpy as np
import panel as pn
from matplotlib.figure import Figure
import love_waves as L

pn.extension(sizing_mode="stretch_width")

CACHE_DIR = Path(__file__).parent / ".cache"
GRIDS = {"fast (dx = 2 m, up to 12 Hz)": (2.0, 12.0), "standard (dx = 1 m, up to 20 Hz)": (1.0, 20.0)}
C_TRIAL = np.linspace(100, 1300, 601)


def love_line(site, dx, progress=None, on_snapshot=None):
    """One FD impulse run for a site, cached in memory (all sessions) and on disk."""
    key = f"love_v2_dx{dx}_b{site.b1:.0f}-{site.b2:.0f}_H{site.H:.1f}_r{site.r1:.0f}-{site.r2:.0f}"
    mem = pn.state.cache.setdefault("love", {})
    if key in mem:
        return mem[key]
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        ll = pickle.loads(path.read_bytes())
    else:
        ll = L.LoveLine(site, L.Grid(dx=dx), progress=progress, on_snapshot=on_snapshot)
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(ll))
    mem[key] = ll
    return ll


# ---------------------------------------------------------------- widgets
w_grid = pn.widgets.Select(name="Grid", options=list(GRIDS), value=list(GRIDS)[0])
w_b1 = pn.widgets.IntSlider(name="Soil shear speed β₁ (m/s)", start=150, end=400, step=10, value=250)
w_b2 = pn.widgets.IntSlider(name="Bedrock shear speed β₂ (m/s)", start=400, end=1500, step=50, value=800)
w_H = pn.widgets.IntSlider(name="Soil thickness H (m)", start=5, end=30, step=1, value=15)
w_rho = pn.widgets.FloatSlider(name="Density ratio ρ₂/ρ₁ (ρ₁ = 1800 kg/m³)", start=1.0, end=1.4, step=0.05, value=1.15)
w_run = pn.widgets.Button(name="Run solver", button_type="primary")
bar = pn.indicators.Progress(name="solver", value=0, max=100, visible=False)
status = pn.pane.Markdown("")

w_layout = pn.widgets.Select(name="Noise sources", value="both", options={
    "both ends (in-line)": "both", "left end only": "left", "right end only": "right",
    "everywhere along the line": "everywhere", "off-line, at angle θ": "offline"})
w_gap = pn.widgets.IntSlider(name="Gap to nearest noise source (m)", start=0, end=150, step=10, value=100)
w_theta = pn.widgets.IntSlider(name="Off-line angle θ (deg)", start=0, end=60, step=5, value=20)
w_depth = pn.widgets.Select(name="Source depth (m)", options=list(L.SOURCE_DEPTHS), value=0.0)
w_win = pn.widgets.Select(name="Noise record", value=0,
                          options={"expected (infinite)": 0, "1 window": 1, "10 windows": 10, "100 windows": 100})
w_nch = pn.widgets.IntSlider(name="Channels", start=12, end=48, step=4, value=48)
w_dch = pn.widgets.FloatSlider(name="Channel spacing (m)", start=1, end=6, step=0.5, value=4)
w_band = pn.widgets.RangeSlider(name="Picking band (Hz)", start=2, end=20, step=0.5, value=(4, 12))
w_track = pn.widgets.Checkbox(name="Track the fundamental (instead of the image peak)", value=False)
refresh = pn.widgets.IntInput(value=0, visible=False)
state = {"ll": None, "lim": 1.0}


def current_site():
    return L.Site(b1=w_b1.value, b2=w_b2.value, H=w_H.value, r1=1800.0, r2=1800.0 * w_rho.value)


async def rerun(_=None):
    """Run (or load) the FD solve in a worker thread so the page stays responsive."""
    site, (dx, fmax) = current_site(), GRIDS[w_grid.value]
    w_run.disabled, bar.visible, bar.value = True, True, 0
    status.object = "*Running the SH finite-difference solver (one impulse run)...*"
    t0 = time.time()

    def progress(q):
        bar.value = int(100 * q)
        print(f"  FD solver: {100 * q:3.0f}%", flush=True)

    grid = L.Grid(dx=dx)
    player.disabled = True
    player.pause()

    last = [0.0]

    def live(t, frame):                 # called from the worker thread for every new frame
        if time.time() - last[0] < 0.4:  # draw at most ~2.5 frames/s so the solver isn't slowed
            return
        last[0] = time.time()
        lim = max(np.percentile(np.abs(frame), 99.5), 1e-30)
        wave_pane.object = wave_figure(frame, t, site, grid.half_width, grid.depth, lim, live=True)

    try:
        state["ll"] = await asyncio.to_thread(love_line, site, dx, progress, live)
        g = state["ll"].grid
        status.object = f"Solver ready ({time.time() - t0:.1f} s): grid {g.nx} x {g.nz}, dx = {dx} m."
        w_band.end = fmax
        w_band.value = (w_band.value[0], min(w_band.value[1], fmax))
        start_movie()
    except Exception as err:
        status.object = f"**Solver failed:** {err!r}"
        raise
    finally:
        w_run.disabled, bar.visible = False, False
    refresh.value += 1


w_run.on_click(rerun)
pn.state.onload(rerun)


# ---------------------------------------------------------------- wavefield movie
ANIM_GAP = 250.0   # distance [m] from the first sensor to the source shown in the animation


def wave_figure(frame, t, site, half_width, depth, lim, live=False):
    """One frame of the SH wavefield over the model cross-section.

    The solver fires its impulse at x = 0, but the model is laterally uniform, so the field
    of a source anywhere else is the same field shifted sideways. For display we place the
    source ANIM_GAP m before the first sensor, so the animation shows a distant noise
    source's waves travelling in and sweeping across the array."""
    rx = L.channels(w_nch.value, w_dch.value, -90.0)
    xs = rx[0] - ANIM_GAP
    fig = Figure(figsize=(11, 3.2)); ax = fig.subplots()
    ax.imshow(frame, extent=[xs - half_width, xs + half_width, depth, 0], cmap="seismic", vmin=-lim,
              vmax=lim, aspect="auto", interpolation="bilinear")
    ax.axhline(site.H, color="k", lw=1)
    ax.plot(rx, np.ones_like(rx), "gv", ms=4)
    ax.plot(xs, 1, "k*", ms=12, clip_on=False)
    x0 = xs - 40
    ax.text(xs + 12, site.H / 2, f"soil β₁ = {site.b1:.0f} m/s", va="center", fontsize=8)
    ax.text(xs + 12, site.H + 8, f"bedrock β₂ = {site.b2:.0f} m/s", va="center", fontsize=8)
    what = "solver running" if live else "replay"
    ax.set(xlim=(x0, xs + half_width), ylim=(min(depth, 120), 0), xlabel="x (m)", ylabel="depth (m)",
           title=f"Waves from a noise source (star) {ANIM_GAP:.0f} m from the array (green), t = {t:.2f} s  [{what}]")
    fig.tight_layout(); return fig


def show_frame(i):
    """Draw stored frame i of the current model (called by the player)."""
    ll = state["ll"]
    if ll is None:
        return
    t = L.LoveLine.FRAME_TIMES[i]
    wave_pane.object = wave_figure(ll.snaps[t], t, ll.site, ll.grid.half_width, ll.grid.depth, state["lim"])


def start_movie():
    """After a solve: fix one colour scale for all frames (Love waves, not the source
    singularity, set it) and loop the animation."""
    ll = state["ll"]
    late = [ll.snaps[t] for t in L.LoveLine.FRAME_TIMES if t >= 0.4]
    state["lim"] = float(np.percentile(np.abs(np.stack(late)), 99.5))
    player.disabled = False
    player.value = 0
    show_frame(0)
    player.play()


wave_pane = pn.pane.Matplotlib(None, tight=True, format="png", dpi=90, height=300, sizing_mode="stretch_width")
player = pn.widgets.Player(name="time step", start=0, end=len(L.LoveLine.FRAME_TIMES) - 1, value=0,
                           loop_policy="loop", interval=150, show_loop_controls=False, width=380,
                           sizing_mode="fixed", disabled=True)
player.param.watch(lambda e: show_frame(e.new), "value")


# ---------------------------------------------------------------- plots
def gather_plot(ll, U, off):
    fig = Figure(figsize=(5, 3.4)); ax = fig.subplots()
    t, tr = ll.to_time(U, 2.0, ll.f[-1] - 2)
    tr = tr / np.maximum(np.abs(tr).max(axis=0), 1e-30)
    step = max(1, len(off) // 24)
    for j in range(0, len(off), step):
        ax.plot(off[j] + 0.8 * step * (off[1] - off[0]) * tr[:, j], t, "k", lw=0.6)
    ax.set(ylim=(1.5, 0), xlabel="offset from channel 0 (m)", ylabel="lag (s)", title="Virtual shot gather (noise correlation)")
    fig.tight_layout(); return fig


def dispersion_plot(ll, img, fp, cp, site):
    fig = Figure(figsize=(6.5, 4)); ax = fig.subplots()
    ax.pcolormesh(ll.f, C_TRIAL, img.T, cmap="magma", shading="auto", vmin=0, vmax=1)
    fs = np.linspace(0.5, ll.f[-1], 250)
    for n, cn in enumerate(L.love_modes(fs, site, 3)):
        ax.plot(fs, cn, "c--", lw=1, label="exact modes" if n == 0 else None)
    ax.plot(fp, cp, "o", ms=3.5, mfc="w", mec="k", mew=0.5, label="picks")
    ax.set(xlim=(2, w_band.end), ylim=(100, min(1300, 1.3 * site.b2)), xlabel="f (Hz)", ylabel="phase velocity (m/s)",
           title="Dispersion image (phase-shift transform)")
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout(); return fig


def profile_plot(site, p):
    fig = Figure(figsize=(3.6, 4)); ax = fig.subplots()
    z = np.linspace(0, 40, 400)
    ax.plot(site.profile(z)[0], z, "k", lw=3, alpha=0.35, label="true")
    ax.plot(L.Site(p[0], p[2], p[1]).profile(z)[0], z, "r", label="inverted")
    ax.axhline(30, color="0.5", ls=":", lw=1)
    ax.invert_yaxis(); ax.set(xlabel="β (m/s)", ylabel="depth (m)", title="Vs profile")
    ax.legend(fontsize=8); fig.tight_layout(); return fig


def view(*_):
    ll = state["ll"]
    if ll is None:
        return pn.pane.Markdown("### Running the solver for the first time...\nThe plots appear when it finishes.")
    site = ll.site
    rx = L.channels(w_nch.value, w_dch.value, -90.0)
    off = rx - rx[0]
    src = L.line_sources(w_layout.value, rec_x=rx, theta_deg=w_theta.value, reach=120.0,
                         gap=w_gap.value, standoff=max(w_gap.value, 10.0),
                         depth_index=L.SOURCE_DEPTHS.index(w_depth.value))
    try:
        U = (ll.noise_gather(rx, src) if w_win.value == 0 else ll.random_gather(rx, src, w_win.value))
    except ValueError as err:
        return pn.pane.Markdown(f"**{err}**: use fewer channels or a smaller spacing.")
    img = L.phase_shift(U, off, ll.f, C_TRIAL)
    fp, cp = L.pick_ridge(img, C_TRIAL, ll.f, *w_band.value, track=w_track.value)
    p, misfit = L.invert_site(fp, cp, start=(0.8 * site.b1, 10.0, 0.8 * site.b2), r1=site.r1, r2=site.r2)
    v_true, v_inv = L.vs30(site.b1, site.H, site.b2), L.vs30(*p)
    exact = np.array([L.love_root(x, site) for x in fp])
    table = pn.pane.Markdown(f"""
| | β₁ (m/s) | H (m) | β₂ (m/s) | Vs30 (m/s) | NEHRP class |
|---|---|---|---|---|---|
| true | {site.b1:.0f} | {site.H:.1f} | {site.b2:.0f} | {v_true:.0f} | {L.site_class(v_true)} |
| inverted | {p[0]:.0f} | {p[1]:.1f} | {p[2]:.0f} | **{v_inv:.0f}** ({100 * (v_inv / v_true - 1):+.1f} %) | **{L.site_class(v_inv)}** |

Inversion misfit {misfit:.1f} m/s (assumes the picks are the fundamental mode). Pick error vs exact
fundamental: mean {100 * np.mean(cp / exact - 1):+.1f} %, max {100 * np.max(np.abs(cp / exact - 1)):.1f} %.
Array aperture {off[-1]:.0f} m; mode-1 cutoff {site.cutoff(1):.1f} Hz.
""")
    return pn.Column(
        pn.Row(pn.pane.Matplotlib(gather_plot(ll, U, off), tight=True, format="svg", width=420, sizing_mode="fixed"),
               pn.pane.Matplotlib(dispersion_plot(ll, img, fp, cp, site), tight=True, format="svg")),
        pn.Row(pn.pane.Matplotlib(profile_plot(site, p), tight=True, format="svg", width=360, sizing_mode="fixed"),
               table))


controls = pn.Column(
    "### Site model (needs a solver run)", w_grid, w_b1, w_b2, w_H, w_rho, w_run, bar, status,
    "### Noise (instant)", w_layout, w_gap, w_theta, w_depth, w_win,
    "### Array and picking (instant)", w_nch, w_dch, w_band, w_track,
    pn.pane.Markdown("SH waves in a vertical plane: ρu_tt = ∇·(μ∇u), FD leapfrog with a conservative "
                     "variable-μ operator and a free surface. One impulse run gives every noise "
                     "correlation (lateral uniformity + reciprocity)."),
)
pn.template.FastListTemplate(
    title="Passive Love-wave site characterisation (Vs30 from ambient noise)",
    sidebar=[controls],
    main=[pn.Column(wave_pane, player),
          pn.bind(view, w_layout, w_gap, w_theta, w_depth, w_win, w_nch, w_dch, w_band, w_track, refresh)],
).servable()
