"""
bridge.py -- glue between the Web Worker (worker.js) and the simulation module hvsr.py.

Runs inside Pyodide in the browser. It holds no physics: it builds an hvsr.Model from the
slider values, calls the same hvsr.simulate_noise / spectral_ratio / pick_f0 used by the
notebooks, and packs the results into plain JavaScript objects for the page.
"""
import time
import numpy as np
import hvsr as H

REC_X = np.arange(-150.0, 150.1, 10.0)                  # 31 surface receivers, every 10 m
FRAMES = np.round(np.arange(1.0, 7.001, 0.05), 3)       # animation frame times [s]
_cache = {}                                             # finished runs, per model
_state = {"run": None, "ratio": None, "ratio_key": None}


def _js(obj):
    """Python dict/list/numpy -> plain JS objects and typed arrays (postMessage-safe)."""
    from js import Object
    from pyodide.ffi import to_js
    return to_js(obj, dict_converter=Object.fromEntries)


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).ravel())


def _f64(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())


def make_model(p):
    return H.Model(vs1=float(p["vs1"]), vs2=float(p["vs2"]), rho1=1800.0,
                   rho2=1800.0 * float(p["rho_ratio"]), h0=float(p["h0"]),
                   dip=float(p["dip"]), q1=float(p["q"]))


def model_info(params):
    """Impedance contrast and 1D f0 at x = 0 (shown before any run)."""
    p = params.to_py() if hasattr(params, "to_py") else params
    m = make_model(p)
    return _js(dict(contrast=float(m.contrast), f0=float(m.f0_1d(0.0))))


def run(params, progress=None, on_frame=None):
    """Site run + bare-rock reference run with the same noise sources.

    params   : dict of slider values (vs1, vs2, rho_ratio, h0, dip, q, n_surface, T, dx)
    progress : JS callback(fraction)
    on_frame : JS callback(t, Float32Array frame), called a few times per second while
               the soft-layer run is going, so the page can draw the live wavefield
    Returns the geometry and every animation frame (for the replay)."""
    from pyodide.ffi import to_js
    p = params.to_py() if hasattr(params, "to_py") else params
    key = tuple(sorted((k, float(v)) for k, v in p.items()))
    t0 = time.time()
    m = make_model(p)
    dx, T, ns = float(p["dx"]), float(p["T"]), int(p["n_surface"])
    if key not in _cache:
        grid = H.default_grid(dx)
        src = H.noise_sources(n_surface=ns)
        last = [0.0]

        def live(t, frame):
            if on_frame is not None and time.time() - last[0] > 0.25:
                last[0] = time.time()
                on_frame(float(t), to_js(_f32(frame)))

        p1 = (lambda q: progress(0.5 * q)) if progress else None
        p2 = (lambda q: progress(0.5 + 0.5 * q)) if progress else None
        site = H.simulate_noise(m, grid, T, REC_X, src=src, progress=p1, snapshots=FRAMES,
                                on_snapshot=live)
        ref = H.simulate_noise(m, grid, T, REC_X, src=src, homogeneous=True, progress=p2)
        _cache[key] = dict(model=m, src=src, site=site, ref=ref, grid=grid)
    r = _cache[key]
    _state["run"], _state["ratio_key"] = r, None
    s = r["site"]
    times = sorted(s["snaps"])
    frames = np.stack([s["snaps"][t] for t in times])
    xx = np.linspace(s["x_snap"][0], s["x_snap"][-1], 200)
    return _js(dict(
        seconds=time.time() - t0, dx=dx, T=T, dt_solver=float(s["dt_solver"]),
        x_snap=_f64(s["x_snap"]), z_snap=_f64(s["z_snap"]), frame_times=_f64(times),
        nz=frames.shape[1], nx=frames.shape[2], frames=_f32(frames),
        lim=float(np.percentile(np.abs(frames), 99.5)) or 1.0,
        src_x=_f64(r["src"][0]), src_z=_f64(r["src"][1]), rec_x=_f64(REC_X),
        iface_x=_f64(xx), iface_z=_f64(m.thickness(xx)), vs1=m.vs1, vs2=m.vs2))


def _ratios(b):
    r = _state["run"]
    if _state["ratio_key"] != b:
        f, R = H.spectral_ratio(r["site"], r["ref"], b=b)
        f0, A0 = H.pick_f0(f, R)
        _state["ratio"], _state["ratio_key"] = (f, R, f0, A0), b
    return _state["ratio"]


def analysis(b):
    """Spectral ratio along the line, f0 picks, and the thickness they imply."""
    r = _state["run"]
    m, x = r["model"], r["site"]["rec_x"]
    f, R, f0, A0 = _ratios(float(b))
    sel = (f >= 0.6) & (f <= 9.0)
    xx = np.linspace(x[0], x[-1], 300)
    inband = (m.f0_1d(x) > 0.8) & (m.f0_1d(x) < 8.0)
    return _js(dict(
        x=_f64(x), f=_f64(f[sel]), R=_f64(R[sel]), nf=int(sel.sum()), nx=len(x),
        vmax=float(np.percentile(R[sel], 99)), f0=_f64(f0), A0=_f64(A0),
        inband=[bool(v) for v in inband], H_implied=_f64(m.vs1 / (4 * f0)),
        xx=_f64(xx), f0_1d=_f64(m.f0_1d(xx)), H_true=_f64(m.thickness(xx)),
        err_mean=float(np.mean(100 * (f0[inband] / m.f0_1d(x[inband]) - 1))) if inband.any() else 0.0))


def station(b, j):
    """H/V curve at receiver j against the true local 1D amplification, and the site
    predictions a field survey would make from that curve alone."""
    r = _state["run"]
    m, x = r["model"], r["site"]["rec_x"]
    f, R, f0, A0 = _ratios(float(b))
    j = int(j)
    sel = (f >= 0.6) & (f <= 9.0)
    Hj = float(m.thickness(x[j]))
    mj = H.Model(**{**m.__dict__, "h0": Hj, "dip": 0.0})
    ff = np.geomspace(0.6, 9.0, 600)
    tf = mj.transfer_1d(ff)
    lo = ff < 1.5 * mj.f0_1d(0)
    A_true, f_true = float(tf[lo].max()), float(ff[lo][np.argmax(tf[lo])])
    H_pred = m.vs1 / (4 * f0[j])
    T0 = 1 / f0[j]
    v_pred, v_true = H.vs30(m.vs1, H_pred, m.vs2), H.vs30(m.vs1, Hj, m.vs2)
    risk = "high" if A0[j] >= 4 else "moderate" if A0[j] >= 2 else "low"
    return _js(dict(
        j=j, x=float(x[j]), f=_f64(f[sel]), R=_f64(R[sel, j]), ff=_f64(ff), tf=_f64(tf),
        f0=float(f0[j]), A0=float(A0[j]), H_true=Hj, f_true=f_true, A_true=A_true,
        H_pred=float(H_pred), T0=float(T0), floors=max(1, int(round(10 * T0))), risk=risk,
        vs30_pred=float(v_pred), vs30_true=float(v_true),
        class_pred=H.site_class(v_pred)[0], class_true=H.site_class(v_true)[0]))
