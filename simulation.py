"""
simulation.py -- numerical core of the directional-noise interferometry app.

Physical question
-----------------
Ambient-noise interferometry assumes noise reaches a station pair equally from
all directions; then the time-averaged cross-correlation C_AB(t) of the two
records turns into the inter-station Green's function. Real noise is
directional: in a sedimentary basin at ~1-3 Hz it comes mostly from roads,
towns, industry or a nearby coast. This module measures how a directional
noise field biases the travel time, and so the basin velocity, read from C_AB
for station pairs 1-5 km apart.

Numerical method (course module 06, hyperbolic PDEs)
----------------------------------------------------
2D scalar wave equation with a damping (sponge) term near the edges:

    u_tt + sigma(x) u_t = c(x)^2 (u_xx + u_yy) + f(x, t)

discretised with
  * 2nd-order central differences (leapfrog) in time,
  * a 4th-order 5-point stencil for each second space derivative,
  * a quadratic sponge sigma(x) in a layer around the domain (absorbing BC),
  * zero initial conditions u = u_t = 0 and u = 0 on the outer edge.
Stability: c_max dt / dx <= sqrt(3/8) ~ 0.61 for this stencil in 2D.

Interferometry via reciprocity
------------------------------
Noise sources sit on a ring of radius R around the pair. Instead of one run per
source, we run ONE impulse simulation per station and record it at every ring
point: by reciprocity G(x_s -> A) = G(A -> x_s). For mutually uncorrelated
sources the expected correlation is a weighted sum over sources,

    C_AB(w) = sum_s  B(theta_s) * conj(G_A,s(w)) * G_B,s(w) * |W(w)|^2,

so changing the source pattern B(theta) only re-weights stored spectra.

Units: km, s, km/s throughout. FFT convention: numpy (forward ~ exp(-i w t)).
"""
import copy
from dataclasses import dataclass, field
import numpy as np
from scipy.special import hankel1, i0e
from scipy.ndimage import gaussian_filter

CFL_LIMIT = np.sqrt(3.0 / 8.0)   # stability limit of leapfrog + 4th-order Laplacian

# Default physical scale: a sedimentary basin probed with ~2 Hz ambient noise
# (traffic / local cultural noise and high-frequency ocean noise).
C0 = 2.2        # wave speed [km/s] (2200 m/s, slow basin-fill surface waves)
F0 = 2.0        # Ricker peak frequency [Hz]  ->  wavelength C0/F0 = 1.1 km
DX = 0.05       # grid spacing [km]: 11 points per wavelength at 2*F0
RING_R = 12.0   # radius of the noise-source ring [km] (~10 wavelengths: far field)


# ----------------------------------------------------------------------------
# Model / grid
# ----------------------------------------------------------------------------
@dataclass
class Model:
    """Square 2D grid centred on the origin, with a sponge layer around it."""
    half_width: float = 13.5    # half-width of the interior (non-sponge) region [km]
    dx: float = DX              # grid spacing [km]
    c0: float = C0              # background wave speed [km/s]
    n_sponge: int = 80          # sponge thickness [cells]
    cfl: float = 0.5            # Courant number c_max dt / dx
    c: np.ndarray = field(default=None, repr=False)  # optional c(y, x) [km/s]

    def __post_init__(self):
        m = int(round(self.half_width / self.dx)) + self.n_sponge
        self.x = self.dx * np.arange(-m, m + 1)          # node coordinates [km]
        self.n = self.x.size
        if self.c is None:
            self.c = np.full((self.n, self.n), self.c0)
        self.dt = self.cfl * self.dx / self.c.max()       # time step [s]
        if self.cfl > CFL_LIMIT:
            print(f"warning: CFL {self.cfl:.2f} > stability limit {CFL_LIMIT:.2f}")

    def node(self, xy):
        """Index (iz, ix) of the grid node nearest to point(s) xy = (x, y)."""
        xy = np.atleast_2d(xy)
        i = np.rint(xy / self.dx).astype(int) + self.n // 2
        return i[:, 1], i[:, 0]

    def sponge(self):
        """Damping profile sigma(y, x) [1/s]: zero inside, growing ~ d^2 in the layer."""
        L = self.n_sponge * self.dx
        d = np.clip(np.abs(self.x) - (self.x[-1] - L), 0, None) / L  # 0..1 depth into layer
        # sigma_max tuned by test (see verification notebook): ~0.5% reflection for 80 cells
        prof = 20.0 * self.c0 / L * d**2
        return np.maximum(prof[:, None], prof[None, :])


def random_medium(model, rel_std=0.03, corr_len=0.1, seed=0):
    """Smooth random velocity c = c0 (1 + dc), std(dc) = rel_std, Gaussian correlation."""
    rng = np.random.default_rng(seed)
    dc = gaussian_filter(rng.standard_normal((model.n, model.n)), corr_len / model.dx)
    dc *= rel_std / dc.std()
    return model.c0 * (1.0 + dc)


# ----------------------------------------------------------------------------
# Source wavelet and FD solver
# ----------------------------------------------------------------------------
def ricker(t, f0, t0):
    """Ricker wavelet with peak frequency f0 [Hz], centred at t0 [s]."""
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def laplacian4(u, dx, out):
    """4th-order Laplacian (-1, 16, -30, 16, -1)/12 per axis; 2-cell border left at 0."""
    c = u[2:-2, 2:-2]
    out[2:-2, 2:-2] = (16.0 * (u[1:-3, 2:-2] + u[3:-1, 2:-2] + u[2:-2, 1:-3] + u[2:-2, 3:-1])
                       - (u[:-4, 2:-2] + u[4:, 2:-2] + u[2:-2, :-4] + u[2:-2, 4:])
                       - 60.0 * c) / (12.0 * dx * dx)
    return out


def simulate(model, src_xy, rec_xy, f0, T, t0=None, snapshots=(), progress=None):
    """
    Solve u_tt + sigma u_t = c^2 lap(u) + w(t) delta(x - x_src) by leapfrog.

    src_xy   : (x, y) of the point source [km]
    rec_xy   : (nrec, 2) receiver coordinates [km] (snapped to nearest nodes)
    f0, t0   : Ricker peak frequency [Hz] and delay [s] (default 1.5/f0)
    T        : record length [s]
    snapshots: optional step indices at which to save the full wavefield
    progress : optional callback progress(fraction) called every ~10% of the run
    Returns t (nt,), traces (nt, nrec), dict of snapshots.
    """
    t0 = 1.5 / f0 if t0 is None else t0
    dt, dx = model.dt, model.dx
    nt = int(np.ceil(T / dt)) + 1
    t = dt * np.arange(nt)
    w = ricker(t, f0, t0) / dx**2          # discrete delta: 1/dx^2 at one node
    isz, isx = model.node(src_xy)
    irz, irx = model.node(rec_xy)

    c2dt2 = (model.c * dt) ** 2
    s = 0.5 * model.sponge() * dt          # sigma dt / 2
    a_prev, a_den = (1.0 - s), 1.0 / (1.0 + s)

    u_prev = np.zeros((model.n, model.n))
    u = np.zeros_like(u_prev)
    lap = np.zeros_like(u_prev)
    traces = np.zeros((nt, irz.size))
    snaps = {}
    for it in range(nt):
        if progress and it % max(nt // 10, 1) == 0:
            progress(it / nt)
        traces[it] = u[irz, irx]
        if it in snapshots:
            snaps[it] = u.copy()
        # leapfrog with damping:  (1+s) u^{n+1} = 2u^n - (1-s) u^{n-1} + dt^2 (c^2 lap u^n + f^n)
        laplacian4(u, dx, lap)
        u_next = 2.0 * u - a_prev * u_prev + c2dt2 * lap
        u_next[isz, isx] += c2dt2[isz, isx] / model.c[isz, isx] ** 2 * w[it]  # = dt^2 f
        u_next *= a_den
        u_prev, u = u, u_next
    return t, traces, snaps


# ----------------------------------------------------------------------------
# Analytic 2D Green's function (homogeneous medium) for verification
# ----------------------------------------------------------------------------
def greens_hat(r, f, c):
    """Frequency-domain 2D Green's function of u_tt - c^2 lap u = delta, numpy FFT
    convention: conj[(i / 4c^2) H0^(1)(w r / c)]. Zero at f = 0 (singular there)."""
    kr = 2 * np.pi * np.asarray(f, float) * np.asarray(r, float) / c
    g = np.conj(1j / (4 * c**2) * hankel1(0, np.where(kr > 0, kr, 1.0)))
    return np.where(kr > 0, g, 0.0)


def analytic_trace(r, t, f0, c, t0=None):
    """Ricker-convolved 2D Green's function at distance r, sampled at times t."""
    t0 = 1.5 / f0 if t0 is None else t0
    dt, nt = t[1] - t[0], t.size
    nfft = 8 * nt                                # long padding: 2D G has a 1/t tail
    tt = dt * np.arange(nfft)
    f = np.fft.rfftfreq(nfft, dt)
    W = np.fft.rfft(ricker(tt, f0, t0)) * dt     # continuous-FT approximation
    return np.fft.irfft(W * greens_hat(r, f, c), nfft)[:nt] / dt


# ----------------------------------------------------------------------------
# Noise source distribution
# ----------------------------------------------------------------------------
def source_density(theta, theta0=0.0, kappa=0.0, eps=0.0):
    """
    Angular noise intensity B(theta) [1/rad], normalised to integrate to 1:
        B = eps / 2pi + (1 - eps) * vonMises(theta; theta0, kappa)
    theta0: dominant direction the noise comes FROM [rad]; kappa: concentration
    (0 = isotropic); eps: isotropic background fraction (0..1).
    """
    vm = np.exp(kappa * (np.cos(theta - theta0) - 1.0)) / (2 * np.pi * i0e(kappa))
    return eps / (2 * np.pi) + (1.0 - eps) * vm


def ring(model, R, spacing):
    """Source ring of radius R [km], points `spacing` km apart, snapped to grid nodes.
    Returns node coordinates (ns, 2) and their exact azimuths [rad]."""
    ns = int(np.ceil(2 * np.pi * R / spacing))
    phi = 2 * np.pi * np.arange(ns) / ns
    iz, ix = model.node(np.c_[R * np.cos(phi), R * np.sin(phi)])
    xy = np.c_[model.x[ix], model.x[iz]]
    return xy, np.arctan2(xy[:, 1], xy[:, 0]) % (2 * np.pi)


# ----------------------------------------------------------------------------
# Station-pair interferometry experiment
# ----------------------------------------------------------------------------
class PairExperiment:
    """
    Station A at (-r/2, 0) and B at (+r/2, 0): the pair axis is the x-axis and
    A -> B points along azimuth 0. Positive correlation lag = wave travelling A -> B
    (fed by sources behind A, theta ~ 180 deg); negative lag = B -> A (theta ~ 0).

    Two FD runs (impulse at A, impulse at B) are recorded on the source ring.
    Everything else (source pattern, frequency band, pair azimuth) re-weights them.
    """

    def __init__(self, model, r=3.0, R=RING_R, f0=F0, ring_spacing=0.15, progress=None):
        self.model, self.f0 = model, f0
        self.r = 2 * model.dx * round(r / (2 * model.dx))       # keep A, B on nodes
        self.xA, self.xB = np.array([-self.r / 2, 0.0]), np.array([self.r / 2, 0.0])
        self.src_xy, self.theta = ring(model, R, ring_spacing)
        self.dtheta = 2 * np.pi / self.theta.size

        # record length: farthest ring point to a station + wavelet duration
        T = (R + self.r) / model.c.min() + 6.0 / f0
        rec = np.vstack([self.src_xy, self.xB])                  # last receiver = station B
        # progress(fraction) callback, if given, spans both runs (0-0.5, 0.5-1)
        pA = (lambda q: progress(0.5 * q)) if progress else None
        pB = (lambda q: progress(0.5 + 0.5 * q)) if progress else None
        self.t, uA, _ = simulate(model, self.xA, rec, f0, T, progress=pA)
        _, uB, _ = simulate(model, self.xB, self.src_xy, f0, T, progress=pB)
        self.G_AB = uA[:, -1]                                    # direct A -> B (Ricker-filtered)
        uA = uA[:, :-1]

        # cross-spectra per source; nfft >= 2 nt so correlation lags do not wrap
        dt = model.dt
        self.nfft = 2 * self.t.size
        self.f = np.fft.rfftfreq(self.nfft, dt)
        UA, UB = np.fft.rfft(uA, self.nfft, axis=0), np.fft.rfft(uB, self.nfft, axis=0)
        self.X = np.conj(UA) * UB                                # (nf, ns)
        # source power spectrum |W|^2 of the Ricker (for spectral whitening)
        W = np.fft.rfft(ricker(self.t, f0, 1.5 / f0), self.nfft)
        self.W2 = np.abs(W) ** 2
        self.lags = dt * (np.arange(self.nfft) - self.nfft // 2)

    # -- correlation -----------------------------------------------------------
    def weights(self, theta0=0.0, kappa=0.0, eps=0.0, pair_az=0.0):
        """Source weights B(theta_s) dtheta. Rotating the pair by pair_az is the
        same as rotating the source pattern by -pair_az (exact for a homogeneous medium)."""
        return source_density(self.theta, theta0 - pair_az, kappa, eps) * self.dtheta

    def correlate(self, w, fc=F0, rel_bw=0.2):
        """Expected correlation C_AB(lag) for source weights w, after spectral whitening
        (divide by |W|^2) and a Gaussian band-pass centred on fc [Hz] with std rel_bw*fc."""
        band = np.exp(-0.5 * ((self.f - fc) / (rel_bw * fc)) ** 2)
        white = self.W2 / (self.W2 + 1e-3 * self.W2.max())        # water-level whitening
        Chat = (self.X @ w) * band * white / (self.W2 + 1e-12)
        return np.fft.fftshift(np.fft.irfft(Chat, self.nfft))

    # -- measurements ------------------------------------------------------------
    def measure(self, C, Cref, fc=F0, t_exp=None):
        """Travel-time bias of C relative to Cref for this pair (see measure_bias)."""
        return measure_bias(self.lags, C, Cref, fc, self.r / self.model.c0 if t_exp is None else t_exp)

    def weaver_prediction(self, theta0, kappa, eps, fc, pair_az=0.0):
        """Stationary-phase bias for this pair and a von Mises source pattern (see weaver_bias)."""
        B = lambda th: source_density(th, theta0 - pair_az, kappa, eps)
        return weaver_bias(B, 0.0, self.r / self.model.c0, fc)

    def azimuth_sweep(self, theta0, kappa, eps, fc, rel_bw=0.2, n_az=36):
        """Apparent velocity change [%] vs pair azimuth (symmetric branch), plus a
        least-squares fit dv = a0 + a2c cos 2az + a2s sin 2az (the 'fake anisotropy')."""
        az = np.linspace(0, np.pi, n_az, endpoint=False)    # symmetric branch has 180 deg period
        Cref = self.correlate(self.weights(), fc, rel_bw)
        dv = np.array([self.measure(self.correlate(self.weights(theta0, kappa, eps, a),
                                                   fc, rel_bw), Cref, fc)["dv_sym"] for a in az])
        G = np.c_[np.ones_like(az), np.cos(2 * az), np.sin(2 * az)]
        a0, ac, as_ = np.linalg.lstsq(G, dv, rcond=None)[0]
        return az, dv, {"mean": a0, "A2": np.hypot(ac, as_),
                        "fast_az": (0.5 * np.arctan2(as_, ac)) % np.pi}


def measure_bias(lags, C, Cref, fc, t_exp):
    """
    Travel-time bias of correlation C relative to the isotropic reference Cref, picked
    by waveform cross-correlation in windows (+-1.5 periods) around each branch
    (expected arrival +-t_exp). Returns dict: dt_causal, dt_acausal, dt_sym [s],
    dv_sym [%] (= -dt/t), and the causal/acausal energy ratio asym_db [dB].
    """
    half = 1.5 / fc
    sym = lambda y: 0.5 * (y + np.roll(y[::-1], 1))   # C(t) + C(-t), lag grid is symmetric
    out = {}
    for name, sgn, a, b in [("causal", 1, C, Cref), ("acausal", -1, C, Cref),
                            ("sym", 1, sym(C), sym(Cref))]:
        out["dt_" + name] = sgn * waveform_shift(a, b, lags, sgn * t_exp, half)
    out["dv_sym"] = -100.0 * out["dt_sym"] / t_exp
    win = lambda s: np.abs(lags - s * t_exp) < half
    out["asym_db"] = 10 * np.log10(np.sum(C[win(1)] ** 2) / np.sum(C[win(-1)] ** 2))
    return out


def weaver_bias(B, pair_az, t, fc, h=1e-3):
    """
    Stationary-phase bias (Weaver, Froment & Campillo 2009) for a pair pointing along
    azimuth pair_az (A -> B), inter-station time t [s] and noise intensity B(theta):
        dt = -B''(theta_e) / (2 w^2 t B(theta_e)),
    theta_e = endfire direction feeding the branch: pair_az + 180 deg for the causal
    (A -> B) branch, pair_az for the acausal one. The symmetric-branch value is the
    B-weighted mean of the two, because C(t) + C(-t) is dominated by the stronger branch.
    """
    w = 2 * np.pi * fc
    dts, amps = [], []
    for th in (pair_az + np.pi, pair_az):
        B2 = (B(th + h) - 2 * B(th) + B(th - h)) / h**2   # finite-difference B''
        dts.append(-B2 / (2 * w**2 * t * B(th)))
        amps.append(B(th))
    dt_sym = np.average(dts, weights=amps, axis=0)
    return {"dt_causal": dts[0], "dt_acausal": dts[1], "dt_sym": dt_sym,
            "dv_sym": -100 * dt_sym / t}


def waveform_shift(a, b, lags, center, half):
    """Time shift [s] of a relative to b (positive = a later) inside a Hann window of
    half-width `half` around lag `center`; parabolic interpolation gives sub-sample precision."""
    dt = lags[1] - lags[0]
    m = np.abs(lags - center) < half
    taper = np.hanning(m.sum())
    sa, sb = a[m] * taper, b[m] * taper
    cc = np.correlate(sa, sb, "full")
    k = int(np.argmax(cc))
    if 0 < k < cc.size - 1:     # parabola through the peak and its neighbours
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (k - (sb.size - 1)) * dt


def hankel_correlation(exp, w, fc=F0, rel_bw=0.2):
    """Semi-analytic reference for PairExperiment.correlate in a homogeneous medium:
    the same weighted source sum, but with exact Hankel-function Green's functions."""
    c = exp.model.c0
    rA = np.linalg.norm(exp.src_xy - exp.xA, axis=1)
    rB = np.linalg.norm(exp.src_xy - exp.xB, axis=1)
    f = exp.f[:, None]
    Chat = (np.conj(greens_hat(rA[None], f, c)) * greens_hat(rB[None], f, c)) @ w
    band = np.exp(-0.5 * ((exp.f - fc) / (rel_bw * fc)) ** 2)
    white = exp.W2 / (exp.W2 + 1e-3 * exp.W2.max())   # same band as the FD version
    return np.fft.fftshift(np.fft.irfft(Chat * band * white, exp.nfft))


# ----------------------------------------------------------------------------
# 2D array: beamforming estimate of B(theta) and bias correction
# ----------------------------------------------------------------------------
def hex_array(radii=(0.5, 1.0, 2.0, 4.0), n=6):
    """Centre station plus rings of n stations at the given radii [km]; alternate rings
    are rotated by 180/n deg so that inter-station azimuths are well spread."""
    xy = [np.zeros(2)]
    for k, r in enumerate(radii):
        a = 2 * np.pi * np.arange(n) / n + (k % 2) * np.pi / n
        xy += list(np.c_[r * np.cos(a), r * np.sin(a)])
    return np.array(xy)


class ArrayExperiment:
    """
    A 2D array of N stations inside the noise-source ring. One FD impulse run per
    station (reciprocity) gives every station-source Green's function, U[i, f, s].
    From these we form, for any source pattern B(theta):
      * the expected correlation of every station pair (interferometry), and
      * the cross-spectral matrix K_ij(f) = sum_s w_s conj(U_is) U_js, which is what
        a beamformer measures from the raw noise records.
    """

    def __init__(self, model, stations, R=RING_R, f0=F0, ring_spacing=0.15, progress=None):
        self.model, self.f0 = model, f0
        iz, ix = model.node(stations)
        self.xy = np.c_[model.x[ix], model.x[iz]]                # stations snapped to nodes
        self.src_xy, self.theta = ring(model, R, ring_spacing)
        self.dtheta = 2 * np.pi / self.theta.size
        N = len(self.xy)
        T = (R + np.abs(self.xy).max() * np.sqrt(2)) / model.c.min() + 6.0 / f0
        U = []
        for i, x in enumerate(self.xy):
            p = (lambda q, i=i: progress((i + q) / N)) if progress else None
            self.t, u, _ = simulate(model, x, self.src_xy, f0, T, progress=p)
            U.append(np.fft.rfft(u, 2 * self.t.size, axis=0).astype(np.complex64))
        self.U = np.array(U)                                     # (N, nf, ns)
        self.nfft = 2 * self.t.size
        self.f = np.fft.rfftfreq(self.nfft, model.dt)
        self.W2 = np.abs(np.fft.rfft(ricker(self.t, f0, 1.5 / f0), self.nfft)) ** 2
        self.lags = model.dt * (np.arange(self.nfft) - self.nfft // 2)
        # all station pairs i < j: distance and azimuth of the i -> j direction
        self.pi, self.pj = np.triu_indices(N, 1)
        d = self.xy[self.pj] - self.xy[self.pi]
        self.r, self.az = np.hypot(*d.T), np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)

    def weights(self, theta0=0.0, kappa=0.0, eps=0.0):
        return source_density(self.theta, theta0, kappa, eps) * self.dtheta

    def _filter(self, fc, rel_bw):
        """Whitening x Gaussian band-pass, applied to cross-spectra (divides out |W|^2)."""
        band = np.exp(-0.5 * ((self.f - fc) / (rel_bw * fc)) ** 2)
        return band / (self.W2 + 1e-3 * self.W2.max())

    # -- interferometry ----------------------------------------------------------
    def correlate(self, i, j, w, fc=F0, rel_bw=0.2):
        Chat = (np.conj(self.U[i]) * self.U[j]) @ w * self._filter(fc, rel_bw)
        return np.fft.fftshift(np.fft.irfft(Chat, self.nfft))

    def pair_bias(self, w, fc=F0, rel_bw=0.2, rmin=None):
        """Symmetric-branch travel-time bias [s] of every pair with r >= rmin
        (default 2 wavelengths), relative to the isotropic correlation of that pair.
        Returns pair indices and the bias."""
        rmin = 2 * self.model.c0 / fc if rmin is None else rmin
        idx = np.flatnonzero(self.r >= rmin)
        w0 = self.weights()
        dt = [measure_bias(self.lags, self.correlate(self.pi[k], self.pj[k], w, fc, rel_bw),
                           self.correlate(self.pi[k], self.pj[k], w0, fc, rel_bw), fc,
                           self.r[k] / self.model.c0)["dt_sym"] for k in idx]
        return idx, np.array(dt)

    def subset(self, keep):
        """Same experiment restricted to the stations `keep` (e.g. a smaller sub-array)."""
        sub = copy.copy(self)
        sub.xy, sub.U = self.xy[keep], self.U[keep]
        sub.pi, sub.pj = np.triu_indices(len(sub.xy), 1)
        d = sub.xy[sub.pj] - sub.xy[sub.pi]
        sub.r, sub.az = np.hypot(*d.T), np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
        return sub

    # -- beamforming ---------------------------------------------------------------
    def _band(self, fc, rel_bw):
        g = self._filter(fc, rel_bw)
        sel = g > 1e-3 * g.max()
        return self.f[sel], g[sel], sel

    def csdm(self, w, fc=F0, rel_bw=0.2, n_windows=None, seed=0):
        """Whitened cross-spectral matrix K_ij(f) = <d_i conj(d_j)> of the noise records d
        in the band around fc. n_windows=None gives the exact expectation; an integer
        averages that many independent noise windows (random source spectra with power
        w_s), which is what a finite recording provides. Shape (nfb, N, N)."""
        f, g, sel = self._band(fc, rel_bw)
        Us = self.U[:, sel] * np.sqrt(g)[None, :, None]          # (N, nfb, ns)
        if n_windows is None:
            return np.einsum("ifs,s,jfs->fij", Us, w, np.conj(Us))
        rng = np.random.default_rng(seed)
        K = 0
        for _ in range(n_windows):
            a = np.sqrt(w / 2) * (rng.standard_normal(w.size) + 1j * rng.standard_normal(w.size))
            d = np.einsum("ifs,s->fi", Us, a)                    # (nfb, N) one window
            K = K + d[:, :, None] * np.conj(d[:, None, :])
        return K / n_windows

    def _steering(self, f, theta, slow):
        """Plane-wave phase factors e_j = exp(-i w tau_j) for waves arriving FROM azimuth
        theta with slowness slow [s/km]: tau_j = -slow * (x_j . n(theta)).
        Shape (nf, ntheta, N)."""
        n = np.c_[np.cos(theta), np.sin(theta)]
        tau = -slow * (n @ self.xy.T)                            # (ntheta, N)
        return np.exp(-2j * np.pi * f[:, None, None] * tau[None])

    def beam_scan(self, K, fc=F0, rel_bw=0.2, thetas=None, slows=None):
        """Bartlett beam power P(slowness, theta) = sum_f conj(e)^T K e, summed over the
        band, for a cross-spectral matrix K from csdm(). Returns thetas, slows, P."""
        thetas = np.radians(np.arange(0, 360, 2.0)) if thetas is None else thetas
        slows = np.linspace(0.2, 0.8, 61) if slows is None else slows
        f, _, _ = self._band(fc, rel_bw)
        P = np.zeros((slows.size, thetas.size))
        for k, s in enumerate(slows):
            E = self._steering(f, thetas, s)
            P[k] = np.einsum("fpi,fij,fpj->p", np.conj(E), K, E).real
        return thetas, slows, P

    def beam_matrix(self, fc, slow, thetas, rel_bw=0.2):
        """Modelled array response M[theta, s]: beam power at azimuth theta produced by
        ring source s alone, so P(theta) ~ M @ w. Built from Hankel-function Green's
        functions in a homogeneous medium of speed c0 (what an analyst would assume),
        not from the FD wavefield, so the inversion does not use the true medium."""
        f, g, sel = self._band(fc, rel_bw)
        g = g * self.W2[sel]                       # band weights only: G carries no wavelet
        dist = np.linalg.norm(self.xy[:, None] - self.src_xy[None], axis=2)  # (N, ns)
        G = greens_hat(dist[:, None, :], f[None, :, None], self.model.c0)    # (N, nfb, ns)
        E = self._steering(f, thetas, slow)                                  # (nfb, ntheta, N)
        b = np.einsum("fpj,jfs->fps", np.conj(E), G)
        return np.einsum("f,fps->ps", g, np.abs(b) ** 2)

    def estimate_B(self, P, M, n_harm=8, damping=1e-3):
        """
        Invert beam power P(theta) = M @ (B(theta_s) dtheta) for the noise intensity B,
        written as a truncated Fourier series B = a0/2pi + sum_n (a_n cos n th + b_n sin n th)/pi.
        Damped linear least squares (penalty ~ n^2 on each harmonic) as in the course
        regression notebook. Returns a callable B_est(theta), normalised to integral 1.
        """
        n = np.arange(1, n_harm + 1)
        F = np.c_[np.ones_like(self.theta) / (2 * np.pi),
                  np.cos(np.outer(self.theta, n)) / np.pi, np.sin(np.outer(self.theta, n)) / np.pi]
        A = M @ F * self.dtheta
        scale = np.abs(A).max()
        L = np.diag(np.r_[0, n**2, n**2]) * damping * scale     # roughness penalty
        coef = np.linalg.lstsq(np.vstack([A, L]), np.r_[P, np.zeros(L.shape[0])], rcond=None)[0]
        coef /= coef[0]                                          # integral of B = 1
        a, b = coef[1:n_harm + 1], coef[n_harm + 1:]

        def B_est(th):
            th = np.asarray(th, float)[..., None]
            return 1 / (2 * np.pi) + (np.cos(n * th) @ a + np.sin(n * th) @ b) / np.pi
        return B_est

    def model_bias(self, B, fc=F0, rel_bw=0.2, idx=None):
        """Predicted symmetric-branch bias of each pair for a noise intensity B (callable),
        computed in an assumed homogeneous medium (speed c0) with exact Hankel-function
        Green's functions: the 'model-based' correction. Negative B is clipped to 0."""
        c = self.model.c0
        g = self._filter(fc, rel_bw) * self.W2                  # band only (no |W|^2 here)
        sel = g > 1e-3 * g.max()
        f = self.f[sel]
        dist = np.linalg.norm(self.xy[:, None] - self.src_xy[None], axis=2)   # (N, ns)
        G = greens_hat(dist[:, None, :], f[None, :, None], c)                 # (N, nfb, ns)
        w = np.clip(B(self.theta), 0, None) * self.dtheta
        w0 = self.weights()
        idx = np.arange(self.r.size) if idx is None else idx

        def corr(X, ww):                      # band-limited correlation from cross-spectra
            Chat = np.zeros(self.f.size, complex)
            Chat[sel] = X @ ww
            return np.fft.fftshift(np.fft.irfft(Chat, self.nfft))

        out = []
        for k in idx:
            X = np.conj(G[self.pi[k]]) * G[self.pj[k]] * g[sel][:, None]
            out.append(measure_bias(self.lags, corr(X, w), corr(X, w0), fc,
                                    self.r[k] / c)["dt_sym"])
        return np.array(out)


def fit_anisotropy(az, dv):
    """Least-squares fit dv = a0 + A2 cos 2(az - psi): returns a0, A2, fast azimuth psi."""
    G = np.c_[np.ones_like(az), np.cos(2 * az), np.sin(2 * az)]
    a0, ac, as_ = np.linalg.lstsq(G, dv, rcond=None)[0]
    return a0, np.hypot(ac, as_), (0.5 * np.arctan2(as_, ac)) % np.pi


# ----------------------------------------------------------------------------
# DAS fiber: dense line of strain channels
# ----------------------------------------------------------------------------
def cable_nodes(model, legs):
    """Grid nodes along a cable made of straight legs parallel to the grid axes.
    legs: list of corner points [(x0, y0), (x1, y1), ...] [km]. Returns (n, 2) node
    coordinates ordered along the fiber (node spacing = dx = channel spacing)."""
    pts = [np.asarray(legs[0], float)]
    for a, b in zip(legs[:-1], legs[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = int(round(np.abs(b - a).max() / model.dx))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    iz, ix = model.node(np.array(pts))
    return np.c_[model.x[ix], model.x[iz]]


def road(p0, p1, spacing=0.15):
    """Traffic noise sources every `spacing` km along a straight road from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = int(np.ceil(np.linalg.norm(p1 - p0) / spacing)) + 1
    return p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]


_WORKER = {}


def _worker_init(model, rec, f0, T, nfk):
    _WORKER.update(model=model, rec=rec, f0=f0, T=T, nfk=nfk)


def _node_spectra(xy):
    """One impulse run at node xy; returns the record spectra up to the kept frequency."""
    w = _WORKER
    t, tr, _ = simulate(w["model"], xy, w["rec"], w["f0"], w["T"])
    return np.fft.rfft(tr, 2 * t.size, axis=0)[:w["nfk"]].astype(np.complex64)


def strain_matrix(n, h, gauge=0.2, ok=None):
    """
    Matrix D (n x n) turning displacement-potential records u at n equally spaced fiber
    nodes (spacing h) into DAS channels: axial strain averaged over gauge length L,
        eps(s) = [v(s + L/2) - v(s - L/2)] / L,   v = du/ds,
    with du/ds from the 4th-order centred stencil (1, -8, 0, 8, -1)/(12 h) of course
    module 02. Plane-wave response: -(k cos th)^2 sinc(k L cos th / 2), plus stencil error.
    Rows too close to the cable ends (or where ok is False) are zero: no channel there.
    """
    m = max(int(round(gauge / (2 * h))), 1)                      # half gauge in nodes
    Dv = np.zeros((n, n))
    for k in range(2, n - 2):
        Dv[k, k - 2:k + 3] = np.array([1, -8, 0, 8, -1]) / (12 * h)
    D = np.zeros((n, n))
    for k in range(m + 2, n - m - 2):
        if ok is None or ok[k]:
            D[k] = (Dv[k + m] - Dv[k - m]) / (2 * m * h)
    return D


def das_bias_prediction(offset, fc, c=C0, gauge=0.0):
    """Bias of DAS (strain) vs geophone travel times for isotropic noise, from the
    stationary-phase formula dt/t = -B''/(2 B (w t)^2) with the DAS sensitivity
    B(theta) ~ cos^4(theta) sinc^2(a cos theta), a = k L / 2 (gauge length L):
        B''/B at endfire = -4 + 2 (1 - a cot a)      (= -4 for a point gauge),
    so DAS arrivals are late (the cable looks slow). Returns dv/v [%] = -dt/t."""
    wt = 2 * np.pi * fc * np.asarray(offset) / c
    a = np.pi * fc * gauge / c
    curv = -4.0 + (2 * (1 - a / np.tan(a)) if a > 0 else 0.0)
    return 100.0 * curv / (2 * wt**2)


class FiberExperiment:
    """
    A fiber-optic cable sampled at every grid node along its path. By reciprocity, one
    FD impulse run per node gives the Green's function from every noise source to
    every node (displacement potential u, i.e. what a geophone would record).
    DAS channels are then linear combinations of node records (strain operator).

    sources: dict name -> (ns, 2) source coordinates, e.g. {"ring": ..., "road": ...}.
    Only frequencies up to fmax are kept, which keeps memory small.
    """

    def __init__(self, model, nodes_xy, sources, f0=F0, fmax=4.0, n_jobs=None, progress=None):
        from concurrent.futures import ProcessPoolExecutor
        import os
        self.model, self.f0, self.xy = model, f0, np.asarray(nodes_xy)
        names = list(sources)
        allsrc = np.vstack([sources[k] for k in names])
        iz, ix = model.node(allsrc)
        self.src_xy = np.c_[model.x[ix], model.x[iz]]
        bounds = np.cumsum([0] + [len(sources[k]) for k in names])
        self.groups = {k: slice(bounds[i], bounds[i + 1]) for i, k in enumerate(names)}
        dmax = np.linalg.norm(self.src_xy[:, None] - self.xy[None], axis=2).max()
        T = dmax / model.c.min() + 6.0 / f0
        nt = int(np.ceil(T / model.dt)) + 1
        self.nfft = 2 * nt
        f = np.fft.rfftfreq(self.nfft, model.dt)
        nfk = int(np.searchsorted(f, fmax)) + 1
        self.f = f[:nfk]
        t = model.dt * np.arange(nt)
        self.W2 = (np.abs(np.fft.rfft(ricker(t, f0, 1.5 / f0), self.nfft)) ** 2)[:nfk]
        self.lags = model.dt * (np.arange(self.nfft) - self.nfft // 2)

        n_jobs = n_jobs or os.cpu_count() or 1
        args = (model, self.src_xy, f0, T, nfk)
        U = []
        if n_jobs > 1:      # nodes are independent: run them in parallel processes
            try:
                with ProcessPoolExecutor(n_jobs, initializer=_worker_init, initargs=args) as ex:
                    for k, spec in enumerate(ex.map(_node_spectra, list(self.xy))):
                        U.append(spec)
                        if progress: progress((k + 1) / len(self.xy))
            except Exception as err:   # e.g. no multiprocessing available: fall back to serial
                print(f"parallel run failed ({err!r}); running serially")
                U = []
        if not U:
            _worker_init(*args)
            for k, xy in enumerate(self.xy):
                U.append(_node_spectra(xy))
                if progress: progress((k + 1) / len(self.xy))
        self.U = np.array(U)                                     # (n_nodes, nfk, ns)
        # unit tangent of the fiber at each node (for the DAS directional response)
        tg = np.gradient(self.xy, axis=0)
        self.tangent = tg / np.linalg.norm(tg, axis=1, keepdims=True)

    # -- channels --------------------------------------------------------------
    def strain_operator(self, gauge=0.2):
        """DAS operator for this cable (see strain_matrix); channels whose stencil wraps
        round a corner of the cable are left out (zero rows)."""
        n, h = len(self.xy), self.model.dx
        m = max(int(round(gauge / (2 * h))), 1)
        ok = [np.allclose(self.tangent[max(k - m - 2, 0):k + m + 3], self.tangent[k]) for k in range(n)]
        return strain_matrix(n, h, gauge, np.array(ok))

    def das_channels(self, gauge=0.2):
        """Indices of nodes that carry a valid DAS channel for this gauge length."""
        return np.flatnonzero(np.abs(self.strain_operator(gauge)).sum(1) > 0)

    def channel_spectra(self, kind="das", gauge=0.2):
        """Spectra (n_nodes, nf, ns) of geophone ('geo') or DAS ('das') channels.
        The strain operator is sparse (a few nodes per channel), so it is applied as a
        sparse matrix, and the result is cached per gauge length."""
        if kind == "geo":
            return self.U
        cache = self.__dict__.setdefault("_das_cache", {})
        if gauge not in cache:
            from scipy.sparse import csr_matrix
            D = csr_matrix(self.strain_operator(gauge))
            n = len(self.xy)
            cache.clear()                                    # keep memory to one gauge at a time
            cache[gauge] = (D @ self.U.reshape(n, -1)).reshape(self.U.shape).astype(np.complex64)
        return cache[gauge]

    def weights(self, group, density=None):
        """Source weights: zero outside `group`; inside it uniform, or `density` (array)."""
        w = np.zeros(len(self.src_xy))
        sl = self.groups[group]
        w[sl] = 1.0 if density is None else density
        return w / w.sum()

    def _filter(self, fc, rel_bw):
        band = np.exp(-0.5 * ((self.f - fc) / (rel_bw * fc)) ** 2)
        return band / (self.W2 + 1e-3 * self.W2.max())

    def correlate(self, Ua, Ub, w, fc=F0, rel_bw=0.2, k4=False):
        """Expected correlation of two channels (spectra (nf, ns)) for source weights w.
        k4=True divides by (w/c0)^4, the endfire amplitude of a strain-strain correlation,
        so DAS correlations can be compared waveform-to-waveform with geophone ones."""
        Chat = np.zeros(self.nfft // 2 + 1, complex)
        g = self._filter(fc, rel_bw)
        if k4:   # only inside the band: 1/k^4 would blow up the (filtered-out) low frequencies
            k = 2 * np.pi * self.f / self.model.c0
            inband = np.abs(self.f - fc) < 4 * rel_bw * fc
            g = np.where(inband, g / np.maximum(k, 1e-12) ** 4, 0.0)
        Chat[:self.f.size] = (np.conj(Ua) * Ub) @ w * g
        return np.fft.fftshift(np.fft.irfft(Chat, self.nfft))

    def gather(self, w, kind="das", gauge=0.2, src=None, chans=None, fc=F0, rel_bw=0.2):
        """Virtual shot gather: correlations of virtual-source channel `src` with `chans`."""
        Uc = self.channel_spectra(kind, gauge)
        return np.array([self.correlate(Uc[src], Uc[c], w, fc, rel_bw, k4=(kind == "das"))
                         for c in chans])

    def offset_bias(self, w, kind="das", gauge=0.2, src=None, chans=None, fc=F0, rel_bw=0.2,
                    w_ref=None):
        """Symmetric-branch velocity error [%] of each channel pair (src, c) relative to the
        GEOPHONE correlation for isotropic noise w_ref (the displacement Green's function
        reference). Returns offsets [km] and dv [%]."""
        G = self.gather(w, kind, gauge, src, chans, fc, rel_bw)
        R = self.gather(w_ref, "geo", gauge, src, chans, fc, rel_bw)
        off = np.linalg.norm(self.xy[chans] - self.xy[src], axis=1)
        dv = [measure_bias(self.lags, g, r, fc, o / self.model.c0)["dv_sym"] for g, r, o in zip(G, R, off)]
        return off, np.array(dv)

    # -- slant stack / beamforming along the cable ----------------------------------
    def beam(self, w, chans, kind="das", gauge=0.2, fc=F0, rel_bw=0.2, thetas=None, slows=None):
        """Bartlett beam power P(slowness, theta) from the channels `chans`, with steering
        vectors that include each channel's directional response ((t . n)^2 for DAS)."""
        thetas = np.radians(np.arange(0, 360, 2.0)) if thetas is None else thetas
        slows = np.linspace(0.1, 0.8, 71) if slows is None else slows
        Uc = self.channel_spectra(kind, gauge)[chans]
        g = self._filter(fc, rel_bw)
        sel = g > 1e-3 * g.max()
        Us = Uc[:, sel] * np.sqrt(g[sel])[None, :, None]
        K = np.einsum("ifs,s,jfs->fij", Us, w, np.conj(Us))
        n = np.c_[np.cos(thetas), np.sin(thetas)]
        amp = (n @ self.tangent[chans].T) ** 2 if kind == "das" else np.ones((thetas.size, len(chans)))
        P = np.zeros((slows.size, thetas.size))
        for k, s in enumerate(slows):
            E = amp[None] * np.exp(2j * np.pi * self.f[sel][:, None, None] * s * (n @ self.xy[chans].T)[None])
            P[k] = np.einsum("fpi,fij,fpj->p", np.conj(E), K, E).real
        # normalised Bartlett: divide by |e|^2 so directions where the channels are more
        # sensitive (large (t.n)^2) are not favoured just for that reason
        return thetas, slows, P / np.maximum((amp ** 2).sum(1), 1e-12)[None, :]


def das_layout(model, ring_spacing=0.15):
    """Default DAS survey: an L-shaped cable with two 6 km legs, one along x from the
    corner (-3, 0) to (3, 0) and one along y from the corner up to (-3, 6); three straight
    roads with traffic noise; and the far-field noise ring. Returns nodes (ordered from
    the far end of the y-leg, through the corner, to the far end of the x-leg), the
    sources dict, and slices of node indices for the two legs."""
    nodes = cable_nodes(model, [(-3.0, 6.0), (-3.0, 0.0), (3.0, 0.0)])
    corner = int(np.argmin(np.linalg.norm(nodes - [-3.0, 0.0], axis=1)))
    ring_xy, _ = ring(model, RING_R, ring_spacing)
    sources = {
        "ring": ring_xy,
        "road parallel": road((-11.0, -1.5), (11.0, -1.5)),       # 1.5 km south of the x-leg
        "road perpendicular": road((6.0, -11.0), (6.0, 11.0)),    # 3 km beyond the end of the x-leg
        "road oblique": road((-4.0, -10.0), (12.0, 6.0)),         # 45 deg, through (6, 0)
    }
    legs = {"y-leg": slice(0, corner + 1), "x-leg": slice(corner, len(nodes))}
    return nodes, sources, legs


def leg_slowness(F, w, chans, axis, kind="das", gauge=0.2, fc=F0, rel_bw=0.2):
    """Apparent slowness [s/km] along one straight cable leg (slant stack over the
    along-leg slowness p; axis = leg direction in rad). A wave from back-azimuth theta
    gives p = cos(theta - axis) / c, so one leg cannot separate theta from c."""
    p = np.linspace(-0.8, 0.8, 641)
    _, _, P = F.beam(w, chans, kind, gauge, fc, rel_bw, thetas=np.array([axis]), slows=p)
    return p[P[:, 0].argmax()]


def l_cable_velocity(px, py):
    """Combine the apparent slownesses of two perpendicular legs (x and y): the wave
    slowness vector is (px, py), so c = 1/|p| and the back-azimuth is atan2(py, px)."""
    return 1 / np.hypot(px, py), np.arctan2(py, px) % (2 * np.pi)
