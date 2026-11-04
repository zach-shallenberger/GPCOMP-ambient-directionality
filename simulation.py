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
        """
        Travel-time bias of C relative to the isotropic reference Cref, picked by
        waveform cross-correlation in windows (+-1.5 periods) around each branch.
        Returns dict: dt_causal, dt_acausal, dt_sym [s], dv_sym [%] (= -dt/t),
        and the causal/acausal energy ratio [dB].
        """
        t_exp = self.r / self.model.c0 if t_exp is None else t_exp
        half = 1.5 / fc
        sym = lambda y: 0.5 * (y + np.roll(y[::-1], 1))   # C(t) + C(-t), lag grid is symmetric
        out = {}
        for name, sgn, a, b in [("causal", 1, C, Cref), ("acausal", -1, C, Cref),
                                ("sym", 1, sym(C), sym(Cref))]:
            out["dt_" + name] = sgn * waveform_shift(a, b, self.lags, sgn * t_exp, half)
        out["dv_sym"] = -100.0 * out["dt_sym"] / t_exp
        win = lambda s: np.abs(self.lags - s * t_exp) < half
        out["asym_db"] = 10 * np.log10(np.sum(C[win(1)] ** 2) / np.sum(C[win(-1)] ** 2))
        return out

    def weaver_prediction(self, theta0, kappa, eps, fc, pair_az=0.0):
        """Stationary-phase bias (Weaver, Froment & Campillo 2009) for each branch,
            dt = -B''(theta_e) / (2 w^2 t B(theta_e)),
        theta_e = endfire direction feeding that branch (180 deg causal, 0 deg acausal).
        The symmetric-branch value is the amplitude (B)-weighted mean of the two, because
        C(t) + C(-t) is dominated by the stronger branch."""
        t, w, h = self.r / self.model.c0, 2 * np.pi * fc, 1e-3
        B = lambda th: source_density(th, theta0 - pair_az, kappa, eps)
        dts, amps = [], []
        for th in (np.pi, 0.0):
            B2 = (B(th + h) - 2 * B(th) + B(th - h)) / h**2   # finite-difference B''
            dts.append(-B2 / (2 * w**2 * t * B(th)))
            amps.append(B(th))
        dt_sym = np.average(dts, weights=amps)
        return {"dt_causal": dts[0], "dt_acausal": dts[1], "dt_sym": dt_sym,
                "dv_sym": -100 * dt_sym / t}

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
