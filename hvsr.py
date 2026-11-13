"""
HVSR-style site response of a dipping soft layer, from simulated ambient noise.

Model: a 2D depth slice (x, z) with a soft layer over rock. The interface dips, so the
layer thickness H(x) changes along the profile. We solve the SH (antiplane) wave equation

    rho u_tt = d/dx(mu du/dx) + d/dz(mu du/dz) + rho f,     mu = rho Vs^2,   mu du/dz = 0 at z = 0

with a conservative 2nd-order leapfrog scheme (harmonic mu
on cell faces, mirror free surface, quadratic sponges), but driven by MANY simultaneous
sources, each emitting its own band-limited random noise. Sources sit in the rock below the
line (body waves arriving at all incidence angles) and near the surface beyond both ends
(surface waves arriving from the sides).

u is the horizontal (out-of-plane) displacement, so the surface records are the horizontal
component. A scalar/SH model has no vertical component, so instead of a true H/V ratio we
use the standard site-to-reference spectral ratio: the surface spectrum over the soft layer
divided by the surface spectrum of the same noise on bare rock. For a flat layer and
vertical incidence it equals the 1D SH transfer function, whose first peak is at
f0 = Vs1 / (4 H), with height ~ the impedance contrast. Field HVSR is used as a proxy for
the same f0.
"""
from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------------------
# grid and small helpers
# --------------------------------------------------------------------------------------
@dataclass
class Grid:
    """x-z grid. z = 0 is row 0 (free surface unless free_surface=False); x is centred on 0.
    Sponges are added *outside* the physical region [-half_width, half_width] x [0, depth]."""
    dx: float = 1.0               # grid spacing [m] (same in x and z)
    half_width: float = 450.0     # physical half-width [m]
    depth: float = 200.0          # physical depth [m]
    side: float = 250.0           # side sponge width [m] (low-frequency waves need long sponges)
    bottom: float = 100.0         # bottom sponge width [m]
    cfl: float = 0.6              # beta_max dt / dx  (limit 1/sqrt 2)
    free_surface: bool = True     # False: sponge on top as well (whole-space tests)
    periodic_x: bool = False      # True: no side sponges, periodic in x (1D plane-wave tests)

    def __post_init__(self):
        self.ns = 0 if self.periodic_x else int(round(self.side / self.dx))
        self.nb = int(round(self.bottom / self.dx))
        self.nt_top = 0 if self.free_surface else self.nb
        self.nx = 2 * int(round(self.half_width / self.dx)) + 1 + 2 * self.ns
        self.nz = int(round(self.depth / self.dx)) + 1 + self.nb + self.nt_top
        self.x = self.dx * (np.arange(self.nx) - (self.nx - 1) // 2)
        self.z = self.dx * (np.arange(self.nz) - self.nt_top)

    def ix(self, x):
        return np.rint(np.asarray(x) / self.dx).astype(int) + (self.nx - 1) // 2

    def iz(self, z):
        return np.rint(np.asarray(z) / self.dx).astype(int) + self.nt_top

    def sponge(self, vmax):
        """Quadratic damping sigma(x, z) [1/s], zero in the physical region."""
        i = np.arange(self.nx)
        dxs = np.maximum(np.maximum(self.ns - i, i - (self.nx - 1 - self.ns)), 0) / max(self.ns, 1)
        sx = dxs**2 * 20 * vmax / max(self.side, 1e-9)
        dz = np.arange(self.nz)
        bot = np.maximum(dz - (self.nz - 1 - self.nb), 0) / self.nb
        top = np.maximum(self.nt_top - dz, 0) / max(self.nt_top, 1)
        sz = (bot**2 + top**2) * 20 * vmax / self.bottom
        return np.maximum(sx[None, :], sz[:, None])


def harmonic(a, b):
    """Harmonic mean, the correct average of mu across a half-point (series springs)."""
    return 2 * a * b / (a + b)


def vs30(b1, H, b2):
    """Time-averaged shear speed over the top 30 m: 30 / sum(h_i / beta_i)."""
    h1 = min(H, 30.0)
    return 30.0 / (h1 / b1 + (30.0 - h1) / b2)


def site_class(v):
    """NEHRP site class from Vs30 [m/s]."""
    for lim, c in [(1500, "A (hard rock)"), (760, "B (rock)"), (360, "C (very dense soil)"),
                   (180, "D (stiff soil)")]:
        if v > lim:
            return c
    return "E (soft soil)"


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
@dataclass
class Model:
    vs1: float = 250.0     # soft layer shear speed [m/s]
    rho1: float = 1800.0   # soft layer density [kg/m^3]
    vs2: float = 1000.0    # rock shear speed [m/s]
    rho2: float = 2300.0   # rock density
    h0: float = 30.0       # layer thickness at x = 0 [m]
    dip: float = 8.0       # interface dip [deg]; positive: the layer thickens towards +x
    h_min: float = 8.0     # the interface never comes closer to the surface than this
    q1: float = 20.0       # soft-layer quality factor at the local resonance (0 = no damping)

    def thickness(self, x):
        return np.maximum(self.h0 + np.tan(np.radians(self.dip)) * np.asarray(x, float), self.h_min)

    @property
    def contrast(self):
        """Impedance contrast rho2 Vs2 / (rho1 Vs1) (> 1 for soft over hard)."""
        return self.rho2 * self.vs2 / (self.rho1 * self.vs1)

    def f0_1d(self, x):
        """Local 1D fundamental frequency Vs1 / (4 H(x))."""
        return self.vs1 / (4.0 * self.thickness(x))

    def damping(self, x=0.0):
        """Viscous damping rate sigma(x) [1/s] of the layer (rho sigma u_t term). A viscous
        term gives Q = omega / sigma, so we set sigma from the LOCAL resonance,
        sigma = 2 pi f0(x) / q1: every column then has Q = q1 at its own f0 (rock is elastic)."""
        return 2 * np.pi * self.f0_1d(x) / self.q1 if self.q1 else 0.0 * np.asarray(x, float)

    def transfer_1d(self, f, H=None, damped=True):
        """|surface / rock-outcrop| for a vertically incident SH plane wave on a flat
        layer of thickness H over a half-space. With damping, the layer has the same
        viscous term as the FD model: k1 = (w / Vs1) sqrt(1 - i sigma / w)."""
        H = self.h0 if H is None else H
        w = 2 * np.pi * np.maximum(np.asarray(f, float), 1e-9)
        sig = self.damping(0.0) if damped else 0.0       # flat layer: H = h0
        k1 = w / self.vs1 * np.sqrt(1 - 1j * sig / w)
        mu1k = self.rho1 * self.vs1 ** 2 * k1                      # mu1 * k1
        mu2k = self.rho2 * self.vs2 * w                             # mu2 * k2
        return np.abs(1.0 / (np.cos(k1 * H) + 1j * (mu1k / mu2k) * np.sin(k1 * H)))

    def arrays(self, grid, homogeneous=False):
        """Node arrays beta, rho and damping rate sigma on the grid. Each node owns the cell [z - dx/2, z + dx/2]
        ([0, dx/2] on the surface row); a cell cut by the interface gets the series
        (harmonic) average of mu and the arithmetic average of rho, so the thickness
        is not rounded to the grid."""
        g = grid
        rho = np.full((g.nz, g.nx), self.rho2)
        mu = np.full((g.nz, g.nx), self.rho2 * self.vs2 ** 2)
        if homogeneous:
            return np.sqrt(mu / rho), rho, np.zeros((g.nz, g.nx))
        H = self.thickness(g.x)[None, :]
        z = g.z[:, None]
        top = np.maximum(z - g.dx / 2, 0.0)
        bot = z + g.dx / 2
        frac = np.clip((H - top) / (bot - top), 0.0, 1.0)            # fraction of cell in the layer
        mu1, mu2 = self.rho1 * self.vs1 ** 2, self.rho2 * self.vs2 ** 2
        mu = 1.0 / (frac / mu1 + (1 - frac) / mu2)
        rho = frac * self.rho1 + (1 - frac) * self.rho2
        return np.sqrt(mu / rho), rho, frac * self.damping(g.x)[None, :]


def default_grid(dx=2.5):
    """Physical box 600 m x 250 m. The sponges are long (300 m sides, 400 m bottom)
    because a rock wavelength at 2 Hz is 500 m: shorter sponges reflect the low
    frequencies and shift the resonance peaks by up to 10%."""
    return Grid(dx=dx, half_width=300.0, depth=250.0, side=300.0, bottom=400.0, cfl=0.6)


# --------------------------------------------------------------------------------------
# noise sources
# --------------------------------------------------------------------------------------
def noise_sources(n_deep=60, n_surface=0, seed=0, x_span=280.0, z_range=(110.0, 240.0),
                  surface_x=(170.0, 290.0)):
    """Source positions: n_deep scattered in the rock below the line (so plane-ish waves
    arrive from all incidence angles, up to ~70 deg) and n_surface at 2 m depth beyond
    both ends of the receiver line (surface/Love waves from both sides)."""
    rng = np.random.default_rng(seed)
    xd = rng.uniform(-x_span, x_span, n_deep)
    zd = rng.uniform(*z_range, n_deep)
    side = np.where(np.arange(n_surface) % 2 == 0, -1.0, 1.0)
    xs = side * rng.uniform(*surface_x, n_surface)
    return np.r_[xd, xs], np.r_[zd, np.full(n_surface, 2.0)]


def band_noise(nt, dt, n, fmin, fmax, seed, pulse=False):
    """n independent, unit-RMS noise series, flat between fmin and fmax (cosine edges).
    pulse=True instead gives one deterministic zero-phase pulse with the same spectrum,
    centred at t = 2 s (for clean transfer-function tests)."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(nt, dt)
    edge = 0.3
    amp = np.clip(np.minimum((f - fmin) / edge + 0.5, (fmax - f) / edge + 0.5), 0, 1)
    amp = np.sin(0.5 * np.pi * amp) ** 2
    if pulse:
        w = np.fft.irfft(amp * np.exp(-2j * np.pi * f * 2.0), nt)[:, None] * np.ones(n)
        return w / np.abs(w).max()
    spec = amp[:, None] * (rng.standard_normal((len(f), n)) + 1j * rng.standard_normal((len(f), n)))
    w = np.fft.irfft(spec, nt, axis=0)
    return w / w.std(axis=0, keepdims=True)


# --------------------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------------------
def simulate_noise(model, grid=None, T=40.0, rec_x=None, src=None, fband=(0.6, 9.0), seed=1,
                   homogeneous=False, rec_dt=0.01, snapshots=(), on_snapshot=None, progress=None,
                   plane_wave_depth=None, pulse=False):
    """Run the SH solver with all noise sources firing at once.

    rec_x       : surface receiver x positions [m] (default: every 10 m over +-150 m)
    src         : (x, z) arrays of source positions (default noise_sources())
    homogeneous : True -> bare rock (the reference site), same sources and noise
    pulse       : every source emits the same deterministic band-limited pulse (tests)
    plane_wave_depth : instead of point sources, a horizontal line source at this depth
                 (with a periodic grid this is a vertically incident plane wave; used in
                 the verification)
    Returns dict t, u (nt, nrec), rec_x, snaps {t: frame}, x_snap, z_snap, dt_solver.
    """
    g = grid or default_grid()
    rec_x = np.arange(-150.0, 150.1, 10.0) if rec_x is None else np.asarray(rec_x, float)
    beta, rho, sig = model.arrays(g, homogeneous)
    dt = g.cfl * g.dx / beta.max()
    nt = int(np.ceil(T / dt)) + 1
    decim = max(1, int(round(rec_dt / dt)))

    mu = rho * beta ** 2
    mux = harmonic(mu[:, 1:], mu[:, :-1])
    muz = harmonic(mu[1:], mu[:-1])
    if g.periodic_x:
        mux = harmonic(mu, np.roll(mu, -1, axis=1))
    k = dt ** 2 / (rho * g.dx ** 2)
    s = 0.5 * (g.sponge(beta.max()) + sig) * dt
    a_prev, a_den = 1 - s, 1 / (1 + s)

    if plane_wave_depth is not None:
        iz_src, ix_src = np.array([g.iz(plane_wave_depth)]), None
        w = band_noise(nt, dt, 1, *fband, seed, pulse) / g.dx
    else:
        sx, sz = noise_sources() if src is None else src
        iz_src, ix_src = g.iz(sz), g.ix(sx)
        w = band_noise(nt, dt, len(sx), *fband, seed, pulse) / g.dx ** 2

    jr = g.ix(rec_x)
    keep = np.abs(g.x) <= g.half_width + 1e-9
    zkeep = (g.z >= -1e-9) & (g.z <= g.depth + 1e-9)
    sd = max(1, int(round(4.0 / g.dx)))                     # snapshots on a ~4-5 m grid
    snap_steps = {int(round(ts / dt)): round(float(ts), 3) for ts in snapshots}

    u_prev, u = np.zeros((g.nz, g.nx)), np.zeros((g.nz, g.nx))
    Lu = np.zeros_like(u)
    rec = np.zeros(((nt - 1) // decim + 1, len(jr)))
    snaps = {}
    for it in range(nt):
        if progress and it % max(nt // 50, 1) == 0:
            progress(it / nt)
        if it % decim == 0:
            rec[it // decim] = u[0, jr]
        if it in snap_steps:
            frame = u[zkeep][:, keep][::sd, ::sd].astype(np.float32)
            snaps[snap_steps[it]] = frame
            if on_snapshot:
                on_snapshot(snap_steps[it], frame)
        Lu[:] = 0.0
        if g.periodic_x:
            Fx = mux * (np.roll(u, -1, axis=1) - u)
            Lu += Fx - np.roll(Fx, 1, axis=1)
        else:
            Fx = mux * (u[:, 1:] - u[:, :-1])
            Lu[:, 1:-1] += Fx[:, 1:] - Fx[:, :-1]
        Fz = muz * (u[1:] - u[:-1])
        Lu[1:-1] += Fz[1:] - Fz[:-1]
        Lu[0] += 2 * Fz[0]                                    # free surface (mirror)
        u_next = 2 * u - a_prev * u_prev + k * Lu
        if ix_src is None:
            u_next[iz_src[0]] += dt ** 2 * w[it, 0]
        else:
            u_next[iz_src, ix_src] += dt ** 2 * w[it]
        u_next *= a_den
        u_prev, u = u, u_next
    if progress:
        progress(1.0)
    return dict(t=dt * decim * np.arange(rec.shape[0]), u=rec, rec_x=rec_x, snaps=snaps,
                x_snap=g.x[keep][::sd], z_snap=g.z[zkeep][::sd], dt_solver=dt)


# --------------------------------------------------------------------------------------
# spectra, ratios, f0 picks
# --------------------------------------------------------------------------------------
def amplitude_spectra(u, dt, win=10.24, skip=3.0, overlap=0.5):
    """Welch-averaged amplitude spectra sqrt(<|U|^2>) of every column of u (Hann windows,
    50% overlap), skipping the first `skip` s while the noise field builds up."""
    n = int(round(win / dt))
    i0 = int(round(skip / dt))
    step = max(1, int(n * (1 - overlap)))
    starts = range(i0, u.shape[0] - n + 1, step)
    h = np.hanning(n)[:, None]
    P = np.mean([np.abs(np.fft.rfft(h * u[a:a + n], axis=0)) ** 2 for a in starts], axis=0)
    return np.fft.rfftfreq(n, dt), np.sqrt(P)


def konno_ohmachi(f, A, b=40.0):
    """Konno & Ohmachi (1998) log-frequency smoothing of spectra A (nf, ...)."""
    out = np.zeros_like(A)
    pos = f > 0
    fp = f[pos]
    for i, fc in enumerate(f):
        if fc <= 0:
            continue
        x = b * np.log10(fp / fc)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = (np.sin(x) / x) ** 4
        w[np.abs(x) < 1e-9] = 1.0
        w[np.abs(x) > np.pi] = 0.0                    # first lobe only (|log10 ratio| < pi/b)
        out[i] = w @ A[pos] / w.sum()
    return out


def spectral_ratio(site, ref, b=25.0):
    """Smoothed site / reference amplitude-spectrum ratio at each receiver."""
    f, As = amplitude_spectra(site["u"], site["t"][1])
    _, Ar = amplitude_spectra(ref["u"], ref["t"][1])
    return f, konno_ohmachi(f, As, b) / np.maximum(konno_ohmachi(f, Ar, b), 1e-300)


def pick_f0(f, R, fmin=0.7, fmax=8.0, rel=0.6):
    """f0 at each receiver: the LOWEST-frequency local maximum of the ratio in [fmin, fmax]
    that reaches rel x the largest one (higher modes and lateral resonances can be taller
    than the fundamental), refined by a parabola in log f. Also returns its amplitude."""
    sel = np.where((f >= fmin) & (f <= fmax))[0]
    lf = np.log(f[sel])
    f0, A0 = [], []
    for j in range(R.shape[1]):
        r = R[sel, j]
        peaks = np.where((r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:]))[0] + 1
        peaks = peaks[r[peaks] >= rel * r.max()] if len(peaks) else np.array([int(np.argmax(r))])
        i = int(peaks[0]) if len(peaks) else int(np.argmax(r))
        if 0 < i < len(r) - 1:
            y0, y1, y2 = r[i - 1], r[i], r[i + 1]
            den = y0 - 2 * y1 + y2
            d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0
            f0.append(np.exp(lf[i] + d * (lf[i + 1] - lf[i])))
            A0.append(y1 - 0.25 * (y0 - y2) * d)
        else:
            f0.append(f[sel][i]); A0.append(r[i])
    return np.array(f0), np.array(A0)


def run_profile(model, grid=None, T=40.0, seed=1, rec_x=None, progress=None, **kw):
    """Site run + bare-rock reference run with the same noise -> ratios and f0 picks."""
    p1 = (lambda q: progress(0.5 * q)) if progress else None
    p2 = (lambda q: progress(0.5 + 0.5 * q)) if progress else None
    site = simulate_noise(model, grid, T, rec_x, seed=seed, progress=p1, **kw)
    ref = simulate_noise(model, grid, T, rec_x, seed=seed, homogeneous=True, progress=p2)
    f, R = spectral_ratio(site, ref)
    f0, A0 = pick_f0(f, R)
    return dict(site=site, ref=ref, f=f, R=R, f0=f0, A0=A0, rec_x=site["rec_x"])
