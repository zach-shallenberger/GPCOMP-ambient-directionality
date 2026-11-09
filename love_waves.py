"""
love_waves.py -- passive Love-wave site characterisation from ambient noise.

Physics
-------
In a vertical x-z plane, SH motion u = u_y(x, z, t) decouples from P-SV and obeys

    rho(z) u_tt = d/dx(mu u_x) + d/dz(mu u_z) + f,      mu = rho beta^2,

with a stress-free surface mu u_z = 0 at z = 0. Love waves are SH waves trapped in a slow
surface layer, so this scalar 2D hyperbolic PDE gives real, dispersive surface waves.

Numerics (course module 06, extended to variable coefficients)
---------------------------------------------------------------
* Conservative 2nd-order operator: fluxes mu_{i+1/2} (u_{i+1} - u_i)/h with mu averaged
  *harmonically* at half-points, so traction is continuous across a sharp interface.
  (A non-conservative beta^2 lap(u) operator gives the wrong reflection coefficient; see
  verification notebook.)
* Free surface: even mirror ghost row u_{-1} = u_1, i.e. zero flux through z = 0.
* Leapfrog in time, CFL limit beta_max dt / h <= 1/sqrt(2); quadratic damping sponge on
  the sides and bottom (thicker on the sides, where trapped Love waves travel).

Noise interferometry
--------------------
For a laterally uniform model G depends only on x - x_s, so ONE impulse run at a surface
point gives the Green's function for every source-receiver pair (translation trick). The
expected noise correlation  C_0j = sum_s w_s conj(G(s,0)) G(s,j)  for any source layout is
then a re-weighting of stored spectra. Dispersion images use the phase-shift transform
(Park et al. 1998); the exact answer comes from the Love-wave dispersion relation.

Units: SI (m, s, kg/m^3, m/s).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from simulation import ricker            # Ricker wavelet shared with the DAS project

CFL_LIMIT = 1 / np.sqrt(2)               # 2D leapfrog + 2nd-order Laplacian


# ----------------------------------------------------------------------------
# Layer-over-half-space site model and its analytic Love-wave dispersion
# ----------------------------------------------------------------------------
@dataclass
class Site:
    """Soil layer (beta1, rho1, thickness H) over a half-space (beta2, rho2)."""
    b1: float = 250.0     # layer shear speed [m/s]
    b2: float = 800.0     # half-space shear speed [m/s]
    H: float = 15.0       # layer thickness [m]
    r1: float = 1800.0    # layer density [kg/m^3]
    r2: float = 2100.0    # half-space density [kg/m^3]

    def profile(self, z):
        """beta(z), rho(z) at depths z [m]."""
        top = np.asarray(z) < self.H
        return np.where(top, self.b1, self.b2), np.where(top, self.r1, self.r2)

    def cutoff(self, n):
        """Cutoff frequency of Love mode n [Hz]: f_n = n / (2 H sqrt(1/b1^2 - 1/b2^2))."""
        return n / (2 * self.H * np.sqrt(self.b1**-2 - self.b2**-2))


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


def love_root(f, site, n=0):
    """Phase velocity [m/s] of Love mode n at frequency f, or NaN below cutoff.

    Dispersion relation (Aki & Richards, layer over half-space):
        tan(w H eta1) = mu2 eta2 / (mu1 eta1),
        eta1 = sqrt(1/b1^2 - 1/c^2),  eta2 = sqrt(1/c^2 - 1/b2^2),  b1 < c < b2.
    With x = w H eta1, mode n has x in (n pi, n pi + pi/2). Writing it as
    F(x) = mu1 eta1 sin x - mu2 eta2 cos x avoids the poles of tan, and F changes sign
    exactly once on each branch, so brentq is safe.
    """
    w = 2 * np.pi * f
    mu1, mu2 = site.r1 * site.b1**2, site.r2 * site.b2**2
    smax = site.b1**-2 - site.b2**-2                 # eta1^2 at c = b2
    xmax = w * site.H * np.sqrt(smax)
    lo, hi = n * np.pi, min(n * np.pi + np.pi / 2, xmax)
    if hi <= lo:
        return np.nan

    def F(x):
        eta1 = x / (w * site.H)
        eta2 = np.sqrt(max(smax - eta1**2, 0.0))
        return mu1 * eta1 * np.sin(x) - mu2 * eta2 * np.cos(x)

    if F(lo + 1e-12) * F(hi - 1e-12) > 0:            # root sits at the cutoff itself
        return np.nan
    x = brentq(F, lo + 1e-12, hi - 1e-12, xtol=1e-13)
    return 1.0 / np.sqrt(site.b1**-2 - (x / (w * site.H)) ** 2)


def love_modes(f, site, n_modes=3):
    """Phase velocities, shape (n_modes, len(f)), NaN where a mode does not exist."""
    return np.array([[love_root(fi, site, n) for fi in np.atleast_1d(f)] for n in range(n_modes)])


def group_velocity(f, site, n=0, df=1e-3):
    """U = d(omega)/dk by centred differences of k(omega) = omega / c."""
    k = [2 * np.pi * fi / love_root(fi, site, n) for fi in (f - df, f + df)]
    return 2 * np.pi * 2 * df / (k[1] - k[0])


# ----------------------------------------------------------------------------
# Finite-difference grid and SH solver
# ----------------------------------------------------------------------------
@dataclass
class Grid:
    """x-z grid. z = 0 is row 0 (free surface unless free_surface=False); x is centred on 0.
    Sponges are added *outside* the physical region [-half_width, half_width] x [0, depth]."""
    dx: float = 1.0               # grid spacing [m] (same in x and z)
    half_width: float = 450.0     # physical half-width [m]
    depth: float = 200.0          # physical depth [m]
    side: float = 250.0           # side sponge width [m] (Love waves travel sideways)
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


def simulate_sh(grid, beta, rho, src_xz, f0, T, t0=None, rows=(), points=(), decim=1,
                snapshots=(), conservative=True, src_row=False, muz=None, snap_decim=1, on_snapshot=None,
                progress=None):
    """
    Leapfrog solution of rho u_tt + rho sigma u_t = div(mu grad u) + rho w(t) delta(x - x_s).

    The source term is scaled by rho so that, in a homogeneous medium, u is exactly the
    Ricker-convolved Green's function of u_tt = beta^2 lap(u) + delta (as in simulation.py).

    beta, rho  : (nz, nx) arrays of shear speed [m/s] and density
    src_xz     : (x, z) of the point source [m]; with src_row=True only z is used and the
                 source is a whole horizontal line (plane wave, for periodic 1D tests)
    rows       : depths [m] of horizontal lines recorded in full (|x| <= half_width)
    points     : (n, 2) array of (x, z) point receivers
    decim      : keep every decim-th time step in the records
    snapshots  : times [s] at which to save the wavefield (physical region only, every
                 snap_decim-th node, float32); on_snapshot(t, frame) is called with each
                 one as it is taken (used by the app to show the wave live)
    conservative: False -> naive beta^2 lap(u) operator (for the interface test only)
    muz        : optional (nz-1, nx) mu at the vertical half-points (see layered_grid);
                 default is the harmonic mean of the node values
    Returns dict with t, rows (nt, nrow, nx_phys), x (nx_phys), points (nt, n), snaps.
    """
    g = grid
    t0 = 1.5 / f0 if t0 is None else t0
    dt = g.cfl * g.dx / beta.max()
    nt = int(np.ceil(T / dt)) + 1
    w = ricker(dt * np.arange(nt), f0, t0) / (g.dx if src_row else g.dx**2)

    mu = rho * beta**2
    mux = harmonic(mu[:, 1:], mu[:, :-1])            # mu at (i, j+1/2)
    if muz is None:
        muz = harmonic(mu[1:], mu[:-1])              # mu at (i+1/2, j)
    if g.periodic_x:
        mux = harmonic(mu, np.roll(mu, -1, axis=1))
    k = dt**2 / (rho * g.dx**2)                      # dt^2 / (rho h^2)
    b2 = beta**2 * dt**2 / g.dx**2                   # for the naive operator
    s = 0.5 * g.sponge(beta.max()) * dt
    a_prev, a_den = 1 - s, 1 / (1 + s)

    isz, isx = g.iz(src_xz[1]), g.ix(src_xz[0])
    jr = g.iz(np.atleast_1d(rows)) if len(rows) else np.zeros(0, int)
    keep = np.abs(g.x) <= g.half_width + 1e-9
    pts = np.reshape(np.asarray(points, float), (-1, 2))
    pz, px = g.iz(pts[:, 1]), g.ix(pts[:, 0])
    snap_steps = {int(round(ts / dt)): round(float(ts), 3) for ts in snapshots}
    zkeep = (g.z >= -1e-9) & (g.z <= g.depth + 1e-9)

    u_prev, u = np.zeros((g.nz, g.nx)), np.zeros((g.nz, g.nx))
    L = np.zeros_like(u)
    rec_rows = np.zeros(((nt - 1) // decim + 1, len(jr), keep.sum()))
    rec_pts = np.zeros(((nt - 1) // decim + 1, len(pz)))
    snaps = {}
    for it in range(nt):
        if progress and it % max(nt // 20, 1) == 0:
            progress(it / nt)
        if it % decim == 0:
            rec_rows[it // decim] = u[jr][:, keep]
            rec_pts[it // decim] = u[pz, px]
        if it in snap_steps:
            ts = snap_steps[it]
            snaps[ts] = u[zkeep][:, keep][::snap_decim, ::snap_decim].astype(np.float32)
            if on_snapshot:
                on_snapshot(ts, snaps[ts])
        L[:] = 0.0
        if conservative:
            # x fluxes  F_{j+1/2} = mu (u_{j+1} - u_j);  L_j += F_{j+1/2} - F_{j-1/2}
            if g.periodic_x:
                Fx = mux * (np.roll(u, -1, axis=1) - u)
                L += Fx - np.roll(Fx, 1, axis=1)
            else:
                Fx = mux * (u[:, 1:] - u[:, :-1])
                L[:, 1:-1] += Fx[:, 1:] - Fx[:, :-1]
            Fz = muz * (u[1:] - u[:-1])
            L[1:-1] += Fz[1:] - Fz[:-1]
            if g.free_surface:                       # ghost u_{-1} = u_1: flux in = -flux out
                L[0] += 2 * Fz[0]
            u_next = 2 * u - a_prev * u_prev + k * L
        else:
            # naive operator beta^2 (u_xx + u_zz), same free surface and edges
            if g.periodic_x:
                L += np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)
            else:
                L[:, 1:-1] += u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]
            L[1:-1] += u[2:] - 2 * u[1:-1] + u[:-2]
            if g.free_surface:
                L[0] += 2 * (u[1] - u[0])
            u_next = 2 * u - a_prev * u_prev + b2 * L
        if src_row:
            u_next[isz] += dt**2 * w[it]
        else:
            u_next[isz, isx] += dt**2 * w[it]
        u_next *= a_den
        u_next[-1] = 0.0                             # rigid outer edges (inside the sponge)
        if not g.periodic_x:
            u_next[:, [0, -1]] = 0.0
        u_prev, u = u, u_next
    return dict(t=dt * decim * np.arange(rec_rows.shape[0]), dt=dt, rows=rec_rows,
                x=g.x[keep], points=rec_pts, snaps=snaps, z=g.z[zkeep][::snap_decim],
                x_snap=g.x[keep][::snap_decim])


def layered_grid(site, grid):
    """beta, rho at the nodes and mu at the vertical half-points for a Site on a Grid.

    The interface z = H generally falls on or between nodes. To place it at the right
    depth to 2nd order, mu_{i+1/2} is the exact harmonic average of mu(z) over the cell
    [z_i, z_{i+1}] (series springs), and rho_i is the arithmetic average over
    [z_i - h/2, z_i + h/2] (mass). Simply sampling beta(z) at nodes would move the
    effective interface by up to h/2: an O(h) error in every dispersion curve.
    """
    z, h = np.maximum(grid.z, 0.0), grid.dx
    mu1, mu2 = site.r1 * site.b1**2, site.r2 * site.b2**2
    frac = lambda a, b: np.clip((site.H - a) / (b - a), 0, 1)     # layer fraction of [a, b]
    q = frac(z[:-1], z[1:] + 1e-12 * (z[1:] == z[:-1]))
    mu_half = 1.0 / (q / mu1 + (1 - q) / mu2)
    lo, hi = np.maximum(z - h / 2, 0.0), z + h / 2
    p = frac(lo, hi)
    rho = p * site.r1 + (1 - p) * site.r2
    mu_node = 1.0 / (p / mu1 + (1 - p) / mu2)
    beta = np.sqrt(mu_node / rho)
    rep = lambda a: np.repeat(a[:, None], grid.nx, 1)
    return rep(beta), rep(rho), rep(mu_half)


# ----------------------------------------------------------------------------
# Dispersion analysis
# ----------------------------------------------------------------------------
def phase_shift(U, offsets, f, c):
    """Phase-shift dispersion image (Park et al. 1998).

    U       : (nf, nrec) complex spectra of a shot gather (rfft convention e^{-i w t})
    offsets : (nrec,) source-receiver offsets [m]
    Returns |sum_j U_j/|U_j| exp(+i w x_j / c)| / nrec, shape (nf, nc): 1 where the
    gather is a perfect plane wave with phase velocity c.
    """
    Un = U / np.maximum(np.abs(U), 1e-30)
    ph = np.exp(1j * 2 * np.pi * f[:, None, None] * offsets[None, None, :] / c[None, :, None])
    return np.abs(np.einsum("fcj,fj->fc", ph, Un)) / len(offsets)


def pick_ridge(img, c, f, fmin, fmax, track=False, window=0.1):
    """Ridge picks c(f) of a dispersion image between fmin and fmax.

    track=False: global maximum over c at each frequency (parabolic refinement). This is
        what a naive automatic picker does, so a higher mode that is locally stronger *is*
        picked.
    track=True: start from the global maximum at fmin, then follow the ridge upward in
        frequency, searching only within +-window (relative) of the previous pick. This
        stays on the fundamental through a mode jump, as an analyst would.
    """
    sel = np.flatnonzero((f >= fmin) & (f <= fmax))
    out, prev = [], None
    for i in sel:
        row = img[i]
        ok = np.ones(len(c), bool) if (prev is None or not track) else np.abs(c / prev - 1) < window
        k = int(np.flatnonzero(ok)[np.argmax(row[ok])])
        kf = float(k)
        if 0 < k < len(c) - 1:
            y0, y1, y2 = row[k - 1:k + 2]
            den = y0 - 2 * y1 + y2
            kf = k + (0.5 * (y0 - y2) / den if den < 0 else 0.0)
        prev = np.interp(kf, np.arange(len(c)), c)
        out.append(prev)
    return f[sel], np.array(out)


BOUNDS = np.log([[80.0, 1.0, 150.0], [1000.0, 80.0, 3000.0]])   # (beta1, H, beta2) limits


def invert_site(f, c_obs, start=(200.0, 10.0, 600.0), r1=1800.0, r2=2100.0, n_iter=40,
                damping=1e-2):
    """Damped Gauss-Newton (Levenberg-Marquardt) fit of (beta1, H, beta2) to a picked
    curve, assuming it is the FUNDAMENTAL mode, with love_root as the forward model
    (cf. 09_Regression). Parameters are solved in log form and kept inside BOUNDS.
    Returns (params, rms misfit [m/s])."""
    m = np.log(np.asarray(start, float))

    def fwd(p):
        b1, H, b2 = np.exp(p)
        if b2 <= b1 * 1.01:
            return np.full(len(f), 1e6)
        s = Site(b1, b2, H, r1, r2)
        return np.array([love_root(fi, s) for fi in f])

    r = c_obs - fwd(m)
    lam = damping
    for _ in range(n_iter):
        J = np.column_stack([(fwd(m + 1e-4 * e) - fwd(m)) / 1e-4 for e in np.eye(3)])
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-9 * np.eye(3), J.T @ r)
        m_new = np.clip(m + step, *BOUNDS)
        r_new = c_obs - fwd(m_new)
        if r_new @ r_new < r @ r:
            m, r, lam = m_new, r_new, lam / 3
            if np.abs(step).max() < 1e-6:
                break
        else:
            lam *= 5
            if lam > 1e8:
                break
    return np.exp(m), np.sqrt(np.mean(r**2))


# ----------------------------------------------------------------------------
# One FD run -> every source-receiver Green's function (translation trick)
# ----------------------------------------------------------------------------
SOURCE_DEPTHS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)   # depths [m] at which noise sources may sit


class LoveLine:
    """
    Impulse at the surface at x = 0, recorded along horizontal lines at SOURCE_DEPTHS.
    By reciprocity G(receiver at surface x_r <- source at (x_s, z_s)) = row_{z_s}(x_s - x_r),
    and by lateral uniformity it only depends on |x_s - x_r|. Spectra are stored up to fmax
    and divided by the Ricker spectrum, so they are band-limited Green's functions.
    """

    FRAME_TIMES = np.round(np.arange(0.05, 2.501, 0.05), 3)   # wavefield movie frames [s]

    def __init__(self, site=None, grid=None, f0=8.0, T=3.5, fmax=30.0, progress=None,
                 on_snapshot=None):
        self.site, self.grid = site or Site(), grid or Grid()
        beta, rho, muz = layered_grid(self.site, self.grid)
        out = simulate_sh(self.grid, beta, rho, (0.0, 0.0), f0, T, rows=SOURCE_DEPTHS, muz=muz,
                          decim=max(1, int(0.002 / (self.grid.cfl * self.grid.dx / beta.max()))),
                          snapshots=self.FRAME_TIMES, snap_decim=max(1, round(2.0 / self.grid.dx)),
                          on_snapshot=on_snapshot, progress=progress)
        # snaps: {time: wavefield on a ~2 m grid over the physical region}, for animations
        self.t, self.snaps, self.z_snap, self.x_snap = out["t"], out["snaps"], out["z"], out["x_snap"]
        x, rows = out["x"], out["rows"]
        dt = self.t[1] - self.t[0]
        self.nfft = 2 * len(self.t)                  # zero pad: correlations need 2T of lag
        f = np.fft.rfftfreq(self.nfft, dt)
        keep = f <= fmax
        self.f = f[keep]
        W = np.fft.rfft(ricker(dt * np.arange(self.nfft), f0, 1.5 / f0), self.nfft)[keep]
        Wok = np.abs(W) > 1e-3 * np.abs(W).max()
        taper = np.r_[np.ones(len(self.t) - 50), np.hanning(100)[50:]][:, None, None]
        G = np.fft.rfft(rows * taper, self.nfft, axis=0)[keep]    # (nf, nrow, nx)
        G = np.where(Wok[:, None, None], G / np.where(Wok, W, 1)[:, None, None], 0)
        i0 = int(np.argmin(np.abs(x)))
        n = min(i0, len(x) - 1 - i0)
        sym = 0.5 * (G[:, :, i0:i0 + n + 1] + G[:, :, i0 - n:i0 + 1][:, :, ::-1])  # G(|d|)
        self.offsets = x[i0:i0 + n + 1]
        self._spl = CubicSpline(self.offsets, np.moveaxis(sym, 2, 0), axis=0)   # complex OK
        self.max_offset = self.offsets[-1]

    def green(self, dist, depth_index=0):
        """Spectra (nf, len(dist)) of the Green's function at horizontal distances dist."""
        d = np.abs(np.asarray(dist, float))
        if d.max() > self.max_offset:
            raise ValueError(f"offset {d.max():.0f} m exceeds stored {self.max_offset:.0f} m")
        return self._spl(d)[:, :, depth_index].T

    # ---- gathers ---------------------------------------------------------
    def active_gather(self, rec_x, x_shot):
        """Spectra of an active shot at the surface (the classic MASW record)."""
        return self.green(np.asarray(rec_x) - x_shot)

    def noise_gather(self, rec_x, sources, fold=True):
        """Expected noise correlations C_0j between channel 0 and every channel.

        sources: dict of arrays p, theta, depth_index, w (one entry per source). The wave of
        source s reaches receiver j after an in-plane distance |p_s + x_j cos(theta_s)|:
        theta = 0 is an in-line source at x = -p, theta != 0 a far, off-line source whose
        wavefront crosses the line at angle theta (a crossline sensor sees cos(theta) of it).
        With fold=True the acausal branch is time-reversed and added to the causal one
        (standard for a virtual source at the end of the line).
        """
        rec_x = np.asarray(rec_x, float)
        C = np.zeros((len(self.f), len(rec_x)), complex)
        for di in np.unique(sources["depth_index"]):
            m = sources["depth_index"] == di
            p, th, w = sources["p"][m], sources["theta"][m], sources["w"][m]
            dist = np.abs(p[:, None] + rec_x[None, :] * np.cos(th)[:, None])   # (ns, nrec)
            G = self._spl(dist.ravel())[:, :, di].T.reshape(len(self.f), *dist.shape)
            amp = (w * np.cos(th) ** 2)[None, :, None]
            C += np.sum(amp * np.conj(G[:, :, :1]) * G, axis=1)
        return self.fold(C) if fold else C

    def random_gather(self, rec_x, sources, n_windows, seed=0, fold=True):
        """Correlations from n_windows of *simulated* noise: every source fires with a random
        complex spectrum in every window (simultaneous, uncorrelated sources), and the
        window-by-window cross-spectra are stacked. Tends to noise_gather as n_windows grows."""
        rec_x = np.asarray(rec_x, float)
        rng = np.random.default_rng(seed)
        C = np.zeros((len(self.f), len(rec_x)), complex)
        for di in np.unique(sources["depth_index"]):
            m = sources["depth_index"] == di
            p, th, w = sources["p"][m], sources["theta"][m], sources["w"][m]
            dist = np.abs(p[:, None] + rec_x[None, :] * np.cos(th)[:, None])
            G = self._spl(dist.ravel())[:, :, di].T.reshape(len(self.f), *dist.shape)
            G *= (np.sqrt(w) * np.cos(th))[None, :, None]
            for _ in range(n_windows):
                a = rng.normal(size=(len(self.f), m.sum())) + 1j * rng.normal(size=(len(self.f), m.sum()))
                U = np.einsum("fs,fsj->fj", a, G) / np.sqrt(2)
                C += np.conj(U[:, :1]) * U
        C /= n_windows
        return self.fold(C) if fold else C

    def fold(self, C):
        """C(t) + C(-t) for t >= 0: causal plus time-reversed acausal branch."""
        nf_full = self.nfft // 2 + 1
        full = np.zeros((nf_full, C.shape[1]), complex)
        full[:len(self.f)] = C
        c = np.fft.irfft(full, self.nfft, axis=0)
        c[1:self.nfft // 2] += c[:self.nfft // 2:-1].copy()
        c[self.nfft // 2:] = 0.0
        return np.fft.rfft(c, axis=0)[:len(self.f)]

    def to_time(self, U, fmin=2.0, fmax=25.0):
        """Band-passed (cosine-tapered) time traces of spectra U, for display."""
        taper = np.clip(np.minimum((self.f - fmin + 1) / 2, (fmax + 1 - self.f) / 2), 0, 1)
        full = np.zeros((self.nfft // 2 + 1, U.shape[1]), complex)
        full[:len(self.f)] = U * np.sin(0.5 * np.pi * taper)[:, None] ** 2
        dt = self.t[1] - self.t[0]
        return dt * np.arange(self.nfft), np.fft.irfft(full, self.nfft, axis=0)


def line_sources(kind="both", reach=150.0, spacing=2.0, depth_index=0, theta_deg=0.0,
                 rec_x=None, standoff=60.0):
    """Noise source layouts (all weights equal).

    kind = "left" / "right" / "both": in-line sources along the line beyond the array ends,
           out to `reach` m.
           "everywhere": in-line sources along the whole line, including under the array.
           "offline": far sources whose waves cross the line at +-theta_deg, from both
           ends (standoff is the in-plane distance to the first channel).
    """
    rec_x = np.asarray(rec_x)
    x0, x1 = rec_x.min(), rec_x.max()
    left = np.arange(x0 - reach, x0 - 0.5 * spacing, spacing)
    right = np.arange(x1 + spacing, x1 + reach + 1e-9, spacing)
    if kind == "offline":
        th = np.radians(theta_deg)
        p = standoff + np.arange(0, reach, spacing)
        # from the left: distance p + x cos(th); from the right: wave travels -x
        P = np.r_[p - x0 * np.cos(th), p + x1 * np.cos(th)]
        TH = np.r_[np.full(len(p), th), np.full(len(p), np.pi - th)]
    else:
        xs = {"left": left, "right": right, "both": np.r_[left, right],
              "everywhere": np.arange(x0 - reach, x1 + reach + 1e-9, spacing)}[kind]
        P, TH = -xs, np.zeros(len(xs))
    return dict(p=P, theta=TH, depth_index=np.full(len(P), depth_index), w=np.ones(len(P)))


def channels(n=48, spacing=4.0, x_first=-90.0):
    """Receiver x positions [m] of a surface line."""
    return x_first + spacing * np.arange(n)
