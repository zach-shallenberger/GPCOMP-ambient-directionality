"""
fwi.py -- Bayesian full-waveform inversion of ambient-noise correlations for Vs(z).

The SH finite-difference solver of love_waves.py is the forward operator g(m) of a
Bayesian inverse problem:

    posterior  p(m | d)  ∝  exp(-chi(m)) * prior(m),
    chi(m) = 1/2 sum_{f,j} |s(f) C_j(f; m) - D_j(f)|^2 / sigma_j(f)^2

* m        : ln(beta) in a few depth layers over a half-space (density assumed known)
* D_j(f)   : observed noise cross-correlations, channel 0 x channel j ("data")
* C_j(f; m): the same correlations predicted in model m. Because the site is laterally
             uniform, ONE impulse run recorded along the surface gives G(offset) for every
             source-receiver pair, and C_j = sum_s w_s conj(G(|x_s - x_0|)) G(|x_s - x_j|)
             (the translation trick of love_waves.LoveLine).
* s(f)     : unknown noise spectrum, eliminated in closed form at every evaluation
             (variable projection), so only the waveform shape is fitted.
* sigma    : data noise from the scatter of independent sub-stacks of the noise record.

The gradient d chi / d m comes from the *discrete adjoint* of the leapfrog scheme: one
extra run backwards in time, driven by the misfit derivatives at the surface nodes, then a
correlation of forward and adjoint fields at every cell face. It is exact for the discrete
problem (checked against finite differences in fwi_inversion.ipynb).

The MAP model (minimum of chi + prior) is found with L-BFGS-B, from low to high frequency.
Uncertainty: Laplace approximation, Gaussian posterior with covariance
(J^T C_d^-1 J + C_m^-1)^-1 at the MAP point (Gauss-Newton Hessian; J by finite differences).

Units: SI (m, s, kg/m^3, m/s).
"""
import time
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

import love_waves as LW
from simulation import ricker


# ----------------------------------------------------------------------------
# Layered parameterization  ->  mu on the cell faces of the grid
# ----------------------------------------------------------------------------
@dataclass
class Layers:
    """Layer boundaries [m] (top of each layer; the last layer is the half-space) and the
    density of each layer, which is assumed known."""
    tops: tuple = (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0)
    rho: tuple = (1800.0, 1800.0, 1800.0, 1800.0, 2100.0, 2100.0, 2100.0)

    @property
    def n(self):
        return len(self.tops)

    def fractions(self, a, b):
        """(len(a), n) fraction of each interval [a, b] lying in each layer."""
        lo = np.asarray(self.tops)
        hi = np.r_[lo[1:], np.inf]
        width = np.maximum(b - a, 1e-12)[:, None]
        return np.clip(np.minimum(b[:, None], hi) - np.maximum(a[:, None], lo), 0, None) / width

    def beta_profile(self, beta, z):
        return np.asarray(beta)[np.searchsorted(self.tops, z, side="right") - 1]


class LayeredSite:
    """A multi-layer site usable by love_waves.LoveLine (via layered_grid's hook)."""

    def __init__(self, layers, beta):
        self.layers, self.beta = layers, np.asarray(beta, float)

    def grid_arrays(self, grid):
        fm = FaceMap(self.layers, grid)
        mux, muz = fm.faces(np.asarray(self.layers.rho) * self.beta**2)
        beta = np.sqrt(mux / fm.rho)
        rep = lambda a: np.repeat(a[:, None], grid.nx, 1)
        return rep(beta), rep(fm.rho), rep(muz)


def vs30_layers(layers, beta):
    """Vs30 = 30 / sum(h_i / beta_i) over the top 30 m."""
    f = layers.fractions(np.array([0.0]), np.array([30.0]))[0] * 30.0
    return 30.0 / np.sum(f / np.asarray(beta))


class FaceMap:
    """Maps layer shear moduli mu_k to the grid's face values, and back for gradients.

    Vertical faces (between rows i and i+1): layers in series -> exact harmonic average of
    mu(z) over the cell, 1/mu_z = sum_k q_ik / mu_k.
    Horizontal faces (in row i, between columns): layers side by side (parallel) ->
    arithmetic average over the row's cell [z_i - h/2, z_i + h/2], mu_x = sum_k p_ik mu_k.
    Density at a node is the arithmetic average over the same cell.
    """

    def __init__(self, layers, grid):
        z = np.maximum(grid.z, 0.0)
        h = grid.dx
        self.q = layers.fractions(z[:-1], z[1:])                       # (nz-1, n)
        self.p = layers.fractions(np.maximum(z - h / 2, 0), z + h / 2)  # (nz, n)
        self.rho = self.p @ np.asarray(layers.rho)                      # (nz,)

    def faces(self, mu):
        muz = 1.0 / (self.q @ (1.0 / mu))
        mux = self.p @ mu
        return mux, muz

    def pullback(self, mu, g_mux, g_muz):
        """Chain rule: gradients w.r.t. face values -> gradient w.r.t. layer mu_k."""
        muz = 1.0 / (self.q @ (1.0 / mu))
        return self.p.T @ g_mux + (self.q * (muz[:, None] ** 2 / mu[None, :] ** 2)).T @ g_muz


# ----------------------------------------------------------------------------
# Leapfrog propagator with its exact discrete adjoint
# ----------------------------------------------------------------------------
class Propagator:
    """
    Forward:  u^{n+1} = B [ (2 + K A) u^n - (1 - s) u^{n-1} + dt^2 f^n ]
      A    : conservative flux operator (sum over faces of mu_f times a difference),
             free surface by the mirror rule (factor 2 on the surface row)
      K    = dt^2 / (rho h^2),  s = sigma dt / 2 (sponge),  B = P / (1 + s)
      P    : zeroes the rigid outer rows/columns
    Records u^n at the surface nodes x_src + offsets, every `decim` steps.

    Adjoint (Lagrange multipliers p^m of the update equations, m = N-1 ... 1):
        p^m = R^T r^m + 2 B p^{m+1} + A^T(K B p^{m+1}) - (1 - s) B p^{m+2}
        dJ/d mu_f = sum_m (K B p^m) . (dA/d mu_f) u^{m-1}
    dt and the sponge are fixed (from beta_cap), so they do not depend on the model.
    """

    def __init__(self, layers, grid=None, x_src=-150.0, max_offset=310.0, f0=6.0, T=3.6,
                 beta_cap=1200.0, rec_dt=0.0024):
        # The domain and sponges match the data grid's (love_waves.Grid defaults): with a
        # smaller box, sponge reflections at 3-5 Hz alone misfit the data by ~2x the noise.
        self.g = g = grid or LW.Grid(dx=2.0, half_width=160.0)
        self.layers, self.fm = layers, FaceMap(layers, g)
        self.dt = g.cfl * g.dx / beta_cap
        self.nt = int(np.ceil(T / self.dt)) + 1
        self.decim = decim = max(1, int(round(rec_dt / self.dt)))   # records every ~rec_dt
        self.w = ricker(self.dt * np.arange(self.nt), f0, 1.5 / f0) / g.dx**2
        s = 0.5 * g.sponge(beta_cap) * self.dt
        P = np.ones((g.nz, g.nx)); P[-1] = 0; P[:, [0, -1]] = 0
        self.oms, self.B = 1 - s, P / (1 + s)
        self.K = (self.dt**2 / (self.fm.rho * g.dx**2))[:, None]
        self.KB = self.K * self.B
        self.isx = g.ix(x_src)
        self.offsets = g.dx * np.arange(int(round(max_offset / g.dx)) + 1)
        self.rcols = self.isx + np.arange(len(self.offsets))
        assert g.x[self.rcols[-1]] <= g.half_width, "offsets run into the sponge"
        # where each face's flux lands (+ on the upper/left node, - on the lower/right node)
        self.ca_z = np.ones(g.nz - 1); self.ca_z[0] = 2.0               # free surface row
        self.cb_z = (np.arange(g.nz - 1) + 1 <= g.nz - 2).astype(float)
        self.ca_x = (np.arange(g.nx - 1) >= 1).astype(float)
        self.cb_x = (np.arange(g.nx - 1) + 1 <= g.nx - 2).astype(float)
        self.t_rec = self.dt * decim * np.arange((self.nt - 1) // decim + 1)

    # ---- operators -------------------------------------------------------
    def A(self, u, mux, muz):
        Fx = mux[:, None] * (u[:, 1:] - u[:, :-1])
        Fz = muz[:, None] * (u[1:] - u[:-1])
        L = np.zeros_like(u)
        L[:, :-1] += self.ca_x * Fx; L[:, 1:] -= self.cb_x * Fx
        L[:-1] += self.ca_z[:, None] * Fz; L[1:] -= self.cb_z[:, None] * Fz
        return L

    def AT(self, v, mux, muz):
        wx = mux[:, None] * (self.ca_x * v[:, :-1] - self.cb_x * v[:, 1:])
        wz = muz[:, None] * (self.ca_z[:, None] * v[:-1] - self.cb_z[:, None] * v[1:])
        out = np.zeros_like(v)
        out[:, 1:] += wx; out[:, :-1] -= wx
        out[1:] += wz; out[:-1] -= wz
        return out

    # ---- runs ------------------------------------------------------------
    def forward(self, mu, store=False):
        """Surface records (n_rec_times, n_offsets); optionally keep every state u^n."""
        mux, muz = self.fm.faces(mu)
        g = self.g
        u_prev, u = np.zeros((g.nz, g.nx)), np.zeros((g.nz, g.nx))
        rec = np.zeros((len(self.t_rec), len(self.offsets)))
        states = np.zeros((self.nt, g.nz, g.nx), np.float32) if store else None   # ~1 GB
        for n in range(self.nt):
            if n % self.decim == 0:
                rec[n // self.decim] = u[0, self.rcols]
            if store:
                states[n] = u
            nxt = 2 * u - self.oms * u_prev + self.K * self.A(u, mux, muz)
            nxt[0, self.isx] += self.dt**2 * self.w[n]
            u_prev, u = u, self.B * nxt
        self._states, self._mu = states, mu
        return rec

    def adjoint_gradient(self, r):
        """Gradient of J w.r.t. layer mu_k, given r = dJ/d(records) (same shape as records).
        Needs the states of the last forward(store=True) call."""
        mux, muz = self.fm.faces(self._mu)
        U, g = self._states, self.g
        p1 = np.zeros((g.nz, g.nx)); p2 = np.zeros_like(p1)        # p^{m+1}, p^{m+2}
        gx, gz = np.zeros(g.nz), np.zeros(g.nz - 1)
        for m in range(self.nt - 1, 0, -1):
            KBp1 = self.KB * p1
            p = 2 * self.B * p1 + self.AT(KBp1, mux, muz) - self.oms * self.B * p2
            if m % self.decim == 0:
                p[0, self.rcols] += r[m // self.decim]
            # gradient: (K B p^m) . (dA/dmu_f) u^{m-1}
            q, u = self.KB * p, U[m - 1]
            gx += np.sum((u[:, 1:] - u[:, :-1]) * (self.ca_x * q[:, :-1] - self.cb_x * q[:, 1:]), axis=1)
            gz += np.sum((u[1:] - u[:-1]) * (self.ca_z[:, None] * q[:-1] - self.cb_z[:, None] * q[1:]), axis=1)
            p2, p1 = p1, p
        return self.fm.pullback(self._mu, gx, gz)


# ----------------------------------------------------------------------------
# Data, misfit with variable projection, and the posterior
# ----------------------------------------------------------------------------
class NoiseFWI:
    """
    Bayesian FWI of noise correlations recorded on a surface line.

    rec_x   : channel positions [m]; channel 0 is the virtual source
    sources : dict(x=positions [m], w=weights) of in-line surface noise sources; every
              source-receiver distance must be a multiple of the FD grid spacing
    D, var  : observed correlation spectra (nf, nrec) on self.f and their noise variance
    band    : (fmin, fmax) [Hz] used in the misfit (cosine-tapered edges)
    prior   : Gaussian on ln(beta), mean m0 and standard deviation prior_std
    """

    def __init__(self, layers, rec_x, sources, prop=None, prior_std=0.5, m0=None):
        self.layers = layers
        self.prop = prop or Propagator(layers)
        self.rec_x = np.asarray(rec_x, float)
        self.src_x, self.src_w = np.asarray(sources["x"], float), np.asarray(sources["w"], float)
        h = self.prop.g.dx
        da = np.abs(self.src_x - self.rec_x[0]) / h                      # source -> channel 0
        db = np.abs(self.src_x[:, None] - self.rec_x[None, :]) / h      # source -> channel j
        assert np.allclose(da, np.round(da)) and np.allclose(db, np.round(db)), "distances off the grid"
        self.ia, self.ib = np.round(da).astype(int), np.round(db).astype(int)
        assert self.ib.max() < len(self.prop.offsets), "a source is farther than max_offset"
        nr = len(self.prop.t_rec)
        self.nfft = 2 * nr
        self.f = np.fft.rfftfreq(self.nfft, self.prop.t_rec[1])
        self.rho = np.asarray(layers.rho)
        self.prior_std, self.m0 = prior_std, m0
        self.history = []

    # ---- data ------------------------------------------------------------
    def spectra(self, rec):
        """Records (nt, noff) -> G(f, offset) and the predicted correlations C(f, j)."""
        G = np.fft.rfft(rec, self.nfft, axis=0)
        C = np.einsum("s,fs,fsj->fj", self.src_w, np.conj(G[:, self.ia]), G[:, self.ib])
        return G, C

    def observe_from(self, line, n_windows=400, n_sub=10, seed=0):
        """'Observed' data: noise correlations from a LoveLine (a different, finer grid, so
        no inverse crime), simulated with random noise windows. Returns D and its variance
        on this object's frequency grid, from the scatter of n_sub sub-stacks."""
        src = dict(p=-self.src_x, theta=np.zeros(len(self.src_x)),
                   depth_index=np.zeros(len(self.src_x), int), w=self.src_w)
        subs = [line.random_gather(self.rec_x, src, n_windows // n_sub, seed=seed + k, fold=False)
                for k in range(n_sub)]
        subs = [self._resample(line, S) for S in subs]
        D = np.mean(subs, axis=0)
        var = np.var(subs, axis=0) / n_sub + 1e-30            # variance of the mean (|.|^2)
        return D, var

    def _resample(self, line, S):
        """Correlation spectra on the LoveLine frequency grid -> this object's grid, via
        the two-sided lag-time correlation (lags within +-T of this object)."""
        full = np.zeros((line.nfft // 2 + 1, S.shape[1]), complex)
        full[:len(line.f)] = S
        c = np.fft.irfft(full, line.nfft, axis=0)                        # circular lags
        dt_l = line.t[1] - line.t[0]
        lag_l = dt_l * np.r_[np.arange(line.nfft // 2), np.arange(-line.nfft // 2, 0)]
        nr = len(self.prop.t_rec); dt = self.prop.t_rec[1]
        lag = dt * np.r_[np.arange(nr), np.arange(-nr, 0)]
        order = np.argsort(lag_l)
        out = np.array([np.interp(lag, lag_l[order], c[order, j]) for j in range(c.shape[1])]).T
        taper = np.clip((self.prop.t_rec[-1] - np.abs(lag)) / 0.2, 0, 1)    # soft edge at +-T
        return np.fft.rfft(out * taper[:, None], axis=0) * dt / dt_l

    def model_error_var(self, line_ref, m_ref, smooth=5):
        """Bayesian approximation-error estimate of the *modelling* error variance.

        The data come from a finer grid than the inversion's (and a wider sponge), so even
        the true model cannot fit them to the noise level. We estimate that error at a
        reference model the inversion is allowed to know (the prior mean, NOT the truth):
        expected correlations from line_ref (a LoveLine for m_ref on the data grid) minus
        the s-projected prediction of this propagator for m_ref. Its power, smoothed over
        neighbouring channels, is added to the noise variance."""
        src = dict(p=-self.src_x, theta=np.zeros(len(self.src_x)),
                   depth_index=np.zeros(len(self.src_x), int), w=self.src_w)
        Dref = self._resample(line_ref, line_ref.noise_gather(self.rec_x, src, fold=False))
        _, C = self.spectra(self.prop.forward(self.mu(m_ref)))
        w = np.ones((len(self.f), 1))
        s = np.sum(np.conj(C) * Dref, 1) / np.maximum(np.sum(np.abs(C) ** 2, 1), 1e-300)
        e2 = np.abs(s[:, None] * C - Dref) ** 2
        k = np.ones(smooth) / smooth
        return np.array([np.convolve(row, k, mode="same") for row in e2])

    def set_data(self, D, var, band=(3.0, 8.0)):
        self.D, self.var = D, var
        self.set_band(band)

    def set_band(self, band):
        lo, hi = band
        f = self.f
        win = np.clip(np.minimum((f - lo + 1.0) / 1.0, (hi + 1.0 - f) / 1.0), 0, 1)
        self.win = np.sin(0.5 * np.pi * win) ** 2
        self.band = band
        self.cw = self.win[:, None] / self.var                            # 1 / sigma^2 weights

    # ---- objective -------------------------------------------------------
    def mu(self, m):
        return self.rho * np.exp(2 * m)

    def misfit_parts(self, C):
        """Variable projection: best complex s(f) for this C, then residual and chi."""
        s = np.sum(self.cw * np.conj(C) * self.D, 1) / np.maximum(np.sum(self.cw * np.abs(C) ** 2, 1), 1e-300)
        R = s[:, None] * C - self.D
        return s, R, 0.5 * np.sum(self.cw * np.abs(R) ** 2)

    def prior(self, m):
        if self.m0 is None:
            return 0.0, np.zeros_like(m)
        d = (m - self.m0) / self.prior_std
        return 0.5 * d @ d, d / self.prior_std

    def objective(self, m, grad=True):
        """chi(m) + prior(m) and its gradient w.r.t. m = ln(beta) (adjoint state)."""
        mu = self.mu(m)
        rec = self.prop.forward(mu, store=grad)
        G, C = self.spectra(rec)
        s, R, chi = self.misfit_parts(C)
        pr, gpr = self.prior(m)
        self.history.append((self.band, chi, pr, np.exp(m).copy()))
        if not grad:
            return chi + pr
        # dchi = sum_{f,offset} Re[conj(H) dG] (Wirtinger calculus on the bilinear C(G));
        # then dchi/dg(t) = Re sum_f H e^{+i w t} = (nfft/2) irfft(H)
        E = self.cw * np.conj(R) * s[:, None]                               # (nf, nrec)
        H = np.zeros_like(G)
        wE = self.src_w[None, :, None] * E[:, None, :]                       # (nf, ns, nrec)
        np.add.at(H.T, self.ia, np.sum(wE * G[:, self.ib], axis=2).T)        # from conj(G_a)
        HbT = (np.conj(wE) * G[:, self.ia][:, :, None]).reshape(len(self.f), -1).T
        np.add.at(H.T, self.ib.ravel(), HbT)                                 # from G_b
        r = 0.5 * self.nfft * np.fft.irfft(H, self.nfft, axis=0)[:len(self.prop.t_rec)]
        g_mu = self.prop.adjoint_gradient(r)
        return chi + pr, g_mu * 2 * mu + gpr                                 # d mu / d m = 2 mu

    # ---- optimisation and uncertainty -------------------------------------
    def fit(self, m_start, bands=((3, 6), (3, 8), (3, 10)), bounds=(np.log(120), np.log(1000)),
            maxiter=25, log=print):
        """Multiscale MAP estimate with L-BFGS-B (low frequencies first)."""
        m = np.asarray(m_start, float).copy()
        for band in bands:
            self.set_band(band)
            t0 = time.time()
            res = minimize(self.objective, m, jac=True, method="L-BFGS-B",
                           bounds=[bounds] * len(m), options=dict(maxiter=maxiter))
            m = res.x
            if log:
                log(f"band {band[0]}-{band[1]} Hz: objective {res.fun:10.1f}, {res.nit} iterations, "
                    f"{time.time() - t0:5.0f} s; beta = {np.round(np.exp(m))}")
        return m

    def laplace(self, m, h=1e-3):
        """Gaussian (Laplace) approximation of the posterior at m: covariance of ln(beta).
        Gauss-Newton Hessian J^T C_d^-1 J + C_m^-1, J by central finite differences of the
        weighted, s-projected data residual (real and imaginary parts stacked)."""
        def wres(mm):
            _, C = self.spectra(self.prop.forward(self.mu(mm)))
            s, R, _ = self.misfit_parts(C)
            r = np.sqrt(self.cw) * R
            return np.r_[r.real.ravel(), r.imag.ravel()]
        J = np.column_stack([(wres(m + h * e) - wres(m - h * e)) / (2 * h) for e in np.eye(len(m))])
        Hm = J.T @ J
        if self.m0 is not None:
            Hm = Hm + np.eye(len(m)) / self.prior_std**2
        return np.linalg.inv(Hm), J


def line_noise_sources(rec_x, gap=30.0, reach=90.0, spacing=4.0):
    """In-line surface sources beyond both ends of the line (gap .. gap+reach m out)."""
    x0, x1 = np.min(rec_x), np.max(rec_x)
    left = np.arange(x0 - gap - reach, x0 - gap + 1e-9, spacing)
    right = np.arange(x1 + gap, x1 + gap + reach + 1e-9, spacing)
    xs = np.r_[left, right]
    return dict(x=xs, w=np.ones(len(xs)))
