# Directional ambient noise and travel-time bias in seismic interferometry

GPGN 436/536 final project (Zach).

**Question.** Ambient-noise tomography cross-correlates the noise recorded at two stations and treats the result as the Green's function between them. That is only exact when the noise arrives equally from all directions. Real noise is directional. In a sedimentary basin at 1–3 Hz it comes mostly from roads, towns, industry or a nearby coast. This app answers, for station pairs 1–5 km apart in a basin: *how much does a directional noise field bias the travel times and velocities measured this way, and when does it matter?*

**PDE and method.** The 2D scalar wave equation, $u_{tt} + \sigma u_t = c^2\nabla^2 u + f$, which stands in for fundamental-mode Rayleigh waves, solved with the finite-difference leapfrog scheme from course module 06:
- 2nd order in time, with a 4th-order Laplacian
- a quadratic damping sponge as the absorbing boundary
- CFL limit $\sqrt{3/8}$

Noise sources sit on a 12 km ring around the station pair. By reciprocity, two impulse simulations (one at each station) give the Green's function from every source to both stations. After that, any source distribution $B(\theta)$ (von Mises direction $\theta_0$, concentration $\kappa$, isotropic floor $\epsilon$) and any frequency band are just a re-weighted sum, so the UI responds instantly.

![app screenshot](figures/app_screenshot.png)

## Files

| file | contents |
|---|---|
| `simulation.py` | All numerics: grid/model, sponge, Ricker source, FD solver, analytic Green's function, source distribution, reciprocity correlations, travel-time picking, stationary-phase prediction, fake-anisotropy fit. No UI code, importable on its own. |
| `app.py` | Panel UI. It only wires widgets to `simulation.py`. |
| `verification.ipynb` | Tests of `simulation.py` against analytic solutions: 2D Green's function, convergence in Δx (order ≈ 4) and Δt (order ≈ 2), CFL stability limit, sponge reflections, Green's function retrieval from an isotropic ring, and directional bias vs an exact Hankel-function sum and stationary-phase theory. |
| `experiments.ipynb` | The geophysical results: bias vs κ and θ₀, vs station spacing r/λ, fake azimuthal anisotropy, effect of an isotropic background, and (exploratory) scattering. |

## Running it yourself

```bash
git clone https://github.com/zach-shallenberger/GPCOMP-ambient-directionality.git
cd GPCOMP-ambient-directionality
conda env create -f environment.yml     # isolated env named "ambient-noise"
conda activate ambient-noise
```

No conda? Use a virtual environment instead: `python -m venv .venv`, activate it (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), then `pip install -r requirements.txt`.

**Interactive app.** Run `panel serve app.py --show`. The page opens right away and starts the solver, with a progress bar in the sidebar (the terminal also prints `FD solver: NN%`). The default *fast* grid (Δx = 100 m) takes a few seconds; *standard* (Δx = 50 m, the grid the notebooks verify) takes about 15 s on a laptop. Each result is saved to `.cache/`, so a configuration you have already run reloads instantly, even after restarting the app. Once the plots appear, drag the noise-direction, concentration, background, and frequency sliders: these update instantly. Change the grid, spacing, or heterogeneity, then press *Run solver*.

**Notebooks.** Run `jupyter notebook`, then open `verification.ipynb` or `experiments.ipynb` and choose *Run All*. The experiments notebook takes a couple of minutes; the scattering section is the slowest part, because each medium needs new solver runs.

**From your own script.** Any experiment can be run in a few lines:

```python
import numpy as np, simulation as S

m = S.Model(c=S.random_medium(S.Model(), rel_std=0.02, corr_len=0.1, seed=1))  # or S.Model() for homogeneous
exp = S.PairExperiment(m, r=3.0)                 # stations 3 km apart; 2 FD runs
ref = exp.correlate(exp.weights(), fc=2.0)       # isotropic-noise reference
w = exp.weights(theta0=np.pi / 2, kappa=3.0, eps=0.0)   # broadside, concentrated noise
print(exp.measure(exp.correlate(w, fc=2.0), ref, fc=2.0))   # bias in s, dv/v in %
```

Change `rel_std`, `corr_len` (km) or `seed` to try other random media. Running several seeds and averaging them is the natural next step for the scattering study.

Units are km, s, and km/s throughout. Defaults (in `simulation.py`): c = 2.2 km/s, Ricker source at 2 Hz (λ = 1.1 km), Δx = 50 m, Δt ≈ 11 ms, a 701 × 701 grid, and about 500 ring sources.

## Main findings (see `experiments.ipynb`)
- The bias depends on the noise intensity *around the endfire directions* (the station axis), through the curvature $B''/B$ there (Weaver et al. 2009). One-sided noise along the axis still gives a nearly correct travel time on one branch. Broadside noise makes the basin look fast: about +3% for κ = 4 and +7% for κ = 6 with stations 3 km (2.7λ) apart.
- The bias scales as $(r/\lambda)^{-2}$, which is the reason behind the "r > 3λ" rule in practice. At 2 Hz, 1 km pairs (0.9λ) are badly biased and 5 km pairs (4.5λ) are nearly unaffected.
- Weak scattering can cancel the bias or flip its sign (exploratory, depends on the random realisation).
- One dominant noise direction produces a fake $2\theta$ azimuthal anisotropy with its fast axis perpendicular to the noise direction (≈ 0.5% for κ = 2, ≈ 1.8% for κ = 4 at 3 km spacing).

## AI-use statement
This project was developed with Claude (Anthropic) used as an AI coding agent through Claude Code. Claude helped draft the project plan and the solver, interferometry, and UI code and the notebooks, and ran the verification tests. I checked the physics and results and am responsible for them. *(Edit this statement to describe how you actually reviewed and changed the code.)*

## References
- Course notebooks GPGN 436/536 (Dr. Ge Jin, Dr. Jeff Shragge), especially 06_Hyperbolic_PDEs (FD wave solver) and 09_Regression (least-squares fit).
- Snieder, R. (2004). Extracting the Green's function from the correlation of coda waves: a derivation based on stationary phase. *Phys. Rev. E* 69, 046610.
- Wapenaar, K. (2004). Retrieving the elastodynamic Green's function of an arbitrary inhomogeneous medium by cross correlation. *PRL* 93, 254301.
- Weaver, R., Froment, B., & Campillo, M. (2009). On the correlation of non-isotropically distributed ballistic scalar diffuse waves. *JASA* 126, 1817–1826.
- Froment, B., et al. (2010). Estimation of the effect of nonisotropically distributed energy on the apparent arrival time in correlations. *Geophysics* 75, SA85–SA93.
- Stehly, L., Campillo, M., & Shapiro, N. M. (2006). A study of the seismic noise from its long-range correlation properties. *JGR* 111, B10306.
- Bensen, G. D., et al. (2007). Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements. *GJI* 169, 1239–1260.
- Libraries: NumPy, SciPy (`hankel1`, `i0e`, `gaussian_filter`), Matplotlib, Panel.
