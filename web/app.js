// app.js -- browser interface. It only reads the controls, talks to the Web Worker
// (worker.js, which runs the Python solver hvsr.py through Pyodide) and draws the results.
// No physics is computed here.

const $ = (id) => document.getElementById(id);
const MODEL_KEYS = ["vs1", "vs2", "rho_ratio", "h0", "dip", "q"];
const RUN_KEYS = [...MODEL_KEYS, "n_surface", "T", "dx"];
const FMT = { rho_ratio: (v) => Number(v).toFixed(2) };

// ------------------------------------------------------------------ worker plumbing
const worker = new Worker("worker.js");
let nextId = 1;
const pending = new Map();
let live = null;          // geometry for frames drawn while the solver runs
let replay = null;        // finished run: geometry + all frames
let replayTimer = null;

worker.onmessage = (e) => {
  const m = e.data;
  if (m.id !== undefined) {
    const p = pending.get(m.id); pending.delete(m.id);
    if (m.error) p.reject(new Error(m.error)); else p.resolve(m.result);
    return;
  }
  if (m.type === "status") $("status").textContent = m.text;
  else if (m.type === "progress") $("bar").value = m.value;
  else if (m.type === "frame" && live) drawWave(live, m.frame, m.t, "solver running", null);
  else if (m.type === "error") $("status").textContent = "Python failed to load: " + m.text;
  else if (m.type === "ready") {
    $("versions").textContent = `Running Python ${m.python}, NumPy ${m.numpy} and Pyodide ${m.pyodide} in this browser tab.`;
    $("run").disabled = false; $("run").textContent = "Run solver";
    $("status").textContent = "Python is ready.";
    updateInfo();
    runSolver();
  }
};

function call(cmd, args = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args });
  });
}

// ------------------------------------------------------------------ controls
function params() {
  const p = {};
  for (const k of RUN_KEYS) p[k] = Number($(k).value);
  return p;
}

for (const k of [...MODEL_KEYS, "n_surface", "b", "station"]) {
  const show = () => { $(k + "-v").textContent = (FMT[k] || String)($(k).value); };
  $(k).addEventListener("input", show); show();
}
for (const k of MODEL_KEYS) $(k).addEventListener("change", updateInfo);
$("b").addEventListener("change", refreshAnalysis);
$("station").addEventListener("input", () => {
  const j = Number($("station").value) - 1;
  $("station-v").textContent = `${j + 1} (x = ${-150 + 10 * j} m)`;
});
$("station").dispatchEvent(new Event("input"));
$("station").addEventListener("change", refreshAnalysis);
$("run").addEventListener("click", runSolver);

async function updateInfo() {
  const r = await call("info", { params: params() });
  $("contrast").innerHTML = `Impedance contrast ρ₂Vs₂/ρ₁Vs₁ = <b>${r.contrast.toFixed(1)}</b>; ` +
    `1D f₀ at x = 0: Vs₁/4H = <b>${r.f0.toFixed(2)} Hz</b>`;
}

// ------------------------------------------------------------------ solver run
function liveGeometry(p) {
  // the same snapshot grid as hvsr.simulate_noise: physical box +-300 m by 250 m
  const sd = Math.max(1, Math.round(4 / p.dx)), h = p.dx * sd;
  const xs = [], zs = [];
  for (let x = -300; x <= 300 + 1e-9; x += h) xs.push(x);
  for (let z = 0; z <= 250 + 1e-9; z += h) zs.push(z);
  const ix = xs.map((x) => x), iz = xs.map((x) => Math.max(p.h0 + Math.tan(p.dip * Math.PI / 180) * x, 8));
  return { x_snap: xs, z_snap: zs, nx: xs.length, nz: zs.length, iface_x: ix, iface_z: iz,
           rec_x: Array.from({ length: 31 }, (_, i) => -150 + 10 * i), vs1: p.vs1, vs2: p.vs2 };
}

async function runSolver() {
  const p = params();
  $("run").disabled = true; $("bar").hidden = false; $("bar").value = 0;
  $("status").textContent = "Running the SH solver: soft-layer model, then the bare-rock reference...";
  clearInterval(replayTimer); replay = null;
  live = liveGeometry(p);
  $("wave-note").textContent = "Solver running: live wavefield.";
  const t0 = performance.now();
  try {
    replay = await call("run", { params: p });
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    $("status").textContent = `Solver ready (${s} s): ${p.T} s of noise, Δx = ${p.dx} m, Δt = ${(1000 * replay.dt_solver).toFixed(2)} ms.`;
    startReplay();
    await refreshAnalysis();
  } catch (err) {
    $("status").textContent = "Solver failed: " + err.message;
  } finally {
    live = null; $("run").disabled = false; $("bar").hidden = true;
  }
}

function startReplay() {
  const r = replay, n = r.nx * r.nz, nt = r.frame_times.length;
  let i = 0;
  $("wave-note").textContent = `Horizontal (SH) ground motion from ${r.src_x.length} random noise sources in the rock (stars), ` +
    "recorded by 31 surface stations (green). The soft layer traps and amplifies it. Replaying the first 7 s in a loop.";
  replayTimer = setInterval(() => {
    drawWave(r, r.frames.subarray(i * n, (i + 1) * n), r.frame_times[i], "replay", r.lim);
    i = (i + 1) % nt;
  }, 100);
}

// ------------------------------------------------------------------ wavefield canvas
function seismic(v) {          // matplotlib "seismic": dark blue - blue - white - red - dark red
  const t = Math.max(-1, Math.min(1, v));
  if (t < -0.5) { const a = (t + 1) / 0.5; return [0, 0, 77 + 178 * a]; }
  if (t < 0) { const a = (t + 0.5) / 0.5; return [255 * a, 255 * a, 255]; }
  if (t < 0.5) { const a = t / 0.5; return [255, 255 * (1 - a), 255 * (1 - a)]; }
  const a = (t - 0.5) / 0.5; return [255 - 127 * a, 0, 0];
}

function drawWave(g, frame, t, what, lim) {
  const cv = $("wave"), ctx = cv.getContext("2d");
  const W = cv.width, Hc = cv.height, L = 70, R = 20, T = 40, B = 50;
  const pw = W - L - R, ph = Hc - T - B;
  if (!lim) {
    const a = Array.from(frame, Math.abs).sort((p, q) => p - q);
    lim = a[Math.floor(0.995 * (a.length - 1))] || 1;
  }
  const img = new ImageData(g.nx, g.nz);
  for (let k = 0; k < g.nx * g.nz; k++) {
    const [r, gg, b] = seismic(frame[k] / lim);
    img.data[4 * k] = r; img.data[4 * k + 1] = gg; img.data[4 * k + 2] = b; img.data[4 * k + 3] = 255;
  }
  const off = new OffscreenCanvas(g.nx, g.nz);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, Hc);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, L, T, pw, ph);
  const x0 = g.x_snap[0], x1 = g.x_snap[g.x_snap.length - 1], z1 = g.z_snap[g.z_snap.length - 1];
  const X = (x) => L + (x - x0) / (x1 - x0) * pw, Z = (z) => T + z / z1 * ph;
  // interface
  ctx.strokeStyle = "#000"; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < g.iface_x.length; i++) { const px = X(g.iface_x[i]), pz = Z(g.iface_z[i]); i ? ctx.lineTo(px, pz) : ctx.moveTo(px, pz); }
  ctx.stroke();
  // sources (replay only: their random positions come from the solver)
  ctx.fillStyle = "#000"; ctx.font = "16px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  if (g.src_x) for (let i = 0; i < g.src_x.length; i++) ctx.fillText("★", X(g.src_x[i]), Z(g.src_z[i]));
  // receivers
  ctx.fillStyle = "#0a8a2a";
  for (const x of g.rec_x) { const px = X(x); ctx.beginPath(); ctx.moveTo(px - 6, T - 9); ctx.lineTo(px + 6, T - 9); ctx.lineTo(px, T + 1); ctx.fill(); }
  // axes
  ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.strokeRect(L, T, pw, ph);
  ctx.fillStyle = "#222"; ctx.font = "14px sans-serif"; ctx.textBaseline = "top";
  for (let x = -300; x <= 300; x += 100) { ctx.fillText(String(x), X(x), T + ph + 6); }
  ctx.fillText("x (m)", L + pw / 2, T + ph + 26);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let z = 0; z <= z1 + 1e-9; z += 50) ctx.fillText(String(z), L - 8, Z(z));
  ctx.save(); ctx.translate(18, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
  ctx.fillText("depth (m)", 0, 0); ctx.restore();
  ctx.textAlign = "left"; ctx.font = "13px sans-serif";
  ctx.fillText(`soft layer Vs₁ = ${g.vs1} m/s`, L + 8, T + 14);
  ctx.fillText(`rock Vs₂ = ${g.vs2} m/s`, L + 8, T + ph - 14);
  ctx.textAlign = "center"; ctx.font = "15px sans-serif"; ctx.textBaseline = "alphabetic";
  ctx.fillText(`Ambient-noise wavefield u(x, z, t) [${what}], t = ${t.toFixed(2)} s`, L + pw / 2, 20);
}

// ------------------------------------------------------------------ analysis plots
const PLOT_CFG = { responsive: true, displaylogo: false };
const LOG_F = [Math.log10(0.6), Math.log10(9)];

async function refreshAnalysis() {
  if (!replay) return;
  const b = Number($("b").value), j = Number($("station").value) - 1;
  const a = await call("analysis", { b });
  const xj = a.x[j];
  const z = [];
  for (let i = 0; i < a.nf; i++) z.push(Array.from(a.R.subarray(i * a.nx, (i + 1) * a.nx)));

  Plotly.react("ratio-plot", [
    { type: "heatmap", x: Array.from(a.x), y: Array.from(a.f), z, colorscale: "Viridis", zmin: 1, zmax: a.vmax,
      colorbar: { title: { text: "ratio" } }, hovertemplate: "x %{x} m<br>f %{y:.2f} Hz<br>ratio %{z:.2f}<extra></extra>" },
    { x: Array.from(a.xx), y: Array.from(a.f0_1d), mode: "lines", name: "1D: Vs₁ / 4H(x)", line: { color: "#fff", dash: "dash", width: 2 } },
    { x: Array.from(a.x), y: Array.from(a.f0), mode: "markers", name: "picked f₀", marker: { color: "red", size: 7, line: { color: "#fff", width: 1 } } },
  ], {
    title: { text: "H/V proxy (site / rock spectral ratio) along the line", font: { size: 14 } },
    xaxis: { title: { text: "receiver x (m)" } },
    yaxis: { title: { text: "frequency (Hz)" }, type: "log", range: LOG_F },
    shapes: [{ type: "line", x0: xj, x1: xj, yref: "paper", y0: 0, y1: 1, line: { color: "#fff", width: 1 } }],
    legend: { x: 0.55, y: 0.98, bgcolor: "rgba(255,255,255,0.8)" }, margin: { l: 60, r: 10, t: 40, b: 50 },
  }, PLOT_CFG);

  const xin = [], f0in = [], hin = [];
  a.inband.forEach((ok, i) => { if (ok) { xin.push(a.x[i]); f0in.push(a.f0[i]); hin.push(a.H_implied[i]); } });
  Plotly.react("f0-plot", [
    { x: Array.from(a.xx), y: Array.from(a.f0_1d), mode: "lines", name: "1D Vs₁ / 4H(x)", line: { color: "#000" } },
    { x: xin, y: f0in, mode: "markers", name: "picked f₀", marker: { color: "#d62728", size: 6 } },
    { x: Array.from(a.xx), y: Array.from(a.H_true), mode: "lines", name: "true H(x)", line: { color: "#555" }, xaxis: "x2", yaxis: "y2" },
    { x: xin, y: hin, mode: "markers", name: "Vs₁ / 4f₀", marker: { color: "#9467bd", size: 6 }, xaxis: "x2", yaxis: "y2" },
  ], {
    title: { text: `f₀ and the thickness it implies (mean f₀ misfit ${a.err_mean >= 0 ? "+" : ""}${a.err_mean.toFixed(1)} %)`, font: { size: 14 } },
    grid: { rows: 2, columns: 1, pattern: "independent" },
    yaxis: { title: { text: "f₀ (Hz)" }, type: "log" },
    xaxis2: { title: { text: "receiver x (m)" }, matches: "x" },
    yaxis2: { title: { text: "H (m)" }, autorange: "reversed" },
    shapes: [{ type: "line", x0: xj, x1: xj, yref: "paper", y0: 0, y1: 1, line: { color: "#999", dash: "dot" } }],
    legend: { orientation: "h", y: -0.28 }, margin: { l: 60, r: 10, t: 40, b: 70 },
  }, PLOT_CFG);

  await refreshStation(b, j);
}

function tile(name, val, unit, sub, color) {
  return `<div class="tile" style="--c:${color}"><div class="name">${name}</div>` +
    `<div class="val">${val}<span> ${unit}</span></div><div class="sub">${sub}</div></div>`;
}

async function refreshStation(b, j) {
  const s = await call("station", { b, j });
  $("station-title").textContent = `H/V curve and site prediction at station ${j + 1} (x = ${s.x.toFixed(0)} m)`;
  const top = Math.max(1.15 * Math.max(...s.R), 1.1 * s.A_true);
  Plotly.react("station-plot", [
    { x: Array.from(s.f), y: Array.from(s.R), mode: "lines", name: "H/V proxy from the noise", line: { color: "#1f77b4", width: 3 } },
    { x: Array.from(s.ff), y: Array.from(s.tf), mode: "lines", name: `true 1D amplification (H = ${s.H_true.toFixed(1)} m)`, line: { color: "#000", dash: "dash" } },
    { x: [s.f0], y: [s.A0], mode: "markers", name: `picked f₀ = ${s.f0.toFixed(2)} Hz`, marker: { color: "#d62728", size: 9 } },
  ], {
    xaxis: { title: { text: "frequency (Hz)" }, type: "log", range: LOG_F },
    yaxis: { title: { text: "amplification" }, range: [0, top] },
    shapes: [{ type: "line", x0: s.f0, x1: s.f0, yref: "paper", y0: 0, y1: 1, line: { color: "#d62728", width: 1 } }],
    legend: { orientation: "h", y: -0.25 }, margin: { l: 60, r: 10, t: 10, b: 60 },
  }, PLOT_CFG);
  const color = { high: "#d62728", moderate: "#ff7f0e", low: "#2ca02c" }[s.risk];
  const dH = 100 * (s.H_pred / s.H_true - 1);
  $("tiles").innerHTML = [
    tile("Resonance frequency f₀", s.f0.toFixed(2), "Hz", `true 1D ${s.f_true.toFixed(2)} Hz`, "#1f77b4"),
    tile("Predicted peak amplification", "×" + s.A0.toFixed(1), "", `true 1D ×${s.A_true.toFixed(1)}; ${s.risk} amplification`, color),
    tile("Depth to rock, Vs₁/4f₀", s.H_pred.toFixed(0), "m", `true ${s.H_true.toFixed(1)} m (${dH >= 0 ? "+" : ""}${dH.toFixed(0)} %)`, "#9467bd"),
    tile("Buildings at risk of resonance", "~" + s.floors, "storeys", `natural period ≈ T₀ = ${s.T0.toFixed(2)} s`, color),
    tile("Vs30 from the H/V depth", s.vs30_pred.toFixed(0), "m/s", `class ${s.class_pred}; true ${s.vs30_true.toFixed(0)} m/s, class ${s.class_true}`, "#8c564b"),
  ].join("");
}
