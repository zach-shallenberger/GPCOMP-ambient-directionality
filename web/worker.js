// worker.js -- Web Worker that runs the Python simulation (hvsr.py) with Pyodide.
//
// The page (app.js) never runs Python itself: it posts {id, cmd, args} messages here and
// gets {id, result} back, so the interface stays responsive while the solver runs.
// Messages without an id are progress updates: {type: "status" | "progress" | "frame"}.

const PYODIDE_VERSION = "0.27.2";   // also listed in README.md and web/versions.json
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
importScripts(INDEX_URL + "pyodide.js");

let bridge = null;

async function fetchText(candidates) {
  // On GitHub Pages hvsr.py sits next to this file (the workflow copies it there);
  // in a local checkout it is one folder up, in the repository root.
  for (const url of candidates) {
    const r = await fetch(url, { cache: "no-cache" });
    if (r.ok) return r.text();
  }
  throw new Error("could not load " + candidates.join(" or "));
}

async function init() {
  postMessage({ type: "status", text: `Loading Python ${PYODIDE_VERSION} (Pyodide) in the browser...` });
  const py = await loadPyodide({ indexURL: INDEX_URL });
  postMessage({ type: "status", text: "Loading NumPy..." });
  await py.loadPackage("numpy");
  py.FS.writeFile("hvsr.py", await fetchText(["./hvsr.py", "../hvsr.py"]));
  py.FS.writeFile("bridge.py", await fetchText(["./bridge.py"]));
  await py.runPythonAsync("import sys; sys.path.insert(0, '.')");
  bridge = py.pyimport("bridge");
  postMessage({ type: "ready", python: py.runPython("import sys; sys.version.split()[0]"),
                numpy: py.runPython("import numpy; numpy.__version__"), pyodide: PYODIDE_VERSION });
}

const ready = init().catch((err) => postMessage({ type: "error", text: String(err) }));

function progress(q) { postMessage({ type: "progress", value: q }); }
function onFrame(t, frame) { postMessage({ type: "frame", t, frame }, [frame.buffer]); }

onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  await ready;
  try {
    let result;
    if (cmd === "info") result = bridge.model_info(args.params);
    else if (cmd === "run") result = bridge.run(args.params, progress, onFrame);
    else if (cmd === "analysis") result = bridge.analysis(args.b);
    else if (cmd === "station") result = bridge.station(args.b, args.j);
    else throw new Error("unknown command " + cmd);
    const transfer = result && result.frames ? [result.frames.buffer] : [];
    postMessage({ id, result }, transfer);
  } catch (err) {
    postMessage({ id, error: String(err) });
  }
};
