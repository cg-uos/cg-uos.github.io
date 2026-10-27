import { initWebGPU } from './webgpu-context.js';
import { Engine } from './engine.js';
import { buildLattice, randomizeParams, evaluateFrame, RESOLUTIONS } from './lattice.js';
import { renderPlot } from './plot.js';

const PLOT_TITLES = {
    Mh11: '<i>M<sub>h11</sub></i>',
    Mh20: '<i>M<sub>h20</sub></i>',
    Vhex4: '<i>V<sub>4</sub></i>',
};

const filterSelect = document.getElementById('filterSelect');
const resolutionSelect = document.getElementById('resolutionSelect');
const firCheckbox = document.getElementById('firCheckbox');
const showCoeffCheckbox = document.getElementById('showCoeffCheckbox');
const showSplineCheckbox = document.getElementById('showSplineCheckbox');
const showLatticeCheckbox = document.getElementById('showLatticeCheckbox');
const lightShadingCheckbox = document.getElementById('lightShadingCheckbox');
const speedRange = document.getElementById('speedRange');
const speedValue = document.getElementById('speedValue');
const playPauseBtn = document.getElementById('playPauseBtn');
const regenerateBtn = document.getElementById('regenerateBtn');
const bannerEl = document.getElementById('banner');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fpsValue');
const plotDiv = document.getElementById('plotDiv');

for (const r of RESOLUTIONS) {
    const opt = document.createElement('option');
    opt.value = String(r.core);
    opt.textContent = r.label;
    resolutionSelect.appendChild(opt);
}
resolutionSelect.value = String(RESOLUTIONS[1].core); // Medium (17x17) by default

speedRange.addEventListener('input', () => {
    speedValue.textContent = `${Number(speedRange.value).toFixed(1)}×`;
});

let device = null;
let engine = null;
let lattice = null;
let params = null;
let coeffRaw = null;

let playing = true;
let elapsed = 0;
let lastTs = null;
let rafId = null;
let busy = false; // guards against overlapping GPU round-trips if a frame is slow

// Tracks whether the user is mid-drag on the plot, so the animation loop
// can stop scheduling itself entirely during a drag (see `frame()`) and
// restart on mouseup/touchend via `endDrag()`.
let dragging = false;
let loopSuspended = false;
plotDiv.addEventListener('mousedown', () => { dragging = true; });
plotDiv.addEventListener('touchstart', () => { dragging = true; }, { passive: true });
function endDrag() {
    dragging = false;
    if (loopSuspended) {
        loopSuspended = false;
        rafId = requestAnimationFrame((ts) => frame(ts, loopToken));
    }
}
window.addEventListener('mouseup', endDrag);
window.addEventListener('touchend', endDrag);

// Holds the camera orientation from Plotly's 'plotly_relayout' event, fed
// into the next `renderPlot()` call (its `camera` option) so the user's
// rotation survives the drag-pause above.
let liveCamera = null;
let relayoutListenerAttached = false;
function attachRelayoutListenerOnce() {
    if (relayoutListenerAttached || typeof plotDiv.on !== 'function') return;
    relayoutListenerAttached = true;
    plotDiv.on('plotly_relayout', (eventData) => {
        if (eventData && eventData['scene.camera']) {
            liveCamera = eventData['scene.camera'];
        }
    });
}

// Identifies the current animation-loop "epoch", so a stale `frame()`
// chain from before a reconfigure stops instead of rescheduling itself
// alongside the new one.
let loopToken = null;

// Tracks FPS of completed compute+render cycles, not of
// requestAnimationFrame callbacks (which can tick faster than the GPU
// round-trip actually completes).
let lastStepTs = null;
let fpsEma = null;

// Resolves once no `frame()` call is mid-GPU-round-trip, so
// `engine.configure()` can safely destroy the current buffers.
function waitUntilIdle() {
    return new Promise((resolve) => {
        (function poll() {
            if (!busy) resolve();
            else setTimeout(poll, 5);
        })();
    });
}

function rebuildLattice() {
    const core = Number(resolutionSelect.value);
    lattice = buildLattice(core);
    params = randomizeParams(lattice);
    coeffRaw = new Float32Array(lattice.nx * lattice.ny);
}

async function frame(ts, token) {
    if (token !== loopToken) return; // a second onConfigChange ran before this chain's own reconfigure finished; let it die

    if (lastTs === null) lastTs = ts;
    const dt = (ts - lastTs) / 1000;
    lastTs = ts;
    if (playing) elapsed += dt * Number(speedRange.value);

    if (!busy && !dragging) {
        busy = true;
        evaluateFrame(params, elapsed, coeffRaw);
        try {
            const zRecon = await engine.step(coeffRaw, firCheckbox.checked);
            busy = false;
            // Re-check `dragging`: it can turn true while this step was
            // already in flight.
            if (token === loopToken && !dragging) {
                for (const p of lattice.points) p.z = coeffRaw[p.offset];

                renderPlot('plotDiv', {
                    xs: engine.grid.xs,
                    ys: engine.grid.ys,
                    zFlat: zRecon,
                    N: engine.gridN,
                    points: lattice.points,
                    edges: lattice.edges,
                    showSpline: showSplineCheckbox.checked,
                    showCoefficients: showCoeffCheckbox.checked,
                    showLattice: showLatticeCheckbox.checked,
                    lightShading: lightShadingCheckbox.checked,
                    camera: liveCamera,
                    title: PLOT_TITLES[filterSelect.value],
                });
                attachRelayoutListenerOnce();
                statusEl.textContent = '';

                const now = performance.now();
                if (lastStepTs !== null) {
                    const stepDt = (now - lastStepTs) / 1000;
                    const instFps = stepDt > 0 ? 1 / stepDt : 0;
                    fpsEma = fpsEma === null ? instFps : fpsEma * 0.85 + instFps * 0.15;
                    fpsEl.textContent = `${fpsEma.toFixed(1)} fps`;
                }
                lastStepTs = now;
            }
        } catch (err) {
            busy = false;
            if (token !== loopToken) return; // destroyed-buffer abort from a reconfigure; not a real error
            console.error(err);
            statusEl.textContent = `Error: ${err.message}`;
            return; // stop the loop on a hard error rather than spin on it
        }
    }

    if (token !== loopToken) return; // config changed while the step above was in flight

    // Stop scheduling the next tick while a drag is in progress;
    // `endDrag()` restarts the chain once the drag ends.
    if (dragging) {
        loopSuspended = true;
        return;
    }
    rafId = requestAnimationFrame((ts2) => frame(ts2, token));
}

async function onConfigChange() {
    const token = (loopToken = {}); // invalidate any still-running loop immediately
    if (rafId !== null) cancelAnimationFrame(rafId);
    // Clear any pending drag-suspend: a fresh loop starts below regardless,
    // and a later mouseup/touchend shouldn't start a second one.
    loopSuspended = false;
    filterSelect.disabled = true;
    resolutionSelect.disabled = true;
    statusEl.textContent = 'Rebuilding GPU pipeline…';

    await waitUntilIdle();
    rebuildLattice();
    engine.configure(filterSelect.value, lattice);

    statusEl.textContent = '';
    filterSelect.disabled = false;
    resolutionSelect.disabled = false;
    lastTs = null;
    lastStepTs = null;
    fpsEma = null;
    fpsEl.textContent = '-- fps';
    rafId = requestAnimationFrame((ts) => frame(ts, token));
}

filterSelect.addEventListener('change', onConfigChange);
resolutionSelect.addEventListener('change', onConfigChange);

regenerateBtn.addEventListener('click', () => {
    params = randomizeParams(lattice);
});

playPauseBtn.addEventListener('click', () => {
    playing = !playing;
    playPauseBtn.textContent = playing ? 'Pause' : 'Play';
});

async function setup() {
    try {
        device = await initWebGPU();
    } catch (err) {
        bannerEl.textContent = err.message;
        bannerEl.hidden = false;
        filterSelect.disabled = true;
        resolutionSelect.disabled = true;
        return;
    }
    engine = new Engine(device);
    await onConfigChange();
}

setup();
