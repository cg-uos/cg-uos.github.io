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
const speedRange = document.getElementById('speedRange');
const speedValue = document.getElementById('speedValue');
const playPauseBtn = document.getElementById('playPauseBtn');
const regenerateBtn = document.getElementById('regenerateBtn');
const bannerEl = document.getElementById('banner');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fpsValue');

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

// Identifies the current animation-loop "epoch". A stale `frame()` call
// that was already mid-GPU-round-trip when `onConfigChange` ran (and so
// is still awaiting `engine.step()` under the *old* engine/lattice) checks
// this before rescheduling itself; since `onConfigChange` always mints a
// fresh token before starting its own loop, the stale chain sees a
// mismatch and simply stops, rather than going on to reschedule itself
// forever in parallel with the new loop.
let loopToken = null;

// FPS of actual completed compute+render cycles -- *not* of
// requestAnimationFrame callbacks. Those two differ whenever a GPU
// round-trip (dispatch + async mapAsync readback) takes longer than one
// vsync interval: rAF keeps firing at display refresh rate, but `busy`
// makes every such tick a no-op until the in-flight cycle finishes, so
// timing rAF ticks themselves would just report the display's refresh rate
// instead of how fast this demo is actually computing.
let lastStepTs = null;
let fpsEma = null;

// Resolves once no `frame()` call is mid-GPU-round-trip. Needed before
// `engine.configure()` destroys the current kernels' buffers: destroying a
// buffer that a just-started `frame()` is still awaiting `mapAsync()` on
// throws an unhandled AbortError ("Buffer was destroyed before mapping was
// resolved"), and would otherwise land in that stale frame()'s catch block
// and overwrite this reconfiguration's just-cleared status text.
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
    if (token !== loopToken) return; // superseded by a newer onConfigChange; let this chain die

    if (lastTs === null) lastTs = ts;
    const dt = (ts - lastTs) / 1000;
    lastTs = ts;
    if (playing) elapsed += dt * Number(speedRange.value);

    if (!busy) {
        busy = true;
        evaluateFrame(params, elapsed, coeffRaw);
        try {
            const zRecon = await engine.step(coeffRaw, firCheckbox.checked);
            busy = false;
            if (token !== loopToken) return; // config changed while this step was in flight

            for (const p of lattice.points) p.z = coeffRaw[p.offset];

            renderPlot('plotDiv', {
                xs: engine.grid.xs,
                ys: engine.grid.ys,
                zFlat: zRecon,
                N: engine.gridN,
                points: lattice.points,
                showSpline: showSplineCheckbox.checked,
                showCoefficients: showCoeffCheckbox.checked,
                title: PLOT_TITLES[filterSelect.value],
            });
            statusEl.textContent = '';

            const now = performance.now();
            if (lastStepTs !== null) {
                const stepDt = (now - lastStepTs) / 1000;
                const instFps = stepDt > 0 ? 1 / stepDt : 0;
                fpsEma = fpsEma === null ? instFps : fpsEma * 0.85 + instFps * 0.15;
                fpsEl.textContent = `${fpsEma.toFixed(1)} fps`;
            }
            lastStepTs = now;
        } catch (err) {
            busy = false;
            if (token !== loopToken) return; // destroyed-buffer abort from a reconfigure; not a real error
            console.error(err);
            statusEl.textContent = `Error: ${err.message}`;
            return; // stop the loop on a hard error rather than spin on it
        }
    }

    rafId = requestAnimationFrame((ts2) => frame(ts2, token));
}

async function onConfigChange() {
    const token = (loopToken = {}); // invalidate any still-running loop immediately
    if (rafId !== null) cancelAnimationFrame(rafId);
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
