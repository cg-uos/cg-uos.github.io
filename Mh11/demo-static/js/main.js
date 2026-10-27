import { functions } from './functions.js';
import { initWebGPU } from './webgpu-context.js';
import { reconstruct } from './reconstruct.js';
import { makeSurface, MATH_TITLE_FONT } from './plot.js';

const PLOT_TITLES = {
    Mh11: '<i>M<sub>h11</sub></i>',
    Vhex4: '<i>V<sub>4</sub></i>',
    Mh20: '<i>M<sub>h20</sub></i>',
};

const BASIS_OPTION = 'delta (basis)';

const funcSelect = document.getElementById('funcSelect');
const scalingRange = document.getElementById('scalingRange');
const scalingValue = document.getElementById('scalingValue');
const prefilterSelect = document.getElementById('prefilterSelect');
const lightShadingCheckbox = document.getElementById('lightShadingCheckbox');
const reconstructBtn = document.getElementById('reconstructBtn');
const statusEl = document.getElementById('status');
const bannerEl = document.getElementById('banner');
const plotGrid = document.getElementById('plotGrid');
const panelOriginal = document.getElementById('panelOriginal');
const captionMh11 = document.getElementById('captionMh11');
const captionVhex4 = document.getElementById('captionVhex4');
const captionMh20 = document.getElementById('captionMh20');

const NORMAL_CAPTIONS = {
    Mh11: '<span class="math">M<sub>h11</sub></span> <span id="errMh11"></span>',
    Vhex4: '<span class="math">V<sub>4</sub></span> <span id="errVhex4"></span>',
    Mh20: '<span class="math">M<sub>h20</sub></span> <span id="errMh20"></span>',
};
const BASIS_CAPTIONS = {
    Mh11: '<span class="math">M<sub>h11</sub></span> basis',
    Vhex4: '<span class="math">V<sub>4</sub></span> basis',
    Mh20: '<span class="math">M<sub>h20</sub></span> basis',
};

for (const name of Object.keys(functions)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    funcSelect.appendChild(opt);
}
funcSelect.value = 'Saddle';

scalingRange.addEventListener('input', () => {
    scalingValue.textContent = Number(scalingRange.value).toFixed(2);
});

// Selecting "delta (basis)" routes through the same reconstruct() pipeline
// (see that entry in functions.js): `fixedScaling` overrides the UI's
// density control, so it's disabled here since it'd have no effect, but the
// prefilter is a real, user-visible choice for this entry too (FIR/IIR2
// prefiltering of a delta shows the prefilter's own impulse response rather
// than the raw basis function). There's no sampled function and no
// "Original" panel for it either way.
function updateBasisModeUI() {
    const basisMode = funcSelect.value === BASIS_OPTION;
    scalingRange.disabled = basisMode;
    panelOriginal.hidden = basisMode;
    plotGrid.classList.toggle('basis-mode', basisMode);
    captionMh11.innerHTML = basisMode ? BASIS_CAPTIONS.Mh11 : NORMAL_CAPTIONS.Mh11;
    captionVhex4.innerHTML = basisMode ? BASIS_CAPTIONS.Vhex4 : NORMAL_CAPTIONS.Vhex4;
    captionMh20.innerHTML = basisMode ? BASIS_CAPTIONS.Mh20 : NORMAL_CAPTIONS.Mh20;
}
funcSelect.addEventListener('change', updateBasisModeUI);
updateBasisModeUI();

let device = null;

// Cached so toggling "Light shading" can just redraw the existing surfaces
// instead of re-running the whole sample/prefilter/evaluate pipeline for a
// change that's purely about how the result is rendered.
let lastResult = null;
let lastBasisMode = false;

function renderResult(result, basisMode) {
    const lightShading = lightShadingCheckbox.checked;

    if (!basisMode) {
        makeSurface('plotOriginal', result.xs, result.ys, result.zTrue, result.Nx, result.Ny, 'Original', null, null, lightShading);
    }
    makeSurface(
        'plotMh11', result.xs, result.ys, result.results.Mh11.zRecon, result.Nx, result.Ny, PLOT_TITLES.Mh11,
        result.results.Mh11.supportPolygon, MATH_TITLE_FONT, lightShading
    );
    makeSurface(
        'plotVhex4', result.xs, result.ys, result.results.Vhex4.zRecon, result.Nx, result.Ny, PLOT_TITLES.Vhex4,
        result.results.Vhex4.supportPolygon, MATH_TITLE_FONT, lightShading
    );
    makeSurface(
        'plotMh20', result.xs, result.ys, result.results.Mh20.zRecon, result.Nx, result.Ny, PLOT_TITLES.Mh20,
        result.results.Mh20.supportPolygon, MATH_TITLE_FONT, lightShading
    );

    if (!basisMode) {
        document.getElementById('errMh11').textContent = `(mean |err| = ${result.results.Mh11.meanAbsError.toExponential(3)})`;
        document.getElementById('errVhex4').textContent = `(mean |err| = ${result.results.Vhex4.meanAbsError.toExponential(3)})`;
        document.getElementById('errMh20').textContent = `(mean |err| = ${result.results.Mh20.meanAbsError.toExponential(3)})`;
    }
}

lightShadingCheckbox.addEventListener('change', () => {
    if (lastResult) renderResult(lastResult, lastBasisMode);
});

async function run() {
    reconstructBtn.disabled = true;
    statusEl.textContent = 'Reconstructing…';
    try {
        const basisMode = funcSelect.value === BASIS_OPTION;
        const result = await reconstruct(device, {
            funcName: funcSelect.value,
            scaling: Number(scalingRange.value),
            prefilterScheme: prefilterSelect.value,
        });

        lastResult = result;
        lastBasisMode = basisMode;
        renderResult(result, basisMode);

        statusEl.textContent = '';
    } catch (err) {
        console.error(err);
        statusEl.textContent = `Error: ${err.message}`;
    } finally {
        reconstructBtn.disabled = false;
    }
}

async function setup() {
    try {
        device = await initWebGPU();
    } catch (err) {
        bannerEl.textContent = err.message;
        bannerEl.hidden = false;
        reconstructBtn.disabled = true;
        return;
    }
    await run();
}

reconstructBtn.addEventListener('click', run);
setup();
