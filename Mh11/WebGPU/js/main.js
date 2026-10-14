import { functions } from './functions.js';
import { initWebGPU } from './webgpu-context.js';
import { reconstruct, reconstructBasis } from './reconstruct.js';
import { makeSurface, MATH_TITLE_FONT } from './plot.js';

const PLOT_TITLES = {
    Mh11: '<i>M<sub>h11</sub></i>',
    Vhex4: '<i>V<sub>4</sub></i>',
    Mh20: '<i>M<sub>h20</sub></i>',
};

const BASIS_OPTION = 'Basis';

const funcSelect = document.getElementById('funcSelect');
const scalingRange = document.getElementById('scalingRange');
const scalingValue = document.getElementById('scalingValue');
const prefilterSelect = document.getElementById('prefilterSelect');
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
const basisOpt = document.createElement('option');
basisOpt.value = BASIS_OPTION;
basisOpt.textContent = 'Basis (no prefilter)';
funcSelect.appendChild(basisOpt);
funcSelect.value = 'Saddle';

scalingRange.addEventListener('input', () => {
    scalingValue.textContent = Number(scalingRange.value).toFixed(2);
});

// Selecting "Basis" from the Function dropdown replaces the whole pipeline
// (no function sampled, no prefilter applied -- see reconstructBasis in
// reconstruct.js), so the density/prefilter controls that only affect the
// normal reconstruction pipeline are disabled, and there's no "Original"
// panel.
function updateBasisModeUI() {
    const basisMode = funcSelect.value === BASIS_OPTION;
    scalingRange.disabled = basisMode;
    prefilterSelect.disabled = basisMode;
    panelOriginal.hidden = basisMode;
    plotGrid.classList.toggle('basis-mode', basisMode);
    captionMh11.innerHTML = basisMode ? BASIS_CAPTIONS.Mh11 : NORMAL_CAPTIONS.Mh11;
    captionVhex4.innerHTML = basisMode ? BASIS_CAPTIONS.Vhex4 : NORMAL_CAPTIONS.Vhex4;
    captionMh20.innerHTML = basisMode ? BASIS_CAPTIONS.Mh20 : NORMAL_CAPTIONS.Mh20;
}
funcSelect.addEventListener('change', updateBasisModeUI);
updateBasisModeUI();

let device = null;

async function run() {
    reconstructBtn.disabled = true;
    statusEl.textContent = 'Reconstructing…';
    try {
        const basisMode = funcSelect.value === BASIS_OPTION;
        const result = basisMode
            ? await reconstructBasis(device, {})
            : await reconstruct(device, {
                  funcName: funcSelect.value,
                  scaling: Number(scalingRange.value),
                  prefilterScheme: prefilterSelect.value,
              });

        if (!basisMode) {
            makeSurface('plotOriginal', result.xs, result.ys, result.zTrue, result.Nx, result.Ny, 'Original');
        }
        makeSurface(
            'plotMh11', result.xs, result.ys, result.results.Mh11.zRecon, result.Nx, result.Ny, PLOT_TITLES.Mh11,
            result.results.Mh11.supportPolygon, MATH_TITLE_FONT
        );
        makeSurface(
            'plotVhex4', result.xs, result.ys, result.results.Vhex4.zRecon, result.Nx, result.Ny, PLOT_TITLES.Vhex4,
            result.results.Vhex4.supportPolygon, MATH_TITLE_FONT
        );
        makeSurface(
            'plotMh20', result.xs, result.ys, result.results.Mh20.zRecon, result.Nx, result.Ny, PLOT_TITLES.Mh20,
            result.results.Mh20.supportPolygon, MATH_TITLE_FONT
        );

        if (!basisMode) {
            document.getElementById('errMh11').textContent = `(mean |err| = ${result.results.Mh11.meanAbsError.toExponential(3)})`;
            document.getElementById('errVhex4').textContent = `(mean |err| = ${result.results.Vhex4.meanAbsError.toExponential(3)})`;
            document.getElementById('errMh20').textContent = `(mean |err| = ${result.results.Mh20.meanAbsError.toExponential(3)})`;
        }

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
