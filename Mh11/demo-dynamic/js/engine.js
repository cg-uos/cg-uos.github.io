// Owns the two persistent GPU kernels (FIR prefilter, spline evaluation) for
// the currently selected (filter, resolution) and the query grid they
// evaluate onto (sized to track resolution -- see evalGridNFor below).
// Rebuilt on `configure()` (resolution or filter change -- infrequent);
// `step()` runs both every animation frame.

import { SQRT3 } from './hex.js';
import { buildFIRShader, buildEvalShader_Mh11, buildEvalShader_Mh20, buildEvalShader_Vhex4 } from './shaders.js';
import { GPUKernel } from './webgpu-context.js';
import { prefilters } from './prefilters.js';
import { SCALING } from './lattice.js';

// Xfm_coeff (../../web/js/reconstruct.js): its 2x2 block equals hex.G in
// index space; row-major nested-array layout, Xfm[row][col].
const XFM_COEFF = [
    [1, 1, 0],
    [-1, 1, 0],
    [0, 0, 1],
];

const EVAL_SHADER_BUILDERS = {
    Mh11: buildEvalShader_Mh11,
    Mh20: buildEvalShader_Mh20,
    Vhex4: buildEvalShader_Vhex4,
};

// Render-grid *density*: eval samples per unit of lattice spacing. Fixed, as
// opposed to a fixed grid *size* -- the physical domain shown grows with
// `core` (lattice.SCALING is constant at 1 unit/cell), so a fixed absolute
// grid size would sample that growing domain more and more coarsely
// relative to the lattice spacing as resolution increases. Since FIR
// sharpens the coefficients towards 1/phi_hat(w) (Blu & Unser), which
// amplifies high spatial frequencies most strongly against this demo's
// white-noise-like random coefficients, undersampling the render grid that
// way aliases into visible, resolution-dependent "noise" on the surface --
// worse at Fine/Very fine specifically because the undersampling is worst
// there. Scaling N with `core` keeps samples-per-lattice-spacing constant
// instead, so no new aliasing is introduced as resolution grows.
const EVAL_SAMPLES_PER_UNIT = 7;
const EVAL_GRID_N_MAX = 257; // bounds GPU/readback cost at the largest core

function evalGridNFor(core) {
    return Math.min(EVAL_GRID_N_MAX, 2 * core * EVAL_SAMPLES_PER_UNIT + 1);
}

// Builds the dense evaluation query grid over the lattice's visible "core"
// region, in the pre-transformed coordinate space the eval shaders expect
// (ptsRecon = xy/(2*scaling) + shift) -- ported from
// ../../web/js/reconstruct.js's buildEvalGrid.
function buildEvalGrid(core, i_min, j_min, N) {
    const xs = new Float32Array(N);
    const ys = new Float32Array(N);
    for (let a = 0; a < N; a++) {
        xs[a] = -core * SCALING + 2 * core * SCALING * (N === 1 ? 0 : a / (N - 1));
    }
    for (let b = 0; b < N; b++) {
        ys[b] = -core * SCALING * SQRT3 + 2 * core * SCALING * SQRT3 * (N === 1 ? 0 : b / (N - 1));
    }

    const shiftX = -i_min / 2;
    const shiftY = (-j_min * SQRT3) / 2;
    const numPts = N * N;
    const pts = new Float32Array(numPts * 2);
    let idx = 0;
    for (let row = 0; row < N; row++) {
        const y = ys[row];
        for (let col = 0; col < N; col++) {
            const x = xs[col];
            pts[2 * idx] = x / (2 * SCALING) + shiftX;
            pts[2 * idx + 1] = y / (2 * SCALING) + shiftY;
            idx++;
        }
    }
    return { xs, ys, pts, numPts };
}

export class Engine {
    constructor(device) {
        this.device = device;
        this.firKernel = null;
        this.evalKernel = null;
        this.grid = null;
        this.gridN = null;
        this.lattice = null;
    }

    destroy() {
        if (this.firKernel) this.firKernel.destroy();
        if (this.evalKernel) this.evalKernel.destroy();
        this.firKernel = null;
        this.evalKernel = null;
    }

    // lattice: the object returned by lattice.buildLattice(core).
    configure(filterKey, lattice) {
        this.destroy();
        this.filterKey = filterKey;
        this.lattice = lattice;

        const { nx, ny, core, i_min, j_min } = lattice;

        const firWGSL = buildFIRShader(prefilters[filterKey].FIR, nx, ny);
        this.firKernel = new GPUKernel(this.device, firWGSL);
        this.firKernel.configure([nx * ny * 4], [nx * ny * 4]);

        this.gridN = evalGridNFor(core);
        this.grid = buildEvalGrid(core, i_min, j_min, this.gridN);

        const buildEval = EVAL_SHADER_BUILDERS[filterKey];
        const evalWGSL = buildEval(nx, ny, XFM_COEFF, this.grid.numPts);
        this.evalKernel = new GPUKernel(this.device, evalWGSL);
        this.evalKernel.configure([nx * ny * 4, this.grid.numPts * 2 * 4], [this.grid.numPts * 4]);
    }

    // coeffRaw: Float32Array(nx*ny), the current frame's animated samples.
    // Returns Float32Array(gridN*gridN), row-major (row=y, col=x).
    async step(coeffRaw, applyFIR) {
        const { nx, ny } = this.lattice;
        let coeffSpline = coeffRaw;
        if (applyFIR) {
            const [out] = await this.firKernel.run([coeffRaw], [Math.ceil(nx / 16), Math.ceil(ny / 16), 1]);
            coeffSpline = out;
        }
        const numPts = this.grid.numPts;
        const [zRecon] = await this.evalKernel.run(
            [coeffSpline, this.grid.pts],
            [Math.ceil(numPts / 256), 1, 1]
        );
        return zRecon;
    }
}
