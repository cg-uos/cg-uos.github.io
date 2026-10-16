// Orchestrates the sample -> prefilter -> evaluate pipeline for all three
// spline families, mirroring demo_reconstruct_analytic_function.py's
// pipeline (python/demo_reconstruct_analytic_function.py:64-177).

import { sampleFunction, SQRT3 } from './hex.js';
import { functions } from './functions.js';
import { prefilters } from './prefilters.js';
import { applyIirCheckerboard } from './iir2.js';
import {
    buildFIRShader,
    buildEvalShader_Mh11,
    buildEvalShader_Mh20,
    buildEvalShader_Vhex4,
} from './shaders.js';
import { runCompute } from './webgpu-context.js';
import { supportPolygon } from './support.js';

// Xfm_coeff (demo_reconstruct_analytic_function.py:95): its 2x2 block equals
// hex.G in index space; row-major nested-array layout, Xfm[row][col].
const XFM_COEFF = [
    [1, 1, 0],
    [-1, 1, 0],
    [0, 0, 1],
];

const FAMILIES = [
    { key: 'Mh11', label: 'Mh11', buildEvalShader: buildEvalShader_Mh11 },
    { key: 'Mh20', label: 'Mh20', buildEvalShader: buildEvalShader_Mh20 },
    { key: 'Vhex4', label: 'Vhex4', buildEvalShader: buildEvalShader_Vhex4 },
];

async function runFIRPrefilterGPU(device, FIRcoeffs, z, nx, ny) {
    const wgsl = buildFIRShader(FIRcoeffs, nx, ny);
    const dispatch = [Math.ceil(nx / 16), Math.ceil(ny / 16), 1];
    const [out] = await runCompute(device, wgsl, [z], [z.byteLength], dispatch);
    return out;
}

async function runEvalGPU(device, buildEvalShader, coeff, nx, ny, pts, numPts) {
    const wgsl = buildEvalShader(nx, ny, XFM_COEFF, numPts);
    const dispatch = [Math.ceil(numPts / 256), 1, 1];
    const [out] = await runCompute(device, wgsl, [coeff, pts], [numPts * 4], dispatch);
    return out;
}

// Builds the (x,y) evaluation grid over `domain` and its corresponding
// reference-space query points for a coefficient array whose lattice origin
// is (i_min,j_min). Shared by reconstruct() and reconstructBasis().
// xy_recon = xy/(2*scaling) + [-i_min/2, -j_min*sqrt(3)/2]
// (demo_reconstruct_analytic_function.py:98). Matches
// np.meshgrid(linspace_x, linspace_y) with default 'xy' indexing: x varies
// fastest (inner loop), y is the outer (row) axis.
function buildEvalGrid(domain, scaling, i_min, j_min, Nx, Ny) {
    const xs = new Float32Array(Nx);
    const ys = new Float32Array(Ny);
    for (let a = 0; a < Nx; a++) {
        xs[a] = domain.xmin + (domain.xmax - domain.xmin) * (Nx === 1 ? 0 : a / (Nx - 1));
    }
    for (let b = 0; b < Ny; b++) {
        ys[b] = domain.ymin + (domain.ymax - domain.ymin) * (Ny === 1 ? 0 : b / (Ny - 1));
    }

    const shiftX = -i_min / 2;
    const shiftY = (-j_min * SQRT3) / 2;
    const numPts = Nx * Ny;
    const ptsRecon = new Float32Array(numPts * 2);
    let idx = 0;
    for (let row = 0; row < Ny; row++) {
        const y = ys[row];
        for (let col = 0; col < Nx; col++) {
            const x = xs[col];
            ptsRecon[2 * idx] = x / (2 * scaling) + shiftX;
            ptsRecon[2 * idx + 1] = y / (2 * scaling) + shiftY;
            idx++;
        }
    }
    return { xs, ys, ptsRecon, numPts };
}

// options: { funcName, scaling, prefilterScheme: 'FIR'|'IIR2', Nx, Ny }
export async function reconstruct(device, options) {
    const { funcName, scaling, prefilterScheme, Nx = 100, Ny = 100 } = options;
    const { fn, domain } = functions[funcName];

    // margin = scaling*9 (demo_reconstruct_analytic_function.py:14,69)
    const margin = scaling * 9;
    const sampleDomain = [
        domain.xmin - margin,
        domain.ymin - margin,
        domain.xmax + margin,
        domain.ymax + margin,
    ];

    const { i_min, j_min, nx, ny, z } = sampleFunction(fn, sampleDomain, scaling);
    const { xs, ys, ptsRecon, numPts } = buildEvalGrid(domain, scaling, i_min, j_min, Nx, Ny);

    const zTrue = new Float32Array(numPts);
    let idx = 0;
    for (let row = 0; row < Ny; row++) {
        for (let col = 0; col < Nx; col++) {
            zTrue[idx] = fn(xs[col], ys[row]);
            idx++;
        }
    }

    const results = {};
    for (const family of FAMILIES) {
        const pf = prefilters[family.key][prefilterScheme];
        const coeff =
            prefilterScheme === 'FIR'
                ? await runFIRPrefilterGPU(device, pf, z, nx, ny)
                : applyIirCheckerboard(z, nx, ny, pf);

        const zRecon = await runEvalGPU(
            device,
            family.buildEvalShader,
            coeff,
            nx,
            ny,
            ptsRecon,
            numPts
        );

        let errSum = 0;
        for (let k = 0; k < numPts; k++) errSum += Math.abs(zTrue[k] - zRecon[k]);
        const meanAbsError = errSum / numPts;

        results[family.key] = { zRecon, meanAbsError, label: family.label };
    }

    return { Nx, Ny, xs, ys, zTrue, results, domain };
}

// Plots each family's raw basis function: the spline evaluated from a single
// unit coefficient at the lattice origin with every other coefficient zero,
// and no prefilter applied at all. Reuses sampleFunction with a delta
// "function" so the coefficient array is built through the exact same
// checkerboard-lattice machinery as reconstruct(), rather than hand-deriving
// the index mapping separately.
const BASIS_DOMAIN = { xmin: -2, ymin: -2, xmax: 2, ymax: 2 };
const BASIS_SCALING = 0.2;

function deltaFn(x, y) {
    return x === 0 && y === 0 ? 1 : 0;
}

export async function reconstructBasis(device, options = {}) {
    const { Nx = 100, Ny = 100 } = options;
    const domain = BASIS_DOMAIN;
    const scaling = BASIS_SCALING;

    // Small safety margin so no coefficient within the display domain is
    // edge-clamped by sampleFunction's bounding box.
    const margin = scaling * 4;
    const sampleDomain = [
        domain.xmin - margin,
        domain.ymin - margin,
        domain.xmax + margin,
        domain.ymax + margin,
    ];

    const { i_min, j_min, nx, ny, z } = sampleFunction(deltaFn, sampleDomain, scaling);
    const { xs, ys, ptsRecon, numPts } = buildEvalGrid(domain, scaling, i_min, j_min, Nx, Ny);

    const results = {};
    for (const family of FAMILIES) {
        const zRecon = await runEvalGPU(device, family.buildEvalShader, z, nx, ny, ptsRecon, numPts);
        results[family.key] = {
            zRecon,
            label: family.label,
            supportPolygon: supportPolygon(family.key, scaling),
        };
    }

    return { Nx, Ny, xs, ys, results, domain };
}
