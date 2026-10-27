// Orchestrates the sample -> prefilter -> evaluate pipeline for all three
// spline families.

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

// Xfm_coeff: its 2x2 block equals
// Ghex in index space; row-major nested-array layout, Xfm[row][col].
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
// is (i_min,j_min).
// xy_recon = xy/(2*scaling) + [-i_min/2, -j_min*sqrt(3)/2].
// Matches np.meshgrid(linspace_x, linspace_y) with default 'xy' indexing: 
// x varies fastest (inner loop), y is the outer (row) axis.
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

// options: { funcName, scaling, prefilterScheme: 'FIR'|'IIR2'|'none', Nx, Ny }
// `scaling` is the UI's requested sampling density, but a functions[] entry
// can override it with `fixedScaling` (e.g. the "delta (basis)" entry) when
// the named thing isn't really a "function sampled at a density" (a basis
// shape has no sampling density of its own). `noGroundTruth` means there's
// no meaningful meanAbsError for this entry regardless of prefilter choice
// (a delta has nothing sensible to compare a reconstruction against); the
// support-polygon overlay additionally requires prefilterScheme === 'none',
// since FIR/IIR2 spread the delta into the prefilter's own impulse
// response, which isn't compactly supported the same way the raw basis
// function is -- the polygon wouldn't be exact for those.
export async function reconstruct(device, options) {
    const { funcName, scaling: uiScaling, prefilterScheme, Nx = 100, Ny = 100 } = options;
    const { fn, domain, fixedScaling, noGroundTruth } = functions[funcName];
    const scaling = fixedScaling ?? uiScaling;
    const showSupportPolygon = noGroundTruth && prefilterScheme === 'none';

    // margin = scaling*9 (demo_reconstruct_analytic_function.py:14,69) covers
    // the FIR/IIR2 prefilter's own stencil reach (~3 cells) composed with
    // the eval shader's (~3 cells); with no prefilter running, only the eval
    // shader's own reach needs covering, so a smaller margin suffices.
    const margin = scaling * (prefilterScheme === 'none' ? 4 : 9);
    const sampleDomain = [
        domain.xmin - margin,
        domain.ymin - margin,
        domain.xmax + margin,
        domain.ymax + margin,
    ];

    const { i_min, j_min, nx, ny, z } = sampleFunction(fn, sampleDomain, scaling);
    const { xs, ys, ptsRecon, numPts } = buildEvalGrid(domain, scaling, i_min, j_min, Nx, Ny);

    let zTrue = null;
    if (!noGroundTruth) {
        zTrue = new Float32Array(numPts);
        let idx = 0;
        for (let row = 0; row < Ny; row++) {
            for (let col = 0; col < Nx; col++) {
                zTrue[idx] = fn(xs[col], ys[row]);
                idx++;
            }
        }
    }

    const results = {};
    for (const family of FAMILIES) {
        let coeff;
        if (prefilterScheme === 'none') {
            coeff = z;
        } else {
            const pf = prefilters[family.key][prefilterScheme];
            coeff =
                prefilterScheme === 'FIR'
                    ? await runFIRPrefilterGPU(device, pf, z, nx, ny)
                    : applyIirCheckerboard(z, nx, ny, pf);
        }

        const zRecon = await runEvalGPU(
            device,
            family.buildEvalShader,
            coeff,
            nx,
            ny,
            ptsRecon,
            numPts
        );

        if (showSupportPolygon) {
            results[family.key] = { zRecon, label: family.label, supportPolygon: supportPolygon(family.key, scaling) };
        } else if (noGroundTruth) {
            results[family.key] = { zRecon, label: family.label };
        } else {
            let errSum = 0;
            for (let k = 0; k < numPts; k++) errSum += Math.abs(zTrue[k] - zRecon[k]);
            results[family.key] = { zRecon, meanAbsError: errSum / numPts, label: family.label };
        }
    }

    return { Nx, Ny, xs, ys, zTrue, results, domain };
}
