// Builds the animated "coefficient" lattice: a dense checkerboard buffer
// (only (i+j) even entries are real, per CLAUDE.md's convention -- see
// ../../python/CLAUDE.md) in the same flat layout the FIR/eval WGSL shaders
// expect (`coeff[y + x*size_coeff.y]`, i.e. column-major over (i,j)).

import { SQRT3 } from './hex.js';

export const SCALING = 1.0;

// A visible "core" region of lattice points is what gets animated/plotted;
// a few extra rings of margin are carried in the GPU buffer around it
// (unseen) purely so the FIR prefilter's stencil (reach ~3 cells) composed
// with the eval shader's own stencil (reach ~3 cells, for Mh11) never has to
// read an edge-clamped value anywhere inside the visible core -- the same
// "pad the domain, display less than you sample" trick
// ../../web/js/reconstruct.js uses (there via `margin = scaling*9`).
export const MARGIN_CELLS = 6;

export const RESOLUTIONS = [
    { label: 'Coarse (9×9)', core: 4 },
    { label: 'Medium (17×17)', core: 8 },
    { label: 'Fine (25×25)', core: 12 },
    { label: 'Very fine (37×37)', core: 18 },
];

function isLatticePoint(i, j) {
    return (((i + j) % 2) + 2) % 2 === 0;
}

// Returns the buffer layout plus the flat list of *visible* (core-region)
// lattice points, each with its physical (x,y) and flat buffer offset.
export function buildLattice(core) {
    const half = core + MARGIN_CELLS;
    const i_min = -half;
    const j_min = -half;
    const nx = 2 * half + 1; // odd, per the checkerboard convention
    const ny = 2 * half + 1;

    const points = [];
    for (let i = -core; i <= core; i++) {
        for (let j = -core; j <= core; j++) {
            if (!isLatticePoint(i, j)) continue;
            const a = i - i_min;
            const b = j - j_min;
            points.push({
                x: i * SCALING,
                y: j * SCALING * SQRT3,
                offset: b + a * ny,
            });
        }
    }

    return { core, half, i_min, j_min, nx, ny, points };
}

// One independent sinusoid per *buffer* cell (margin included, so the FIR
// stencil never sees a stale/zero value just outside the visible core
// either): value(t) = mid + amp*sin(speed*t + phase), with (min,max,speed,
// phase) all drawn independently at random, per the user's "randomly given
// minimum and maximum ... with different speeds" request.
export function randomizeParams(lattice) {
    const { nx, ny, i_min, j_min } = lattice;
    const n = nx * ny;
    const mid = new Float32Array(n);
    const amp = new Float32Array(n);
    const speed = new Float32Array(n);
    const phase = new Float32Array(n);

    for (let a = 0; a < nx; a++) {
        const i = i_min + a;
        for (let b = 0; b < ny; b++) {
            const j = j_min + b;
            if (!isLatticePoint(i, j)) continue;
            const off = b + a * ny;
            const lo = -0.6 + Math.random() * 1.2; // min in [-0.6, 0.6]
            const hi = lo + 0.3 + Math.random() * 1.2; // max = min + span in [0.3, 1.5]
            mid[off] = (lo + hi) / 2;
            amp[off] = (hi - lo) / 2;
            speed[off] = 0.3 + Math.random() * 1.7; // rad/s, different per point
            phase[off] = Math.random() * 2 * Math.PI;
        }
    }
    return { mid, amp, speed, phase };
}

// Evaluates every cell's sinusoid at time t (seconds) into `out` (reused
// across frames -- no per-frame allocation). Buffer cells outside the
// checkerboard (i+j odd) and outside the margin are left at amp=mid=0, so
// they read back as exactly 0, matching the "unused" convention.
export function evaluateFrame({ mid, amp, speed, phase }, t, out) {
    for (let k = 0; k < out.length; k++) {
        out[k] = mid[k] + amp[k] * Math.sin(speed[k] * t + phase[k]);
    }
    return out;
}
