// Builds the animated "coefficient" lattice: a dense checkerboard buffer
// (only (i+j) even entries are real lattice points; the odd ones are unused
// padding) in the same flat layout the FIR/eval WGSL shaders expect
// (`coeff[y + x*size_coeff.y]`, i.e. column-major over (i,j)).

import { SQRT3 } from './hex.js';

export const SCALING = 1.0;

// A visible "core" region of lattice points is what gets animated/plotted;
// a few extra rings of margin are carried in the GPU buffer around it
// (unseen) purely so the FIR prefilter's stencil (reach ~3 cells) composed
// with the eval shader's own stencil (reach ~3 cells, for Mh11) never has to
// read an edge-clamped value anywhere inside the visible core -- the
// "pad the domain, display less than you sample" trick.
export const MARGIN_CELLS = 6;

// `core` bounds the i-index (x-direction); the j-index (y-direction) is
// bounded separately by coreJ() below, so that the *physical* domain
// (x = i, y = j*sqrt3) comes out square rather than stretched 1:sqrt3 in y.
// The label's "i×j" dimensions reflect that asymmetry in index space even
// though the result looks square on screen.
export const RESOLUTIONS = [
    { label: 'Coarse (9×5)', core: 4 },
    { label: 'Medium (17×9)', core: 8 },
    { label: 'Fine (25×13)', core: 12 },
    { label: 'Very fine (37×21)', core: 18 },
];

// coreJ must have the *same parity* as core, not just be the nearest
// integer to core/sqrt3 -- see buildLattice's comment for why: this isn't
// a cosmetic rounding choice, a parity mismatch here makes the eval shader
// read only always-zero checkerboard cells, producing an identically-zero
// surface.
function coreJFor(core) {
    const target = core / SQRT3;
    let j = Math.round(target);
    if (((j - core) % 2 + 2) % 2 !== 0) {
        const lo = j - 1;
        const hi = j + 1;
        j = Math.abs(lo - target) <= Math.abs(hi - target) ? lo : hi;
    }
    return Math.max(2, j);
}

function isLatticePoint(i, j) {
    return (((i + j) % 2) + 2) % 2 === 0;
}

// The 6 nearest-neighbor offsets of a checkerboard lattice point under this
// module's physical mapping (x=i, y=j*SQRT3) are, by direct distance check,
// (±2,0) and (±1,±1) (matching signs) -- all six land at physical distance
// 2*SCALING, forming a regular hexagon around every point (angles 0°, 60°,
// 120°, 180°, 240°, 300°). Listing only half of them (one from each
// +/- opposite pair) generates every undirected edge exactly once, from
// whichever endpoint is the "offset" one; the other three ((-2,0),(-1,-1),
// (-1,1)) would just regenerate the same edges from the opposite end.
const HALF_NEIGHBOR_OFFSETS = [[2, 0], [1, 1], [1, -1]];

// Returns the buffer layout plus the flat list of *visible* (core-region)
// lattice points, each with its physical (x,y) and flat buffer offset, and
// the ground-plane hex-lattice edges connecting each point to its neighbors
// (for the optional "show ground lattice" overlay).
//
// The visible region is bounded by |i|<=core and |j|<=coreJ -- NOT a
// symmetric |i|,|j|<=core square -- because physical y=j*SCALING*SQRT3
// stretches the j-direction by sqrt(3) relative to x=i*SCALING. Using the
// same bound core for both would make the *physical* domain a 1:sqrt(3)
// rectangle even though it's a square in index space. Shrinking the
// j-bound to coreJ (coreJFor, same parity as core) compensates for that
// stretch so the physical domain comes out square instead (off by a few
// percent in practice, per the RESOLUTIONS presets' computed ratios --
// coreJFor's parity constraint takes priority over exact squareness).
export function buildLattice(core) {
    const coreJ = coreJFor(core);
    const halfI = core + MARGIN_CELLS;
    const halfJ = coreJ + MARGIN_CELLS;
    const i_min = -halfI;
    const j_min = -halfJ;
    const nx = 2 * halfI + 1; // odd, per the checkerboard convention
    const ny = 2 * halfJ + 1;

    const points = [];
    const edges = [];
    for (let i = -core; i <= core; i++) {
        for (let j = -coreJ; j <= coreJ; j++) {
            if (!isLatticePoint(i, j)) continue;
            const a = i - i_min;
            const b = j - j_min;
            points.push({
                x: i * SCALING,
                y: j * SCALING * SQRT3,
                offset: b + a * ny,
            });
            for (const [di, dj] of HALF_NEIGHBOR_OFFSETS) {
                const ni = i + di;
                const nj = j + dj;
                if (ni < -core || ni > core || nj < -coreJ || nj > coreJ) continue;
                edges.push({
                    x1: i * SCALING, y1: j * SCALING * SQRT3,
                    x2: ni * SCALING, y2: nj * SCALING * SQRT3,
                });
            }
        }
    }

    return { core, coreJ, halfI, halfJ, i_min, j_min, nx, ny, points, edges };
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
