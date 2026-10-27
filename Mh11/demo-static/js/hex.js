
export const SQRT3 = Math.sqrt(3);

// Inverse of the hex lattice generator Ghex. Matches the literal constant baked
// into every eval WGSL shader ("Ghex_inv") -- reused as-is rather than
// recomputed, to avoid float drift between the CPU-side sampling code and the
// GPU-side evaluation code.
export const Ghex_inv = [
    [1, 1],
    [-1 / SQRT3, 1 / SQRT3],
];

// shells[0..2]: only the first three "rings" of hex-lattice neighbors are
// needed here, since every FIR/IIR2 coefficient array used by this demo has
// at most 3 entries (h0/g0, h1/g1, h2/g2). 
export const shells = [
    [[0, 0]],
    [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]],
    [[2, 1], [1, 2], [-1, 1], [-2, -1], [-1, -2], [1, -1]],
];

// domain = [xmin, ymin, xmax, ymax]
export function findCoveringLattice(domain, scaling) {
    const [xmin, ymin, xmax, ymax] = domain;
    const i_min = Math.floor(xmin / (2 * scaling)) * 2;
    const j_min = Math.floor(ymin / (2 * scaling * SQRT3)) * 2;
    const i_max = Math.ceil(xmax / scaling);
    const j_max = Math.ceil(ymax / (scaling * SQRT3));
    return { i_min, j_min, i_max, j_max };
}

// Samples fn(x,y) on the hex lattice scaling*Ghex*Z^2 within "domain".
// Returns a flat row-major (i-major) Float32Array z of shape (nx,ny): the
// same layout the WGSL eval/prefilter shaders index via
// "coeff[y + x*size_coeff.y]" (x = i-index in [0,nx), y = j-index in [0,ny)).
export function sampleFunction(fn, domain, scaling) {
    let { i_min, j_min, i_max, j_max } = findCoveringLattice(domain, scaling);

    // Force an odd extent on both axes -- required by the checkerboard array
    // convention ("(i+j) even" = real lattice point).
    if ((i_max - i_min) % 2 === 0) i_max += 1;
    if ((j_max - j_min) % 2 === 0) j_max += 1;

    const nx = i_max - i_min + 1;
    const ny = j_max - j_min + 1;
    const z = new Float32Array(nx * ny);

    for (let a = 0; a < nx; a++) {
        const i = i_min + a;
        const x = i * scaling;
        for (let b = 0; b < ny; b++) {
            const j = j_min + b;
            const y = j * scaling * SQRT3;
            z[b + a * ny] = fn(x, y);
        }
    }

    return { i_min, j_min, i_max, j_max, nx, ny, z };
}
