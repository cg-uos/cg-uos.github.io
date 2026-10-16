// Trimmed down from ../../web/js/hex.js: this demo doesn't sample an
// analytic function onto the lattice (lattice.js builds the coefficient
// grid directly), so only the two pieces `shaders.js` and `lattice.js`
// actually need are kept.

export const SQRT3 = Math.sqrt(3);

// shells[0..2]: the first three "rings" of hex-lattice neighbors -- enough
// for any FIR prefilter here, since every filter's FIR table has at most 3
// entries (h0,h1,h2). Copied verbatim from hex.py's `shells` table.
export const shells = [
    [[0, 0]],
    [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]],
    [[2, 1], [1, 2], [-1, 1], [-2, -1], [-1, -2], [1, -1]],
];
