// Hex-lattice constants used by `shaders.js` and `lattice.js`. This demo
// builds its coefficient grid directly (see lattice.js) rather than
// sampling an analytic function onto the lattice, so that's all that's
// needed here.

export const SQRT3 = Math.sqrt(3);

// shells[0..2]: the first three "rings" of hex-lattice neighbors -- enough
// for any FIR prefilter here, since every filter's FIR table has at most 3
// entries (h0,h1,h2).
export const shells = [
    [[0, 0]],
    [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]],
    [[2, 1], [1, 2], [-1, 1], [-2, -1], [-1, -2], [1, -1]],
];
