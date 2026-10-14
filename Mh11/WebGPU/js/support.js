// Support-polygon outlines for each family's basis function, matching the
// "beige polygons" in Figure 1 of the paper (fig:dir.supp). Vertex lists are
// transcribed verbatim from each family's support_* array
// (Mh11.py/Mh20.py/Vhex.py); the coordinate transforms below are derived
// from Mh11.draw_bb()/Mh20.draw_bb()/Vhex.draw_bb() (which render these in
// "hex.G-native" units) composed with the relationship between
// hex.sample_function's checkerboard embedding and hex.G-native coordinates
// (physical_app = 2*scaling*physical_native). Verified numerically against
// the real GPU-evaluated basis surfaces: points just outside these polygons
// evaluate to exactly 0, points just inside are strictly positive.

import { SQRT3 } from './hex.js';

const SUPPORT_MH11 = [
    [3, 1], [3, 2], [2, 3], [1, 3], [-1, 2], [-2, 1],
    [-3, -1], [-3, -2], [-2, -3], [-1, -3], [1, -2], [2, -1],
];
const SUPPORT_MH20 = [
    [2, 0], [2, 2], [0, 2], [-2, 0], [-2, -2], [0, -2],
];
// Vhex.support_Vhex_dense (order-independent base hexagon; the actual
// polygon is `order * support_Vhex_dense`, per Vhex.draw_bb).
const SUPPORT_VHEX_DENSE = [
    [1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1],
];

// physical_native = hex.G @ (i,j); physical_app = 2*scaling*physical_native
// simplifies (with hex.G = [[0.5,0.5],[-sqrt3/2,sqrt3/2]]) to:
function boxSplineSupportVertex([i, j], scaling) {
    return [scaling * (i + j), scaling * SQRT3 * (j - i)];
}

// Vhex.draw_bb: GhexT = inv([[2,1],[-1,1]]) @ hex.G^T,
// polygon_native = (order*(i,j)) @ GhexT; composed with the 2*scaling
// factor above and simplified in closed form:
function vhexSupportVertex([i, j], order, scaling) {
    const x = scaling * order * j;
    const y = scaling * order * ((-2 * i) / SQRT3 + (j * SQRT3) / 3);
    return [x, y];
}

// Returns the closed-loop [x,y] vertex list (not repeating the first vertex)
// for the given family's support polygon at the given lattice `scaling`.
export function supportPolygon(familyKey, scaling) {
    switch (familyKey) {
        case 'Mh11':
            return SUPPORT_MH11.map((v) => boxSplineSupportVertex(v, scaling));
        case 'Mh20':
            return SUPPORT_MH20.map((v) => boxSplineSupportVertex(v, scaling));
        case 'Vhex4':
            return SUPPORT_VHEX_DENSE.map((v) => vhexSupportVertex(v, 4, scaling));
        default:
            throw new Error(`Unknown family: ${familyKey}`);
    }
}
