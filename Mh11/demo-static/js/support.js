// Support-polygon outlines for each family's basis function. 

import { SQRT3 } from './hex.js';

const SUPPORT_MH11 = [
    [3, 1], [3, 2], [2, 3], [1, 3], [-1, 2], [-2, 1],
    [-3, -1], [-3, -2], [-2, -3], [-1, -3], [1, -2], [2, -1],
];
const SUPPORT_MH20 = [
    [2, 0], [2, 2], [0, 2], [-2, 0], [-2, -2], [0, -2],
];
const SUPPORT_VHEX_DENSE = [
    [1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1],
];

// physical_native = Ghex @ (i,j); physical_app = 2*scaling*physical_native
// simplifies (with Ghex = [[0.5,0.5],[-sqrt3/2,sqrt3/2]]) to:
function boxSplineSupportVertex([i, j], scaling) {
    return [scaling * (i + j), scaling * SQRT3 * (j - i)];
}

// Vhex.draw_bb: GhexT = inv([[2,1],[-1,1]]) @ Ghex^T,
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
