// Port of analytic_functions.py.
// (Full source: OneDrive-Personal/work/python/analytic_functions.py)

function domain(xmin, ymin, xmax, ymax) {
    return { xmin, ymin, xmax, ymax };
}

export const functions = {
    "Franke": {
        fn: (x, y) =>
            0.75 * Math.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4) +
            0.75 * Math.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10) +
            0.5 * Math.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4) -
            0.2 * Math.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2)),
        domain: domain(0, 0, 1, 1),
    },
    "Gaussian(iso)": {
        fn: (x, y) => Math.exp(-8 * (x * x + y * y)),
        domain: domain(-1, -1, 1, 1),
    },
    "Gaussian(aniso)": {
        fn: (x, y) => Math.exp(-(4 * x * x + 16 * y * y)),
        // Kept square (not matched to the x:y decay ratio) so the function's
        // anisotropy is still visible as an elongated bump rather than being
        // rescaled back into looking circular/isotropic. Sized to the slower
        // (x) decay direction: exp(-4x^2) is ~0.3% by |x|=1.2; [-3,3]^2 was
        // mostly flat padding on both axes.
        domain: domain(-1.2, -1.2, 1.2, 1.2),
    },
    "Tprod": {
        fn: (x, y) => Math.sin(2 * Math.PI * x) * Math.sin(2 * Math.PI * y),
        domain: domain(0, 0, 1, 1),
    },
    "Trad": {
        fn: (x, y) => Math.cos(Math.PI * (x * x + y * y)),
        domain: domain(-1, -1, 1, 1),
    },
    "Twave": {
        fn: (x, y) => Math.sin(2 * Math.PI * (2 * x + y)),
        domain: domain(-1, -1, 1, 1),
    },
    "Saddle": {
        fn: (x, y) => (1.25 + Math.cos(5.4 * y)) / (6 + 6 * (3 * x - 1) ** 2),
        domain: domain(0, 0, 1, 1),
    },
    "SphericalDome": {
        fn: (x, y) => Math.sqrt(64 - 81 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 9 - 0.5,
        domain: domain(0, 0, 1, 1),
    },
    "ExponentialCliff": {
        fn: (x, y) => Math.tanh(9 * (y - x) + 1) / 9,
        domain: domain(0, 0, 1, 1),
    },
    "Squared Distance": {
        fn: (x, y) => x * x + y * y,
        domain: domain(0, 0, 1, 1),
    },
    "Cocentric": {
        fn: (x, y) => Math.cos(Math.sqrt(x * x + y * y)),
        // cos(sqrt(r)) has period 2*pi in r (~6.28); [0,2]x[0,2] only covered
        // a corner of the first ring. Centered and widened so the corner
        // radius spans ~2 full periods and the concentric-ripple pattern the
        // function is named for is actually visible.
        domain: domain(-10, -10, 10, 10),
    },
};
