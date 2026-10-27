function domain(xmin, ymin, xmax, ymax) {
    return { xmin, ymin, xmax, ymax };
}

// list of analytic test functions, plus "delta (basis)" -- not a function to
// reconstruct, but a way to see each family's basis shape itself
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
        // Only non-negative under the sqrt within radius sqrt(64/81)~0.889
        // of (0.5,0.5) -- comfortably outside this function's own displayed
        // domain (corner distance ~0.707), but reconstruct()'s margin
        // samples well beyond that (margin = scaling*9 or *4, pushing the
        // sampled corners out to distance ~2 from center), landing deep in
        // NaN territory and poisoning the FIR/eval stencils at every lattice
        // point within reach of the contaminated margin -- not just a few
        // edge pixels: at the default scaling this was ~98% of the
        // reconstructed grid. Clamping to 0 outside that radius flattens the
        // function to its rim height (z=-0.5) there instead, with no effect
        // on the function's own values inside the displayed domain.
        fn: (x, y) => Math.sqrt(Math.max(0, 64 - 81 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))) / 9 - 0.5,
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
    // Not a sampled analytic function: a Kronecker delta, which through the
    // same sample/evaluate pipeline reproduces each family's reconstruction
    // kernel (one unit coefficient at the lattice origin, every other
    // coefficient zero) -- the raw basis function when the user picks "None"
    // for the prefilter, or the prefilter's own impulse response for
    // FIR/IIR2. `fixedScaling` overrides the UI's scaling control -- chosen,
    // together with this domain, to frame every family's support without
    // clipping or looking tiny; it isn't meant to be a "sampling density"
    // for a basis shape, which has none. `noGroundTruth` tells reconstruct()
    // there's no meaningful meanAbsError for this entry (regardless of
    // prefilter choice, a delta has nothing sensible to compare against);
    // it also gates the support-polygon overlay, but only together with
    // prefilterScheme === 'none' -- FIR/IIR2 spread the delta into the
    // prefilter's own impulse response, which isn't compactly supported the
    // same way the raw basis function is, so the polygon wouldn't be exact.
    "delta (basis)": {
        fn: (x, y) => (x === 0 && y === 0 ? 1 : 0),
        domain: domain(-2, -2, 2, 2),
        fixedScaling: 0.2,
        noGroundTruth: true,
    },
};
