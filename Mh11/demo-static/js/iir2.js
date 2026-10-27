// The factored all-pole IIR2 prefilter, applied as 
// three separable mirror-boundary recursive passes
// (columns, then both diagonal directions) over the checkerboard-packed
// lattice array. 

function applyIirMirrorSequence(values, g0, g1, g2) {
    const n = values.length;
    const coeff = Float32Array.from(values);
    if (n < 2) return coeff;

    if (n > 2) {
        coeff[0] = (coeff[0] - g1 * coeff[1] - g2 * coeff[2]) / g0; // mirrored
    } else {
        coeff[0] = (coeff[0] - g1 * coeff[1] - g2 * coeff[0]) / g0;
    }

    coeff[1] = (coeff[1] - g1 * coeff[0] - g2 * coeff[1]) / g0; // mirrored

    for (let idx = 2; idx < n; idx++) {
        coeff[idx] = (coeff[idx] - g1 * coeff[idx - 1] - g2 * coeff[idx - 2]) / g0;
    }

    if (n > 2) {
        coeff[n - 1] = (coeff[n - 1] - g1 * coeff[n - 2] - g2 * coeff[n - 3]) / g0;
    } else {
        coeff[n - 1] = (coeff[n - 1] - g1 * coeff[n - 2] - g2 * coeff[n - 1]) / g0;
    }

    coeff[n - 2] = (coeff[n - 2] - g1 * coeff[n - 1] - g2 * coeff[n - 2]) / g0;

    for (let idx = n - 3; idx >= 0; idx--) {
        coeff[idx] = (coeff[idx] - g1 * coeff[idx + 1] - g2 * coeff[idx + 2]) / g0;
    }

    return coeff;
}

function applyIirMirrorToCoords(data, cols, coords, g0, g1, g2) {
    if (coords.length < 2) return;
    const line = new Float32Array(coords.length);
    for (let k = 0; k < coords.length; k++) {
        const [r, c] = coords[k];
        line[k] = data[r * cols + c];
    }
    const filtered = applyIirMirrorSequence(line, g0, g1, g2);
    for (let k = 0; k < coords.length; k++) {
        const [r, c] = coords[k];
        data[r * cols + c] = filtered[k];
    }
}

function* iterDiagDownRightStarts(rows, cols) {
    for (let c = 0; c < cols; c += 2) yield [0, c];
    for (let r = 1; r < rows; r++) {
        if (r % 2 === 0) yield [r, 0];
    }
}

function* iterDiagDownLeftStarts(rows, cols) {
    const rightCol = cols - 1;
    for (let c = 0; c < cols; c += 2) yield [0, c];
    for (let r = 1; r < rows; r++) {
        if ((r + rightCol) % 2 === 0) yield [r, rightCol];
    }
}

// data: flat row-major Float32Array of shape (rows,cols) (rows=nx, cols=ny
// in this app's lattice convention). g = [g0,g1,g2]. Returns a new Float32Array.
export function applyIirCheckerboard(data, rows, cols, g) {
    const out = Float32Array.from(data);
    const [g0, g1, g2] = g;

    for (let c = 0; c < cols; c++) {
        const coords = [];
        for (let r = c % 2; r < rows; r += 2) coords.push([r, c]);
        applyIirMirrorToCoords(out, cols, coords, g0, g1, g2);
    }

    for (const [startR, startC] of iterDiagDownRightStarts(rows, cols)) {
        const coords = [];
        let r = startR, c = startC;
        while (r < rows && c < cols) { coords.push([r, c]); r++; c++; }
        applyIirMirrorToCoords(out, cols, coords, g0, g1, g2);
    }

    for (const [startR, startC] of iterDiagDownLeftStarts(rows, cols)) {
        const coords = [];
        let r = startR, c = startC;
        while (r < rows && c >= 0) { coords.push([r, c]); r++; c--; }
        applyIirMirrorToCoords(out, cols, coords, g0, g1, g2);
    }

    return out;
}
