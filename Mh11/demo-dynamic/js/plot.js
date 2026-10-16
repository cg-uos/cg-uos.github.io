// Thin Plotly.js wrapper, adapted from ../../web/js/plot.js for a single
// combined plot: an optional reconstructed-spline surface and an optional
// coefficient-marker scatter, called once per animation frame.
// (Plotly is loaded globally via a CDN <script> tag in index.html.)

const Z_ASPECT_MIN = 0.3;
const Z_ASPECT_MAX = 0.6;

function computeAspectRatio(xRange, yRange, zMin, zMax) {
    const zRange = zMax > zMin ? zMax - zMin : 1;
    const xyRange = Math.max(xRange, yRange) || 1;
    const z = Math.min(Z_ASPECT_MAX, Math.max(Z_ASPECT_MIN, zRange / xyRange));
    return { x: 1, y: 1, z };
}

// xs, ys: Float32Array(N) axis coordinates of the eval grid.
// zFlat: row-major flat array of shape (N,N) (row=y, col=x).
// points: [{x,y,z}] current coefficient values at the lattice's visible points.
export function renderPlot(divId, { xs, ys, zFlat, N, points, showSpline, showCoefficients, title }) {
    const data = [];

    let zMin = Infinity;
    let zMax = -Infinity;
    if (showSpline) {
        for (const v of zFlat) {
            if (!Number.isFinite(v)) continue;
            if (v < zMin) zMin = v;
            if (v > zMax) zMax = v;
        }
    }
    if (showCoefficients) {
        // The stems always touch the ground (z=0), so it has to be in range
        // too, even if every current coefficient happens to be one-signed.
        zMin = Math.min(zMin, 0);
        zMax = Math.max(zMax, 0);
        for (const p of points) {
            if (p.z < zMin) zMin = p.z;
            if (p.z > zMax) zMax = p.z;
        }
    }

    if (showSpline) {
        const z = [];
        for (let row = 0; row < N; row++) {
            z.push(Array.from(zFlat.slice(row * N, row * N + N)));
        }
        data.push({
            type: 'surface',
            x: Array.from(xs),
            y: Array.from(ys),
            z,
            showscale: false,
            colorscale: 'Viridis',
            opacity: showCoefficients ? 0.9 : 1,
        });
    }

    if (showCoefficients) {
        // "Lollipop" stems from the ground (z=0) to each coefficient's
        // signed height, as one scatter3d trace holding every point's
        // segment back-to-back, separated by a `null` triple so Plotly
        // draws them as disconnected segments rather than one zig-zagging
        // polyline. Each segment is colored by its own height -- both
        // vertices share the same color value -- on the exact same
        // diverging colorscale (and explicit cmin/cmax, shared between the
        // two traces so a given height maps to the same color in both) as
        // the marker dots, so a bar and its dot always match.
        //
        // Marker/stem size shrinks with point count: at Coarse (~40 points)
        // they're sized to read individually; left at that fixed size at
        // Very fine (~700+ points) they'd overlap into a dense speckled
        // field that, however correct the underlying surface is, reads as
        // "noise" layered on top of it. REF_N anchors the scale so Coarse is
        // unchanged.
        const REF_N = 40;
        const sizeScale = Math.min(1, Math.sqrt(REF_N / Math.max(1, points.length)));
        const markerSize = Math.max(1.2, 3.5 * sizeScale);
        const stemWidth = Math.max(1.2, 5 * sizeScale);

        let cAbsMax = 0;
        for (const p of points) cAbsMax = Math.max(cAbsMax, Math.abs(p.z));
        if (cAbsMax === 0) cAbsMax = 1e-6;
        const cmin = -cAbsMax;
        const cmax = cAbsMax;

        const stemX = [], stemY = [], stemZ = [], stemC = [];
        for (const p of points) {
            stemX.push(p.x, p.x, null);
            stemY.push(p.y, p.y, null);
            stemZ.push(0, p.z, null);
            stemC.push(p.z, p.z, null);
        }
        data.push({
            type: 'scatter3d',
            mode: 'lines',
            x: stemX, y: stemY, z: stemZ,
            line: { color: stemC, colorscale: 'RdBu', cmin, cmax, width: stemWidth },
            showlegend: false,
            hoverinfo: 'skip',
        });
        data.push({
            type: 'scatter3d',
            mode: 'markers',
            x: points.map((p) => p.x),
            y: points.map((p) => p.y),
            z: points.map((p) => p.z),
            marker: { size: markerSize, color: points.map((p) => p.z), colorscale: 'RdBu', cmin, cmax, showscale: false },
            showlegend: false,
            hoverinfo: 'skip',
        });
    }

    const xRange = Math.max(...xs) - Math.min(...xs) || 1;
    const yRange = Math.max(...ys) - Math.min(...ys) || 1;

    const layout = {
        title: { text: title, font: { family: "Georgia, 'Times New Roman', serif" } },
        autosize: true,
        margin: { l: 0, r: 0, t: 30, b: 0 },
        scene: { aspectmode: 'manual', aspectratio: computeAspectRatio(xRange, yRange, zMin, zMax) },
        // Keeps the user's camera orbit/zoom stable across the frequent
        // Plotly.react calls an animation loop makes (as long as this value
        // itself doesn't change) -- without it, every frame would snap the
        // camera back to its default position.
        uirevision: 'keep-camera',
    };

    // eslint-disable-next-line no-undef
    Plotly.react(divId, data, layout, { responsive: true });
}
