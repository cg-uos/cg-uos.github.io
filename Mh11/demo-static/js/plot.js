// Thin Plotly.js wrapper for the 2x2 surface-plot layout.
// (Plotly is loaded globally via a CDN <script> tag in index.html.)

// Plotly's 'cube' aspect mode forces the z-axis to the same pixel length as
// x/y regardless of the actual data range, which makes functions whose z
// range is small relative to their (x,y) domain (e.g. Cocentric's wide
// domain, or the compact basis functions) render as exaggerated spikes.
// Instead, size z relative to the true data ratio, clamped to a moderate
// band so nothing goes perfectly flat or perfectly spiky.
const Z_ASPECT_MIN = 0.3;
const Z_ASPECT_MAX = 0.6;

function computeAspectRatio(xs, ys, zFlat) {
    const xRange = Math.max(...xs) - Math.min(...xs) || 1;
    const yRange = Math.max(...ys) - Math.min(...ys) || 1;
    let zMin = Infinity;
    let zMax = -Infinity;
    for (const v of zFlat) {
        if (!Number.isFinite(v)) continue;
        if (v < zMin) zMin = v;
        if (v > zMax) zMax = v;
    }
    const zRange = zMax > zMin ? zMax - zMin : 1;
    const xyRange = Math.max(xRange, yRange);
    const z = Math.min(Z_ASPECT_MAX, Math.max(Z_ASPECT_MIN, zRange / xyRange));
    return { x: 1, y: 1, z };
}

// Serif italic font for math-notation-style titles (e.g. "<i>M<sub>h11</sub></i>").
// Plotly title text supports a small subset of HTML tags, including <i>/<sub>.
export const MATH_TITLE_FONT = { family: "Georgia, 'Times New Roman', serif" };

// zFlat: row-major flat array of shape (Ny,Nx) -- row = y-axis, col = x-axis,
// matching the layout produced by reconstruct.js. `supportPolygon`, if
// given, is a closed-loop list of [x,y] vertices (not repeating the first)
// outlining the basis function's support (Figure 1 of the paper) -- drawn as
// a 3D line resting on the z=0 plane, which coincides with the surface's own
// zero-contour at the support boundary. `titleFont`, if given, overrides the
// title's font (e.g. MATH_TITLE_FONT for the family-name panels).
export function makeSurface(divId, xs, ys, zFlat, Nx, Ny, title, supportPolygon = null, titleFont = null) {
    const z = [];
    for (let row = 0; row < Ny; row++) {
        z.push(Array.from(zFlat.slice(row * Nx, row * Nx + Nx)));
    }

    const data = [
        {
            type: 'surface',
            x: Array.from(xs),
            y: Array.from(ys),
            z,
            showscale: false,
            colorscale: 'Viridis',
        },
    ];

    if (supportPolygon && supportPolygon.length > 0) {
        const closed = [...supportPolygon, supportPolygon[0]];
        data.push({
            type: 'scatter3d',
            mode: 'lines',
            x: closed.map((p) => p[0]),
            y: closed.map((p) => p[1]),
            z: closed.map(() => 0),
            line: { color: '#c0392b', width: 5 },
            showlegend: false,
            hoverinfo: 'skip',
        });
    }

    const layout = {
        title: titleFont ? { text: title, font: titleFont } : { text: title },
        autosize: true,
        margin: { l: 0, r: 0, t: 30, b: 0 },
        scene: { aspectmode: 'manual', aspectratio: computeAspectRatio(xs, ys, zFlat) },
    };

    // eslint-disable-next-line no-undef
    Plotly.react(divId, data, layout, { responsive: true });
}
