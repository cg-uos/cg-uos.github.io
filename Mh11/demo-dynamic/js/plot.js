// Thin Plotly.js wrapper for a single combined plot: an optional
// reconstructed-spline surface and an optional coefficient-marker scatter,
// called once per animation frame.
// (Plotly is loaded globally via a CDN <script> tag in index.html.)

const Z_ASPECT_MIN = 0.3;
const Z_ASPECT_MAX = 0.6;

// "Light shading" switches the surface from Viridis's height-based colors
// to a uniform "clay" color lit by a Phong-like light.
const SOLID_COLORSCALE = [[0, '#8fa8c8'], [1, '#8fa8c8']];
// Low ambient/specular, wide diffuse/roughness: keeps shading visible
// across both simple and complex shapes without blowing out flat regions
// or highlights.
const LIGHTING = { ambient: 0.3, diffuse: 0.4, specular: 0.15, roughness: 0.2, fresnel: 0.05 };

// Near-white but distinct from pure white, so a lit highlight doesn't
// visually vanish into the background. Applied unconditionally rather
// than only when lit, since Plotly.react() can leave the canvas showing a
// stale bgcolor if toggled.
const SCENE_BGCOLOR = '#eef1f4';

// Scales the light position to the domain's own extent so it looks the
// same across resolutions, with z high enough (~65° elevation) to light
// the whole surface rather than just one grazing edge.
function computeLightPosition(xRange, yRange) {
    const xyRange = Math.max(xRange, yRange) || 1;
    return { x: xyRange * 1.2, y: xyRange * 1.2, z: xyRange * 4 };
}

function computeAspectRatio(xRange, yRange, zMin, zMax) {
    const zRange = zMax > zMin ? zMax - zMin : 1;
    const xyRange = Math.max(xRange, yRange) || 1;
    const z = Math.min(Z_ASPECT_MAX, Math.max(Z_ASPECT_MIN, zRange / xyRange));
    // Scales x and y by their true physical ratio (y is sqrt(3)x here)
    // rather than forcing both to 1, so the lattice's 60° angles don't
    // render skewed.
    const x = xRange / xyRange;
    const y = yRange / xyRange;
    return { x, y, z };
}

// xs, ys: Float32Array(N) axis coordinates of the eval grid.
// zFlat: row-major flat array of shape (N,N) (row=y, col=x).
// points: [{x,y,z}] current coefficient values at the lattice's visible points.
// edges: [{x1,y1,x2,y2}] the ground-plane hex-lattice connectivity (lattice.buildLattice's
// `edges`), drawn flat at z=0 when showLattice is on.
// lightShading: if true, lights the spline surface instead of leaving it flat.
// camera: the user's live camera (eye/up/center), if any -- see main.js.
export function renderPlot(divId, { xs, ys, zFlat, N, points, edges, showSpline, showCoefficients, showLattice, lightShading, camera, title }) {
    const data = [];
    const xRange = Math.max(...xs) - Math.min(...xs) || 1;
    const yRange = Math.max(...ys) - Math.min(...ys) || 1;

    let zMin = Infinity;
    let zMax = -Infinity;
    if (showSpline) {
        for (const v of zFlat) {
            if (!Number.isFinite(v)) continue;
            if (v < zMin) zMin = v;
            if (v > zMax) zMax = v;
        }
    }
    if (showCoefficients || showLattice) {
        // The stems and the ground lattice both touch z=0, so it has to be
        // in range too, even if every current coefficient happens to be
        // one-signed (or coefficients are hidden and only the lattice shows).
        zMin = Math.min(zMin, 0);
        zMax = Math.max(zMax, 0);
    }
    if (showCoefficients) {
        for (const p of points) {
            if (p.z < zMin) zMin = p.z;
            if (p.z > zMax) zMax = p.z;
        }
    }

    if (showLattice && edges && edges.length) {
        const latX = [], latY = [], latZ = [];
        for (const e of edges) {
            latX.push(e.x1, e.x2, null);
            latY.push(e.y1, e.y2, null);
            latZ.push(0, 0, null);
        }
        data.push({
            type: 'scatter3d',
            mode: 'lines',
            x: latX, y: latY, z: latZ,
            line: { color: 'rgba(100,110,130,0.65)', width: 2 },
            showlegend: false,
            hoverinfo: 'skip',
        });
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
            colorscale: lightShading ? SOLID_COLORSCALE : 'Viridis',
            opacity: showCoefficients ? 0.9 : 1,
            ...(lightShading ? { lighting: LIGHTING, lightposition: computeLightPosition(xRange, yRange) } : {}),
        });
    }

    if (showCoefficients) {
        // "Lollipop" stems (ground to each coefficient's signed height) as
        // one scatter3d line trace, plus a marker trace for the dots, both
        // on the same RdBu colorscale/cmin/cmax so a stem and its dot
        // always match. Size shrinks with point count so denser
        // resolutions don't turn into visual noise.
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

    const layout = {
        title: { text: title, font: { family: "Georgia, 'Times New Roman', serif" } },
        autosize: true,
        margin: { l: 0, r: 0, t: 30, b: 0 },
        scene: {
            aspectmode: 'manual',
            aspectratio: computeAspectRatio(xRange, yRange, zMin, zMax),
            bgcolor: SCENE_BGCOLOR,
            // Plotly's default dragmode, 'turntable', does not rotate the
            // camera at all under orthographic projection. 'orbit' rotates
            // correctly under both projections.
            dragmode: 'orbit',
            // Orthographic avoids perspective's skew on this flat, wide box,
            // but Plotly renders `lighting` as flat/uniform under
            // orthographic, so lit mode switches to perspective instead.
            // `camera` (main.js's captured `liveCamera`) is spread in first
            // so it survives the drag-suspension gap; `projection` is set
            // separately so it tracks `lightShading` on its own.
            camera: { ...camera, projection: { type: lightShading ? 'perspective' : 'orthographic' } },
        },
        // Keeps the camera stable across this loop's frequent Plotly.react
        // calls; without it, every frame would reset the view.
        uirevision: 'keep-camera',
    };

    // eslint-disable-next-line no-undef
    Plotly.react(divId, data, layout, { responsive: true });
}
