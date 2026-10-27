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

// Plotly's default surface rendering colors purely by the colorscale, with
// no simulated light. Turning "Light shading" on switches to a uniform
// "clay" color (SOLID_COLORSCALE) lit by a Phong-like light
// (ambient/diffuse/specular/roughness/fresnel) from a fixed position --
// deliberately dropping Viridis's own height-based color gradient, not
// just adding lighting on top of it, since that gradient already reads as
// a strong shape cue on its own and made the lighting's contribution hard
// to see.
//
// There's an inherent tension in picking one fixed (ambient, diffuse,
// specular, roughness) for every function this demo can show: a very tight
// roughness (~0.03) with low diffuse gives a crisp, clearly "shiny"
// highlight dot on a simple single-bump shape (Saddle) -- but on a more
// complex multi-featured shape (Cocentric's rings), that highlight cone can
// miss every facet entirely, and with diffuse this low there's no fallback
// shading at all, leaving the whole surface a single flat tone (confirmed:
// this exact failure mode is what an earlier version of this comment got
// wrong). The next attempt went too far the other way: ambient high enough
// (0.4) relative to diffuse's spread (0.25) that most of the surface sits
// in a narrow band near the ambient floor regardless of orientation --
// reads as "flat" -- while specular (0.4) on top of that narrow band spikes
// sharply bright by comparison, reading as a blown-out spot rather than a
// highlight blending into a gradient. Lowering ambient and raising diffuse
// widens the band every surface normal actually falls into (dark where it
// faces away from the light, bright where it faces toward it, visible
// gradient in between) so shading reads across the *whole* shape; lowering
// specular and broadening roughness keeps the highlight sitting on top of
// that gradient instead of overpowering it.
//
// That version still blew out broad *flat* regions solid white (the ring
// around a basis function's central bump, or the clamped flat skirt of
// SphericalDome): ablating each term from that exact combination showed
// specular (not ambient/diffuse, and not fresnel alone) was responsible --
// specular=0.3 in isolation only lit a thin ring near a slope transition,
// but specular+fresnel together blow out a wide area, because fresnel
// boosts apparent reflectivity at the grazing viewing angles that cover
// most of a flat plane's visible extent under this perspective camera.
// Lowering both (fresnel more, since it's what broadens specular's reach
// rather than its peak) fixed flat regions without losing the highlight on
// curved ones.
const SOLID_COLORSCALE = [[0, '#8fa8c8'], [1, '#8fa8c8']];
const LIGHTING = { ambient: 0.3, diffuse: 0.4, specular: 0.15, roughness: 0.2, fresnel: 0.05 };

// Even with the tight highlight above, a fully white scene background
// would still let a blown-out spot visually vanish into it (confirmed:
// swapping in a pink background revealed a "broken-looking" surface was
// actually fully intact, just camouflaged). A near-white but distinct
// background keeps any such spot visible as a highlight on the surface
// rather than a hole into the page. Applied unconditionally (not just
// when lit) rather than toggled: Plotly.react() was found to leave the
// WebGL canvas showing the previous bgcolor's pixels even after
// `scene.bgcolor` is explicitly changed back (confirmed via
// `_fullLayout.scene.bgcolor` reporting the new value correctly while the
// screenshot still showed the old one) -- toggling it is unreliable, but
// this color is subtle enough that having it on unconditionally makes no
// visible difference to the unlit Viridis view.
const SCENE_BGCOLOR = '#eef1f4';

// `lightposition` is in the same world units as the surface's own x/y/z
// data, not aspect-ratio-adjusted screen space -- a fixed position tuned
// for one function's domain looks wrong on another (this demo's domains
// range from unit squares up to Cocentric's ±10). Scaling the light's
// position to the plotted domain's own extent keeps it looking the same
// regardless of which function or resolution is showing. z is scaled by
// the same xyRange (not zRange, which is tiny for a near-flat surface) so
// the light sits well above the surface at a ~65° elevation -- high enough
// to read as an overhead key light instead of a grazing one (a z near 0,
// level with the surface's own base, is what made the light look too dark
// before: only the one edge facing the light caught any light at all).
function computeLightPosition(xRange, yRange) {
    const xyRange = Math.max(xRange, yRange) || 1;
    return { x: xyRange * 1.2, y: xyRange * 1.2, z: xyRange * 4 };
}

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
    // x and y must scale proportionally to their actual data ranges, not
    // both be forced to 1 -- on the hex lattice, the physical y-extent of
    // a sampled domain is generally sqrt(3) times its x-extent (or some
    // other ratio depending on the function's own domain), so hardcoding
    // x=y=1 silently squeezed the wider axis to match the narrower one,
    // distorting every plotted shape's true proportions (most visibly,
    // the hexagonal symmetry of the basis-function views). Scaling both by
    // the same larger-of-the-two divisor keeps whichever axis has the
    // bigger span at a ratio of 1 and shrinks the other by the true
    // physical ratio between them, so one x-unit and one y-unit always
    // render the same visual length.
    const x = xRange / xyRange;
    const y = yRange / xyRange;
    return { x, y, z };
}

// Serif italic font for math-notation-style titles (e.g. "<i>M<sub>h11</sub></i>").
// Plotly title text supports a small subset of HTML tags, including <i>/<sub>.
export const MATH_TITLE_FONT = { family: "Georgia, 'Times New Roman', serif" };

// zFlat: row-major flat array of shape (Ny,Nx) -- row = y-axis, col = x-axis,
// matching the layout produced by reconstruct.js. `supportPolygon`, if
// given, is a closed-loop list of [x,y] vertices (not repeating the first)
// outlining the basis function's support -- drawn as
// a 3D line resting on the z=0 plane, which coincides with the surface's own
// zero-contour at the support boundary. `titleFont`, if given, overrides the
// title's font (e.g. MATH_TITLE_FONT for the family-name panels).
// `lightShading`, if true, lights the surface instead of leaving it flat.
export function makeSurface(divId, xs, ys, zFlat, Nx, Ny, title, supportPolygon = null, titleFont = null, lightShading = false) {
    const z = [];
    for (let row = 0; row < Ny; row++) {
        z.push(Array.from(zFlat.slice(row * Nx, row * Nx + Nx)));
    }

    const xRange = Math.max(...xs) - Math.min(...xs) || 1;
    const yRange = Math.max(...ys) - Math.min(...ys) || 1;

    const data = [
        {
            type: 'surface',
            x: Array.from(xs),
            y: Array.from(ys),
            z,
            showscale: false,
            colorscale: lightShading ? SOLID_COLORSCALE : 'Viridis',
            ...(lightShading ? { lighting: LIGHTING, lightposition: computeLightPosition(xRange, yRange) } : {}),
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
        scene: {
            aspectmode: 'manual',
            aspectratio: computeAspectRatio(xs, ys, zFlat),
            bgcolor: SCENE_BGCOLOR,
            // Plotly's default dragmode, 'turntable', does not rotate the
            // camera at all under orthographic projection (confirmed on
            // the sibling demo-dynamic: even an 800px scripted drag
            // produced zero change in camera.eye). 'orbit' rotates
            // correctly under both projections.
            dragmode: 'orbit',
            // Perspective projection visibly skews a box this flat (z
            // aspect clamped to [0.3,0.6]) even when aspectratio/axis
            // ranges are correct, so orthographic is used normally (see
            // demo-dynamic/js/plot.js's identical setting). But Plotly's
            // `lighting` on a `surface` trace renders as a flat, uniform
            // color under orthographic projection -- no ambient/diffuse/
            // specular variation at all, confirmed by reproducing the
            // identical trace+lighting config with only the projection
            // type changed. Since there's no lit-and-undistorted option in
            // this Plotly version, lit mode switches to perspective (and
            // accepts the mild skew) rather than silently rendering no
            // lighting at all.
            camera: { projection: { type: lightShading ? 'perspective' : 'orthographic' } },
        },
    };

    Plotly.react(divId, data, layout, { responsive: true });
}
