// WGSL compute shader builders. These mirror the shader-string generation in
// hex.apply_FIR_checkerboard_WebGPU and each family's eval_*_WebGPU_bulk
// (OneDrive-Personal/work/python/{hex,Mh11,Mh20,Vhex}.py). The WGSL kernel
// bodies themselves are copied verbatim -- they are plain WGSL (not
// wgpu-py-specific), so the box-spline evaluation math (the paper's Fig. 3
// EvaluateSpline algorithm) does not need to be re-derived, only re-hosted.

import { shells } from './hex.js';

// Index-space transform used only to turn shells[] offsets into
// checkerboard-array neighbor offsets for the FIR shader. Distinct from
// hex.js's `Ghex_inv` (a different, geometric transform used by the eval
// shaders) -- kept separately named to avoid conflating the two.
const indexXfmG = [[1, 1], [-1, 1]];

function formatXfmCoeffHeader(Xfm) {
    const g = (r, c) => Xfm[r][c];
    return (
        `const Xfm_coeff = mat3x3f(${g(0, 0)}, ${g(1, 0)}, ${g(2, 0)},\n` +
        `                          ${g(0, 1)}, ${g(1, 1)}, ${g(2, 1)},\n` +
        `                          ${g(0, 2)}, ${g(1, 2)}, ${g(2, 2)});\n`
    );
}

// Port of hex.apply_FIR_checkerboard_WebGPU's shader_source generation.
export function buildFIRShader(FIRcoeffs, nx, ny) {
    let src = `const size_data: vec2i = vec2i(${nx},${ny});\n`;
    src += `
        @group(0) @binding(0)
        var<storage,read> data_in: array<f32>;

        @group(0) @binding(1)
        var<storage,read_write> data_out: array<f32>;

        fn get_datum(idx: vec2i) -> f32 {
            var i = min(max(vec2i(0), idx), size_data-1);
            return data_in[i.y + i.x*size_data.y];
        }

        @compute
        @workgroup_size(16,16,1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let idx: vec2i = vec2i(index.xy);
            var value: f32;
            if((idx.x + idx.y)%2 == 1)  {
                return;
            }
            `;
    src += "value = 0.0f;\n";
    for (let i = 0; i < FIRcoeffs.length; i++) {
        const coeff = FIRcoeffs[i];
        if (coeff === 0) continue;
        for (const [ox, oy] of shells[i]) {
            const px = indexXfmG[0][0] * ox + indexXfmG[0][1] * oy;
            const py = indexXfmG[1][0] * ox + indexXfmG[1][1] * oy;
            src += `            value += ${coeff}*get_datum(idx + vec2i(${px},${py}));\n`;
        }
    }
    src += `
            data_out[idx.y + idx.x * size_data.y] = value;
        }
        `;
    return src;
}

const TRANSFORM_TO_REF_6DIR = `
        struct TPreProcess {
            org: vec2f,
            p_ref: vec3f,
            Xfm_stencil: mat3x3f,
        };

        // Ghex_inv = [1, -1/sqrt(3)]
        //            [1,  1/sqrt(3)]
        const Ghex_inv = mat2x2f( 1, 1, -0.57735026919, 0.57735026919);

        fn transform_to_ref(p: vec3f) -> TPreProcess {

            var tbl_Xfm: array<mat3x3f,4> = array(
                mat3x3f( 1, 0, 0, 0, 1, 0, 0, 0, 1),
                mat3x3f( 0,-1, 0,-1, 0, 0, 1, 1, 1),
                mat3x3f( 0, 1, 0, 1, 0, 0, 0, 0, 1),
                mat3x3f(-1, 0, 0, 0,-1, 0, 1, 1, 1));

            var tbl_Xfm_sub: array<mat3x3f,3> = array(
                mat3x3f( 1, 0, 0, 0, 1, 0, 0, 0, 1),
                mat3x3f(-1, 0, 0, 1, 1, 0, 1, 0, 1),
                mat3x3f( 1, 1, 0, 0,-1, 0, 0, 0, 1));

            var tbl_Xfm_inv: array<mat3x3f,4> = array(
                mat3x3f( 1, 0, 0, 0, 1, 0, 0, 0, 1),
                mat3x3f( 0,-1, 0,-1, 0, 0, 1, 1, 1),
                mat3x3f( 0, 1, 0, 1, 0, 0, 0, 0, 1),
                mat3x3f(-1, 0, 0, 0,-1, 0, 1, 1, 1));

            var tbl_Xfm_inv_sub: array<mat3x3f,3> = array(
                mat3x3f( 1, 0, 0, 0, 1, 0, 0, 0, 1),
                mat3x3f(-1, 0, 0, 1, 1, 0, 1, 0, 1),
                mat3x3f( 1, 1, 0, 0,-1, 0, 0, 0, 1));

            var ret: TPreProcess;

            ret.org = floor(p.xy);
            var p_square = vec3f(p.xy - ret.org, 1.0);
            var type_tet = dot(vec2i(step(vec2f(0), mat3x2f(-1,1,1,1,0,-1) * p_square)), vec2i(2,1));

            var p_tet = tbl_Xfm[type_tet] * p_square;
            var type_tet_sub: i32 = 0;
            type_tet_sub += 2*i32(dot(vec3f(-1,2,0), p_tet) > 0);
            type_tet_sub += i32(type_tet_sub == 0)*i32(dot(vec3f(2,-1,-1), p_tet) > 0);

            ret.p_ref = tbl_Xfm_sub[type_tet_sub] * p_tet;
            ret.Xfm_stencil = tbl_Xfm_inv[type_tet] * tbl_Xfm_inv_sub[type_tet_sub];
            ret.org += ret.Xfm_stencil[2].xy;
            return ret;
        }
`;

// Port of Mh11.eval_Mh11_WebGPU_bulk's shader_source (Mh11.py:838-1007).
export function buildEvalShader_Mh11(nx, ny, Xfm) {
    let src = `const size_coeff: vec2i = vec2i(${nx}, ${ny});\n`;
    src += formatXfmCoeffHeader(Xfm);
    src += `
        @group(0) @binding(0)
        var<storage,read> coeff: array<f32>;

        @group(0) @binding(1)
        var<storage,read> pts: array<vec2f>;

        @group(0) @binding(2)
        var<storage,read_write> values: array<f32>;

        fn get_coeff(idx: vec2i) -> f32 {
            var x = min(max(0, idx.x), size_coeff.x-1);
            var y = min(max(0, idx.y), size_coeff.y-1);

            return coeff[y + x*size_coeff.y];
        }
`;
    src += TRANSFORM_TO_REF_6DIR;
    src += `
        fn eval_expr(c: array<f32,24>, u: vec3f) -> f32 {
            var u0_2 = u[0]*u[0];
            var u0_3 = u0_2*u[0];
            var u0_4 = u0_3*u[0];

            var u1_2 = u[1]*u[1];
            var u1_3 = u1_2*u[1];
            var u1_4 = u1_3*u[1];

            var u2_2 = u[2]*u[2];
            var u2_3 = u2_2*u[2];
            var u2_4 = u2_3*u[2];

            return (0
                + u0_4*(1632*c[0] + 752*(c[1] + c[2] + c[3] + c[4] + c[5] + c[6]) + 112*(c[7] + c[8] + c[9] + c[10] + c[11] + c[12] ) + 16*(c[13] + c[14] + c[15] + c[16] + c[17] + c[18] ))
                + u1_4*(1362*(c[0] + c[1]) + 928*(c[2] + c[6]) + 392*(c[3] + c[5] + c[7] + c[12]) + 222*(c[4] + c[13]) + 71*(c[8] + c[11] + c[14] + c[18] ) + 8*(c[9] + c[10] + c[19] + c[23] ) + (c[15] + c[17] + c[20] + c[22] ))
                + u2_4*(1280*(c[0] + c[1] + c[2]) + 574*(c[3] + c[6] + c[7]) + 210*(c[4] + c[5] + c[8] + c[12] + c[13] + c[14]) + 14*(c[9] + c[11] + c[15] + c[18] + c[19] + c[20]) + 2*(c[10] + c[21] + c[23])
                )
                + u0_3*(
                    u[1]*(  6528*c[0] + 3648*c[1] + 3328*(c[2] + c[6]) + 2688*(c[3] + c[5]) + 2368*c[4] + 640*(c[7] + c[12]) + 448*(c[8] + c[11]) + 256*(c[9] + c[10]) + 128*c[13] + 96*(c[14] + c[18]) + 32*(c[15] + c[17]))
                    +u[2]*( 6528*c[0] + 3648*(c[1] + c[2]) + 3008*(c[3] + c[6]) + 2368*(c[4] + c[5]) + 704*c[7] + 576*(c[12] + c[8]) + 320*(c[9] + c[11]) + 192*c[10] + 128*(c[13] + c[14]) + 64*(c[15] + c[18]))
                )
                + u1_3*(
                    u[0]*(  5952*c[0] + 4944*c[1] + 3712*(c[2] + c[6]) + 1920*(c[3] + c[5]) + 1264*c[4] + 1216*(c[7] + c[12]) + 512*c[13] + 352*(c[8] + c[11]) + 216*(c[14] + c[18]) + 64*(c[9] + c[10]) + 8*(c[15] + c[17]))
                    + u[2]*( 5448*(c[0] + c[1]) + 3344*c[6] + 4080*c[2] + 1760*(c[3] + c[7]) + 1376*(c[5] + c[12]) + 888*(c[4] + c[13]) + 376*(c[8] + c[14]) + 192*(c[11] + c[18]) + 40*(c[9] + c[19]) + 24*(c[10] + c[23]) + 8*(c[15] + c[20]))
                )

                + u2_3*(
                    u[0]*(  5760*c[0] + 4800*(c[1] + c[2]) + 2624*(c[3] + c[6]) + 1644*c[7] + 1216*(c[4] + c[5]) + 788*(c[8] + c[12]) + 512*(c[13] + c[14]) + 108*(c[9] + c[11]) + 64*(c[15] + c[18]) + 20*c[10] )
                    + u[1]*( 5280*(c[0] + c[1]) + 4800*c[2] + 2624*c[6] + 2132*(c[3] + c[7]) + 1004*(c[5] + c[12]) + 864*(c[4] + c[13]) + 652*(c[8] + c[14]) + 84*(c[11] + c[18]) + 52*(c[9] + c[19]) + 32*(c[15] + c[20]) + 12*(c[10] + c[23]))
                )
                + u0_2*(
                    u1_2*( 9504*c[0] + 6480*c[1] + 5376*(c[2] + c[6]) + 3456*(c[3] + c[5]) + 2640*c[4] + 1344*(c[7] + c[12]) + 624*(c[8] + c[11]) + 384*c[13] + 216*(c[14] + c[18]) + 192*(c[9] + c[10]) + 24*(c[15] + c[17]))
                    + u2_2*( 9408*c[0] + 6432*(c[1] + c[2]) + 4320*(c[3] + c[6]) + 2592*(c[4] + c[5]) + 1632*c[7] + 1056*(c[8] + c[12]) + 384*(c[13] + c[14]) + 288*(c[9] + c[11]) + 96*(c[10] + c[15] + c[18]))
                    + u[1]*u[2]*( 19008*c[0] + 12960*c[1] + 11808*c[2] + 9696*c[6] + 7776*c[3] + 6048*c[5] + 5280*c[4] + 2976*c[7] + 2400*c[12] + 1632*c[8] + 864*c[11] + 768*c[13] + 576*c[14] + 480*c[9] + 288*(c[10] + c[18]) + 96*c[15] )

                )
                + u1_2*(
                    u2_2*(8088*(c[0] + c[1]) + 6672*c[2] + 4464*c[6] + 2928*(c[3] + c[7]) + 1776*(c[5] + c[12]) + 1320*(c[4] + c[13]) + 744*(c[8] + c[14]) + 192*(c[11] + c[18]) + 72*(c[9] + c[19]) + 24*(c[10] + c[15] + c[20] + c[23]))
                    +u[0]*u[2]*(17856*c[0] + 14832*c[1] + 12240*c[2] + 10032*c[6] + 6480*c[3] + 5040*c[5] + 4080*c[7] + 3792*c[4] + 3216*c[12] + 1536*c[13] + 1392*c[8] + 864*c[14] + 720*c[11] + 432*c[18] + 240*c[9] + 144*c[10] + 48*c[15] )
                )
                + u2_2*u[0]*u[1]*(17664*c[0] + 14688*c[1] + 13344*c[2] + 8928*c[6] + 7200*c[3] + 4512*c[7] + 4320*c[5] + 3744*c[4] + 2784*c[12] + 1824*c[8] + 1536*c[13] + 1152*c[14] + 480*c[11] + 288*(c[9] + c[18]) + 96*c[10] + 96*c[15] )
            );
        }
        @compute
        @workgroup_size(256,1,1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {

            var stencil: array<vec3f,24> = array(
                vec3f( 0, 0, 0),

                vec3f( 1, 0, 0),
                vec3f( 1, 1, 0),
                vec3f( 0, 1, 0),
                vec3f(-1, 0, 0),
                vec3f(-1,-1, 0),
                vec3f( 0,-1, 0),

                vec3f( 2, 1, 0),
                vec3f( 1, 2, 0),
                vec3f(-1, 1, 0),
                vec3f(-2,-1, 0),
                vec3f(-1,-2, 0),
                vec3f( 1,-1, 0),

                vec3f( 2, 0, 0),
                vec3f( 2, 2, 0),
                vec3f( 0, 2, 0),
                vec3f(-2, 0, 0),
                vec3f(-2,-2, 0),
                vec3f( 0,-2, 0),

                vec3f( 3, 1, 0),
                vec3f( 3, 2, 0),
                vec3f( 2, 3, 0),
                vec3f( 1,-2, 0),
                vec3f( 2,-1, 0),
            );

            var p = vec3f(Ghex_inv * pts[index.x], 1);
            var ret = transform_to_ref(p);

            var u = mat3x3f(-2, 2, 0, 1,-4, 3, 1, 0, 0) * ret.p_ref;

            var c: array<f32,24>;
            for(var i = 0 ; i < 24 ; i++) {
                var shift = ret.Xfm_stencil * stencil[i];
                var idx = vec3i(Xfm_coeff * vec3f(ret.org + shift.xy, 1)).xy;
                c[i] = get_coeff(idx);
            }

            values[index.x] = eval_expr(c,u) * 0.00014467592;
        }
        `;
    return src;
}

// Port of Mh20.eval_Mh20_WebGPU_bulk's shader_source (Mh20.py:469-603).
export function buildEvalShader_Mh20(nx, ny, Xfm) {
    let src = `const size_coeff: vec2i = vec2i(${nx}, ${ny});\n`;
    src += formatXfmCoeffHeader(Xfm);
    src += `
        @group(0) @binding(0)
        var<storage,read> coeff: array<f32>;

        @group(0) @binding(1)
        var<storage,read> pts: array<vec2f>;

        @group(0) @binding(2)
        var<storage,read_write> values: array<f32>;

        fn get_coeff(idx: vec2i) -> f32 {
            var x = min(max(0, idx.x), size_coeff.x-1);
            var y = min(max(0, idx.y), size_coeff.y-1);

            return coeff[y + x*size_coeff.y];
        }

        struct TPreProcess {
            org: vec2f,
            p_ref: vec3f,
            Xfm_stencil: mat3x3f,
        };

        // Ghex_inv = [1, -1/sqrt(3)]
        //            [1,  1/sqrt(3)]
        const Ghex_inv = mat2x2f( 1, 1, -0.57735026919, 0.57735026919);

        fn transform_to_ref(p: vec3f) -> TPreProcess {
            var tbl_Xfm: array<mat3x3f,2> = array(
                mat3x3f( 1, 0, 0, 0, 1, 0, 0, 0, 1),
                mat3x3f( 0, 1, 0, 1, 0, 0, 0, 0, 1));
            var tbl_Xfm_inv: array<mat3x3f,2> = array(
                mat3x3f( 1, 0, 0, 0, 1, 0, 0, 0, 1),
                mat3x3f( 0, 1, 0, 1, 0, 0, 0, 0, 1));
            var ret: TPreProcess;

            ret.org = floor(p.xy);
            var p_square = vec3f(p.xy - ret.org, 1.0);
            var type_tet = i32(step(0.0, dot(vec3f(-1,1,0), p_square)));
            ret.p_ref = tbl_Xfm[type_tet] * p_square;
            ret.Xfm_stencil = tbl_Xfm_inv[type_tet];
            ret.org += ret.Xfm_stencil[2].xy;
            return ret;
        }

        fn eval_expr(c: array<f32,12>, u: vec3f) -> f32 {
            var u0_2 = u[0]*u[0];
            var u0_3 = u0_2*u[0];
            var u0_4 = u0_3*u[0];

            var u1_2 = u[1]*u[1];
            var u1_3 = u1_2*u[1];
            var u1_4 = u1_3*u[1];

            var u2_2 = u[2]*u[2];
            var u2_3 = u2_2*u[2];
            var u2_4 = u2_3*u[2];

            return 2*(
                  u0_4*(6*c[0] + (c[1] + c[2] + c[3] + c[4] + c[5] + c[6] ))
                + u1_4*(6*c[1] + (c[0] + c[2] + c[6] + c[7] + c[10] + c[11] ))
                + u2_4*(6*c[2] + (c[0] + c[1] + c[3] + c[7] + c[8] + c[9]))
                + 2*(
                      u0_3*(
                          u[1]*(12*c[0] + 4*c[1] + 3*(c[2] + c[6]) + (c[3] + c[5]))
                        + u[2]*(12*c[0] + 4*c[2] + 3*(c[1] + c[3]) + (c[4] + c[6]))
                    )
                    + u1_3*(
                          u[0]*(12*c[1] + 4*c[0] + 3*(c[2] + c[6]) + (c[7] + c[11]))
                        + u[2]*(12*c[1] + 4*c[2] + 3*(c[0] + c[7]) + (c[6] + c[10]))
                    )
                    + u2_3*(
                          u[0]*(12*c[2] + 4*c[0] + 3*(c[1] + c[3]) + (c[7] + c[8]))
                        + u[1]*(12*c[2] + 4*c[1] + 3*(c[0] + c[7]) + (c[3] + c[9]))
                        )
                )
                + 6*(
                      u0_2*(
                          u1_2*( 4*(c[0] + c[1]) + 2*(c[2] + c[6]))
                        + u2_2*( 4*(c[0] + c[2]) + 2*(c[1] + c[3]))
                        + u[1]*u[2]*( 10*c[0] + 6*(c[1] + c[2]) + (c[3] + c[6]))
                    )
                    + u1_2*(
                          u2_2*( 4*(c[1] + c[2]) + 2*(c[0] + c[7]))
                        + u[0]*u[2]*( 10*c[1] + 6*(c[0] + c[2]) + (c[6] + c[7]))
                        )
                    + u2_2*u[0]*u[1]*( 10*c[2] + 6*(c[0] + c[1]) + (c[3] + c[7]))
                )
            );

        }
        @compute
        @workgroup_size(256,1,1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {

            var stencil: array<vec3f,12> = array(
                vec3f( 0, 0, 0),
                vec3f( 1, 0, 0),
                vec3f( 1, 1, 0),
                vec3f( 0, 1, 0),
                vec3f(-1, 0, 0),
                vec3f(-1,-1, 0),
                vec3f( 0,-1, 0),
                vec3f( 2, 1, 0),
                vec3f( 1, 2, 0),
                vec3f( 2, 2, 0),
                vec3f( 2, 0, 0),
                vec3f( 1,-1, 0));

            var p = vec3f(Ghex_inv * pts[index.x], 1);
            var ret = transform_to_ref(p);

            var u = mat3x3f(-1, 1, 0, 0,-1, 1, 1, 0, 0) * ret.p_ref;

            var c: array<f32,12>;
            for(var i = 0 ; i < 12 ; i++) {
                var shift = ret.Xfm_stencil * stencil[i];
                var idx = vec3i(Xfm_coeff * vec3f(ret.org + shift.xy, 1)).xy;
                c[i] = get_coeff(idx);
            }

            values[index.x] = eval_expr(c,u) / 24.0;
        }
        `;
    return src;
}

// Port of Vhex.eval_Vhex_WebGPU_bulk's shader_source, order==4 branch
// (Vhex.py:1048-1247).
export function buildEvalShader_Vhex4(nx, ny, Xfm) {
    let src = `const size_coeff: vec2i = vec2i(${nx}, ${ny});\n`;
    src += formatXfmCoeffHeader(Xfm);
    src += `
        @group(0) @binding(0)
        var<storage,read> coeff: array<f32>;

        @group(0) @binding(1)
        var<storage,read> pts: array<vec2f>;

        @group(0) @binding(2)
        var<storage,read_write> values: array<f32>;

        fn get_coeff(idx: vec2i) -> f32 {
            var x = min(max(0, idx.x), size_coeff.x-1);
            var y = min(max(0, idx.y), size_coeff.y-1);

            return coeff[y + x*size_coeff.y];
        }
`;
    src += TRANSFORM_TO_REF_6DIR;
    src += `
        fn eval_expr(c: array<f32,16>, u: vec3f) -> f32 {
            var u0_2 = u[0]*u[0];
            var u0_3 = u0_2*u[0];
            var u0_4 = u0_3*u[0];
            var u0_5 = u0_4*u[0];
            var u0_6 = u0_5*u[0];

            var u1_2 = u[1]*u[1];
            var u1_3 = u1_2*u[1];
            var u1_4 = u1_3*u[1];
            var u1_5 = u1_4*u[1];
            var u1_6 = u1_5*u[1];

            var u2_2 = u[2]*u[2];
            var u2_3 = u2_2*u[2];
            var u2_4 = u2_3*u[2];
            var u2_5 = u2_4*u[2];
            var u2_6 = u2_5*u[2];

            return (
                u0_6*( 8880*c[0] + 1740*(c[1] + c[2] + c[3] + c[4] + c[5] + c[6]) + 20*( c[7] + c[8] + c[9] + c[10] + c[11] + c[12]))
                + u1_6*( 5280*(c[0] + c[1] + c[2]) + 960*(c[3] + c[4] + c[7]) + 120*(c[12] + c[13] + c[15] + c[5] + c[6] + c[8]))
                + u2_6*( 5280*(c[0] + c[2] + c[3]) + 960*(c[1] + c[6] + c[8]) + 120*(c[14] + c[15] + c[4] + c[5] + c[7] + c[9]))
                + u0_5*(
                    u[1]*( 53280*c[0] + 13500*(c[1] + c[2]) + 10440*(c[3] + c[4]) + 7380*(c[5] + c[6]) + 240*c[7] + 180*(c[8] + c[12]) + 60*(c[9] + c[11]))
                    +u[2]*( 53280*c[0] + 13500*(c[2] + c[3]) + 10440*(c[1] + c[6]) + 7380*(c[4] + c[5]) + 240*c[8] + 180*(c[7] + c[9]) + 60*(c[10] + c[12]))
                )
                + u1_5*(
                    u[0]*( 37248*c[0] + 28896*(c[1] + c[2]) + 6936*(c[3] + c[4]) + 3408*c[7] + 1200*(c[5] + c[6]) + 624*(c[8] + c[12]) + 336*(c[13] + c[15]))
                    + u[2]*( 34464*(c[0] + c[2]) + 26112*c[1] + 8112*c[3] + 4584*(c[4] + c[7]) + 1104*(c[6] + c[8]) + 816*(c[5] + c[15]) + 240*(c[12] + c[13]))
                )
                + u2_5*(
                    u[0]*( 37248*c[0] + 28896*(c[2] + c[3]) + 6936*(c[1] + c[6]) + 3408*c[8] + 1200*(c[4] + c[5]) + 624*(c[7] + c[9]) + 336*(c[14] + c[15]))
                    +u[1]*( 34464*(c[0] + c[2]) + 26112*c[3] + 8112*c[1] + 4584*(c[6] + c[8]) + 1104*(c[4] + c[7]) + 816*(c[5] + c[15]) + 240*(c[9] + c[14]))
                )
                + u0_4*(
                    u1_2*( 128400*c[0] + 42900*(c[1] + c[2]) + 25080*(c[3] + c[4]) + 12300*(c[5] + c[6]) + 1200*c[7] + 660*(c[8] + c[12]) + 60*(c[9] + c[11]))
                    + u2_2*( 128400*c[0] + 42900*(c[2] + c[3]) + 25080*(c[1] + c[6]) + 12300*(c[4] + c[5]) + 1200*c[8] + 660*(c[7] + c[9]) + 60*(c[10] + c[12]))
                    + u[1]*u[2]*( 261600*c[0] + 86820*c[2] + 66480*(c[1] + c[3]) + 35880*(c[4] + c[6]) + 25620*c[5] + 1800*(c[7] + c[8]) + 420*(c[9] + c[12]))
                )
                + u1_4*(
                    u0_2*( 106920*c[0] + 63180*(c[1] + c[2]) + 20340*(c[3] + c[4]) + 4920*(c[5] + c[6]) + 4680*c[7] + 1260*(c[8] + c[12]) + 300*(c[13] + c[15]))
                    +u2_2*(91020*(c[0] + c[2]) + 51240*c[1] + 28200*c[3] + 8580*(c[4] + c[7]) + 4140*(c[6] + c[8]) + 2220*(c[5] + c[15]) + 120*(c[12] + c[13]))
                    +u[0]*u[2]*( 200040*c[0] + 154320*c[2] + 116520*c[1] + 48480*c[3] + 26880*c[4] + 13200*c[7] + 9120*c[6] + 6720*c[5] + 4680*c[8] + 1800*c[15] + 960*c[12] + 480*c[13] )
                )
                + u2_4*(
                    u0_2*( 106920*c[0] + 63180*(c[2] + c[3]) + 20340*(c[1] + c[6]) + 4920*(c[4] + c[5]) + 4680*c[8] + 1260*(c[7] + c[9]) + 300*(c[14] + c[15]))
                    +u1_2*( 91020*(c[0] + c[2]) + 51240*c[3] + 28200*c[1] + 8580*(c[6] + c[8]) + 4140*(c[4] + c[7]) + 2220*(c[5] + c[15]) + 120*(c[9] + c[14]))
                    +u[0]*u[1]*( 200040*c[0] + 154320*c[2] + 116520*c[3] + 48480*c[1] + 26880*c[6] + 13200*c[8] + 9120*c[4] + 6720*c[5] + 4680*c[7] + 1800*c[15] + 960*c[9] + 480*c[14] )
                )
                + u0_3*(
                    u1_3*( 158880*c[0] + 70740*(c[1] + c[2]) + 30720*(c[3] + c[4]) + 10540*(c[5] + c[6]) + 3200*c[7] + 1260*(c[12] + c[8]) + 80*(c[13] + c[15]) + 20*(c[9] + c[11]))
                    +u2_3*( 158880*c[0] + 70740*(c[2] + c[3]) + 30720*(c[1] + c[6]) + 10540*(c[4] + c[5]) + 3200*c[8] + 1260*(c[7] + c[9]) + 80*(c[14] + c[15]) + 20*(c[10] + c[12]))
                    +u1_2*u[2]*( 495120*c[0] + 218100*c[2] + 165720*c[1] + 125040*c[3] + 67440*c[4] + 46920*c[6] + 33900*c[5] + 7200*c[7] + 5280*c[8] + 1140*c[12] + 300*c[9] + 240*c[15] )
                    +u2_2*u[1]*( 495120*c[0] + 218100*c[2] + 165720*c[3] + 125040*c[1] + 67440*c[6] + 46920*c[4] + 33900*c[5] + 7200*c[8] + 5280*c[7] + 1140*c[9] + 300*c[12] + 240*c[15] )
                )
                + u1_3*(
                    u2_3*( 123780*(c[0] + c[2]) + 51360*(c[1] + c[3]) + 8080*(c[4] + c[6] + c[7] + c[8]) + 3060*(c[5] + c[15]) + 20*(c[9] + c[12] + c[13] + c[14]))
                    + u0_2*u[2]*( 452160*c[0] + 264780*c[2] + 200160*c[1] + 112320*c[3] + 61200*c[4] + 29640*c[6] + 21660*c[5] + 14160*c[7] + 7320*c[8] + 1500*c[12] + 1200*c[15] + 240*c[13] + 60*c[9] )
                    + u2_2*u[0]*( 415680*c[0] + 319740*c[2] + 179520*c[1] + 133440*c[3] + 39120*c[4] + 27120*c[6] + 19440*c[7] + 14340*c[5] + 13680*c[8] + 3720*c[15] + 420*c[12] + 120*c[13] + 60*c[9])
                )
                + u2_3*(
                    u0_2*u[1]*( 452160*c[0] + 264780*c[2] + 200160*c[3] + 112320*c[1] + 61200*c[6] + 29640*c[4] + 21660*c[5] + 14160*c[8] + 7320*c[7] + 1500*c[9] + 1200*c[15] + 240*c[14] + 60*c[12] )
                    + u1_2*u[0]*( 415680*c[0] + 319740*c[2] + 179520*c[3] + 133440*c[1] + 39120*c[6] + 27120*c[4] + 19440*c[8] + 14340*c[5] + 13680*c[7] + 3720*c[15] + 420*c[9] + 120*c[14] + 60*c[12] )
                )
                + u0_2*u1_2*u2_2*( 691560*c[0] + 403740*c[2] + 228240*(c[1] + c[3]) + 65160*(c[4] + c[6]) + 33660*c[5] + 15480*(c[7] + c[8]) + 1800*c[15] + 540*(c[9] + c[12]))
            );


        }
        @compute
        @workgroup_size(256,1,1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {

            var stencil: array<vec3f,16> = array(
                vec3f( 0, 0, 0),
                vec3f( 0,-1, 0),
                vec3f( 1, 0, 0),
                vec3f( 1, 1, 0),
                vec3f(-1,-1, 0),
                vec3f(-1, 0, 0),
                vec3f( 0, 1, 0),
                vec3f( 1,-1, 0),
                vec3f( 2, 1, 0),
                vec3f( 1, 2, 0),
                vec3f(-1, 1, 0),
                vec3f(-2,-1, 0),
                vec3f(-1,-2, 0),
                vec3f( 0,-2, 0),
                vec3f( 2, 2, 0),
                vec3f( 2, 0, 0),
           );

            var p = vec3f(Ghex_inv * pts[index.x], 1);
            var ret = transform_to_ref(p);

            var u = mat3x3f(-2, 1, 1, 1,-2, 1, 1, 0, 0) * ret.p_ref;

            var c: array<f32,16>;
            for(var i = 0 ; i < 16 ; i++) {
                var shift = ret.Xfm_stencil * stencil[i];
                var idx = vec3i(Xfm_coeff * vec3f(ret.org + shift.xy, 1)).xy;
                c[i] = get_coeff(idx);
            }

            values[index.x] = eval_expr(c,u) / 19440.0;
        }
        `;
    return src;
}
