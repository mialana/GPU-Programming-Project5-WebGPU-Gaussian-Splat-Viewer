const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>, // instance_count in DrawIndirect
    //data below is for info inside radix sort
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
}

struct SplatParams {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot: array<u32, 2>,
    scale: array<u32, 2>
}

struct Splat {
    // TODO: store information for 2D splat rendering
    // all params are packed
    position: u32,
    size: u32,
    color: array<u32, 2>,
    conic: array<u32, 2>
}

//TODO: bind your data here
@group(0) @binding(0)
var<storage, read> PC_sh_buffer: array<u32>;
@group(0) @binding(1)
var<storage, read> PC_gaussian_3d_buffer: array<Gaussian>;
@group(0) @binding(2)
var<storage, read_write> quadBuffer: array<Splat>;

@group(1) @binding(0)
var<uniform> camUnifs: CameraUniforms;
@group(1) @binding(1)
var<uniform> splatParams: SplatParams;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths: array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices: array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// reads the ith sh coef from the storage buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let half_idx = c_idx >> 1u; // index.e. c_idx / 2
    let odd = c_idx & 1u; // index.e. c_idx % 2

    let base_index = splat_idx * 24u + half_idx * 3u + odd;

    let color01 = unpack2x16float(PC_sh_buffer[base_index + 0u]); // lower 16 bits
    let color23 = unpack2x16float(PC_sh_buffer[base_index + 1u]); // upper 16 bits

    let even_vec = vec3f(color01.x, color01.y, color23.x);
    let odd_vec = vec3f(color01.y, color23.x, color23.y);

    // no divergence
    return mix(even_vec, odd_vec, f32(odd));
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return max(vec3<f32>(0.), result);
}
@compute @workgroup_size(workgroupSize, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    //TODO: set up pipeline as described in instruction

    // get length of runtime array with `arrayLength`
    if (gid.x >= arrayLength(&PC_gaussian_3d_buffer)) 
    { 
        return; 
    }

    // load gaussian
    let g: Gaussian = PC_gaussian_3d_buffer[gid.x];
    let posLower: vec2f = unpack2x16float(g.pos_opacity[0]);
    let posUpper: vec2f = unpack2x16float(g.pos_opacity[1]);
    let pos: vec3f = vec3<f32>(posLower.x, posLower.y, posUpper[0]);

    // view-space pos
    let vs_Pos: vec3f = (camUnifs.view * vec4<f32>(pos, 1.0)).xyz;
    var ndc_Pos: vec4f = camUnifs.proj * (camUnifs.view * vec4<f32>(pos, 1.0));
    ndc_Pos /= ndc_Pos.w; // normalize by w-component

    // simple view-frustum culling
    let paddedBbox = 1.2f;
    // z lower-bound is just 0
    if (ndc_Pos.x < -paddedBbox || ndc_Pos.y < -paddedBbox || ndc_Pos.z < 0.f 
    || ndc_Pos.x > paddedBbox || ndc_Pos.y > paddedBbox ||ndc_Pos.z > 1.f) 
    {
        return;
    }

    // unpack and apply rotation and scale
    let quatLower = unpack2x16float(g.rot[0]); // this is .w and .x components
    let quatUpper = unpack2x16float(g.rot[1]); // this is .y and .z components
    
    let r = quatLower.x; let x = quatLower.y; let y = quatUpper.x; let z = quatUpper.y;

    let R = mat3x3<f32>(
        1.0 - 2.0*(y*y + z*z),  2.0*(x*y - r*z),      2.0*(x*z + r*y),
        2.0*(x*y + r*z),        1.0 - 2.0*(x*x + z*z), 2.0*(y*z - r*x),
        2.0*(x*z - r*y),        2.0*(y*z + r*x),        1.0 - 2.0*(x*x + y*y)
    );

    let scaleLower = unpack2x16float(g.scale[0]); // scale stored in log space
    let scaleUpper  = unpack2x16float(g.scale[1]);
    let scale = exp(vec3<f32>(scaleLower.x, scaleLower.y, scaleUpper.x)); // bring back actual scale

    let gaussian_multiplier: f32 = splatParams.gaussian_scaling;
    let S = mat3x3<f32>(
        scale.x * gaussian_multiplier, 0.0, 0.0,
        0.0, scale.y * gaussian_multiplier, 0.0,
        0.0, 0.0, scale.z * gaussian_multiplier
    );

    // cov3D
    // remember rotation matrices are always orthonormal
    let byproduct = S * R;
    let cov3D = transpose(byproduct) * byproduct;

    // 2D cov3D
    let W = mat3x3<f32>(
        camUnifs.view[0].x, camUnifs.view[1].x, camUnifs.view[2].x,
        camUnifs.view[0].y, camUnifs.view[1].y, camUnifs.view[2].y,
        camUnifs.view[0].z, camUnifs.view[1].z, camUnifs.view[2].z
    );

    let J = mat3x3<f32>(
        camUnifs.focal.x / vs_Pos.z, 0.0, -(camUnifs.focal.x * vs_Pos.x) / (vs_Pos.z*vs_Pos.z),
        0.0, camUnifs.focal.y / vs_Pos.z, -(camUnifs.focal.y * vs_Pos.y) / (vs_Pos.z*vs_Pos.z),
        0.0, 0.0, 0.0
    );

    let T: mat3x3<f32> = W * J;

    let Vrk = mat3x3<f32>(
        cov3D[0][0], cov3D[0][1], cov3D[0][2],
        cov3D[0][1], cov3D[1][1], cov3D[1][2],
        cov3D[0][2], cov3D[1][2], cov3D[2][2]
    );

    var cov2D = transpose(T) * transpose(Vrk) * T;
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;

    // max radius
    let det = (cov2D[0][0]* cov2D[1][1] - cov2D[0][1]*cov2D[0][1]);
    if (det == 0.0) { return; }

    let mid = 0.5*(cov2D[0][0] + cov2D[1][1]);
    let λ1 = mid + sqrt(max(0.1, mid*mid - det));
    let λ2 = mid - sqrt(max(0.1, mid*mid - det));
    let radiusPixel = ceil(3.0 * sqrt(max(λ1, λ2)));
    let ndc_MaxSize = vec2(2.0 * radiusPixel) / camUnifs.viewport;

    // sort
    let k = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[k] = k;
    sort_depths[k] = bitcast<u32>(100.0 - vs_Pos.z);

    // position and max quad size
    quadBuffer[k].position = pack2x16float(ndc_Pos.xy); // re-pack
    quadBuffer[k].size = pack2x16float(ndc_MaxSize); // re-pack

    // color
    let viewDir = normalize(pos - camUnifs.view_inv[2].xyz);
    let color = computeColorFromSH(viewDir, gid.x, u32(splatParams.sh_deg));
    quadBuffer[k].color = array<u32,2>(
        pack2x16float(color.rg),
        pack2x16float(vec2(color.b, 1.0))
    ); // re-pack

    // conic
    let conic = vec3(cov2D[1][1] * (1.0/det), -cov2D[0][1] * (1.0/det), cov2D[0][0] * (1.0/det));
    let a = 1.0 / (1.0 + exp(-posUpper[1])); // take alpha from .w in packed pos
    quadBuffer[k].conic = array<u32,2>(
        pack2x16float(conic.xy),
        pack2x16float(vec2(conic.z, a))
    ); // re-pack

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    // TODO: avoid modulo?
    if (k % (workgroupSize * sortKeyPerThread) == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u); // indirect draw shenanigans
    }
}
