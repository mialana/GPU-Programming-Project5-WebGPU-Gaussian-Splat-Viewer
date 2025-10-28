@group(0) @binding(0)
var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(1)
var<storage, read> quads: array<Splat>;
@group(0) @binding(2)
var<uniform> camUnifs: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) center: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) conic: vec3<f32>,
    @location(3) opacity: f32,
}

// structs
struct Splat {
    position: u32,
    size: u32,
    color: array<u32, 2>,
    conic: array<u32, 2>,
}

// CPU ver of Struct
/*
interface CameraUniform {
    view_matrix: Mat4;
    view_inv_matrix: Mat4;
    proj_matrix: Mat4;
    proj_inv_matrix: Mat4;

    viewport: Vec2;
    focal: Vec2;
}
*/
struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
}

// static verts array for full-screen s_quadVerts
const s_quadVerts = array<vec2<f32>, 6>(vec2<f32>(-1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));

@vertex
fn vs_main(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat
    var out: VertexOutput;

    // retrieve sorted splat index
    let i = sortedIndices[inst_idx];
    let s = quads[i];

    // unpack
    out.center = unpack2x16float(s.position);
    let ndc_MaxSize = unpack2x16float(s.size);

    let conicLower = unpack2x16float(s.conic[0]);
    let conicUpper = unpack2x16float(s.conic[1]);
    out.conic = vec3<f32>(conicLower, conicUpper.x);
    out.opacity = conicUpper.y;

    let colorLower = unpack2x16float(s.color[0]);
    let colorUpper = unpack2x16float(s.color[1]);
    out.color = vec4<f32>(colorLower, colorUpper);

    // offset quad vertices
    let offset_Pos: vec2f = out.center + (s_quadVerts[vert_idx] * ndc_MaxSize);
    out.position = vec4<f32>(offset_Pos, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // map ndc_Pos to -1, 1
    var ndc_Pos: vec2f = (in.position.xy / camUnifs.viewport) * 2.0 - 1.0;
    // flip
    ndc_Pos.y = -ndc_Pos.y;

    var offset: vec2f = (ndc_Pos - in.center) * camUnifs.viewport * 0.5;

    // evaluate power
    // represents gaussian falloff
    let p: f32 = - 0.5 * (in.conic.x * (- offset.x) * (- offset.x) + in.conic.z * offset.y * offset.y) - in.conic.y * (- offset.x) * offset.y;

    if (p > 0.0) {
        return vec4<f32>(0.0); // discard if out of boundary
    }

    let a: f32 = min(0.99, in.opacity * exp(p)); // scale by opacity
    return in.color * a;
}
