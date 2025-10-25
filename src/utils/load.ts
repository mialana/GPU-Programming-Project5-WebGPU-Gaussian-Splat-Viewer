import { Float16Array } from "@petamoriken/float16";
import { log } from "./simple-console";
import {
    decodeHeader,
    readRawVertex,
    readRawCompressedVertex,
    readChunks,
    nShCoeffs,
} from "./plyreader";

const c_size_float = 2; // byte size of f16

const c_size_3d_gaussian =
    3 * c_size_float + // x y z (position)
    c_size_float + // opacity
    4 * c_size_float + // rotation
    4 * c_size_float; //scale
export type PointCloud = Awaited<ReturnType<typeof load>>;

export async function load(file: File, device: GPUDevice) {
    const isCompressed: boolean = file.name.endsWith(".compressed.ply");
    const arrayBuffer = await file.arrayBuffer();

    const [vertexCount, propertyTypes, vertexData] = decodeHeader(
        arrayBuffer as ArrayBuffer,
        isCompressed,
    );

    let chunks = [];
    if (isCompressed) {
        // Re-decode only the header text portion for parsing
        const headerDecoder = new TextDecoder();
        const headerText = headerDecoder.decode(
            (arrayBuffer as ArrayBuffer).slice(0, 2000),
        );
        chunks = readChunks(arrayBuffer as ArrayBuffer, headerText);
    }

    let sh_deg = 0;
    let num_coefs = 0;
    const max_num_coefs = 16;
    let shFeatureOrder: string[] = [];
    const c_size_sh_coef = 3 * max_num_coefs * c_size_float;

    if (!isCompressed) {
        // figure out the SH degree from the number of coefficients
        let nRestCoeffs = 0;
        for (const propertyName in propertyTypes) {
            if (propertyName.startsWith("f_rest_")) {
                nRestCoeffs += 1;
            }
        }
        const nCoeffsPerColor = nRestCoeffs / 3;
        sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
        num_coefs = nShCoeffs(sh_deg);

        // build SH feature order
        for (let rgb = 0; rgb < 3; ++rgb) {
            shFeatureOrder.push(`f_dc_${rgb}`);
        }
        for (let i = 0; i < nCoeffsPerColor; ++i) {
            for (let rgb = 0; rgb < 3; ++rgb) {
                shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
            }
        }
    }

    const num_points = vertexCount;

    log(`num points: ${num_points}`);
    log(`processing loaded attributes...`);

    const gaussianBufferSize = num_points * c_size_3d_gaussian;
    const shBufferSize = num_points * c_size_sh_coef;

    // check individual + combined limits
    const limit = device.limits.maxBufferSize;

    if (gaussianBufferSize > limit || shBufferSize > limit) {
        throw new RangeError(
            `PLY exceeds GPU Max Buffer Size limit: \n` +
                `gaussian = ${(gaussianBufferSize / 1e6).toFixed(2)}MB, ` +
                `sh = ${(shBufferSize / 1e6).toFixed(2)}MB, allowedLimit = ${(limit / 1e6).toFixed(2)}MB\n` +
                `choose another file.`,
        );
    }

    // xyz (position), opacity, cov (from rot and scale)
    const gaussian_3d_buffer = device.createBuffer({
        label: "ply input 3d gaussians data buffer",
        size: gaussianBufferSize, // buffer size multiple of 4?
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

    // Spherical harmonic function coeffs
    const sh_buffer = device.createBuffer({
        label: "ply input 3d gaussians data buffer",
        size: shBufferSize, // buffer size multiple of 4?
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const sh = new Float16Array(sh_buffer.getMappedRange());

    sh.fill(0);

    var readOffset = 0;
    for (let i = 0; i < num_points; i++) {
        let rawVertex: Record<string, number> = {};

        if (isCompressed) {
            // each vertex = 16 bytes
            const vertex = readRawCompressedVertex(i, vertexData, chunks);
            rawVertex.x = vertex.position[0];
            rawVertex.y = vertex.position[1];
            rawVertex.z = vertex.position[2];
            rawVertex.rot_0 = vertex.rotation[0];
            rawVertex.rot_1 = vertex.rotation[1];
            rawVertex.rot_2 = vertex.rotation[2];
            rawVertex.rot_3 = vertex.rotation[3];
            rawVertex.scale_0 = vertex.scale[0];
            rawVertex.scale_1 = vertex.scale[1];
            rawVertex.scale_2 = vertex.scale[2];
            rawVertex.opacity = vertex.color[3]; // or constant 1.0 if absent
        } else {
            const [newReadOffset, vertexProps] = readRawVertex(
                readOffset,
                vertexData,
                propertyTypes,
            );
            rawVertex = vertexProps;
            readOffset = newReadOffset;
        }

        // write to GPU buffers
        const o = i * (c_size_3d_gaussian / c_size_float);
        const output_offset = i * max_num_coefs * 3;

        if (!isCompressed) {
            for (let order = 0; order < num_coefs; ++order) {
                const order_offset = order * 3;
                for (let j = 0; j < 3; ++j) {
                    const coeffName = shFeatureOrder[order * 3 + j];
                    sh[output_offset + order_offset + j] =
                        rawVertex[coeffName] ?? 0;
                }
            }
        }

        gaussian[o + 0] = rawVertex.x;
        gaussian[o + 1] = rawVertex.y;
        gaussian[o + 2] = rawVertex.z;
        gaussian[o + 3] = rawVertex.opacity ?? 1.0;
        gaussian[o + 4] = rawVertex.rot_0;
        gaussian[o + 5] = rawVertex.rot_1;
        gaussian[o + 6] = rawVertex.rot_2;
        gaussian[o + 7] = rawVertex.rot_3;
        gaussian[o + 8] = rawVertex.scale_0;
        gaussian[o + 9] = rawVertex.scale_1;
        gaussian[o + 10] = rawVertex.scale_2;
    }

    gaussian_3d_buffer.unmap();
    sh_buffer.unmap();

    const result = {
        num_points: num_points,
        sh_deg: sh_deg,
        gaussian_3d_buffer,
        sh_buffer,
    };
    console.log("result:", result);

    return result;
}
