import { PointCloud } from "../utils/load";
import preprocessWGSL from "../shaders/preprocess.wgsl";
import renderWGSL from "../shaders/gaussian.wgsl";
import { get_sorter, c_histogram_block_rows, C } from "../sort/sort";
import { Renderer } from "./renderer";

export interface GaussianRenderer extends Renderer {
    splatParamsBuffer: GPUBuffer;
}

// Utility to create GPU buffers
const createBuffer = (
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer,
) => {
    const buffer = device.createBuffer({ label, size, usage });
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};

export default function get_renderer(
    pc: PointCloud,
    device: GPUDevice,
    presentation_format: GPUTextureFormat,
    camera_buffer: GPUBuffer,
): GaussianRenderer {
    const sorter = get_sorter(pc.num_points, device);

    // ===============================================
    //            Initialize GPU Buffers
    // ===============================================
    const nullingView = new Uint32Array([0]);
    const nullingBuffer = device.createBuffer({
        label: "indirect draw nulling buffer",
        size: 1 * Uint32Array.BYTES_PER_ELEMENT,
        usage:
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.INDIRECT,
    });
    device.queue.writeBuffer(nullingBuffer, 0, nullingView.buffer);

    const indirectDrawView = new Uint32Array([
        6, // vertexCount
        pc.num_points, // instanceCount
        0, // firstVertex
        0, // firstInstance
    ]);
    const indirectDrawBuffer = device.createBuffer({
        label: "indirect draw buffer",
        size: 16, // 4 elements * 4 bytes per elem
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.INDIRECT,
    });
    // write to device
    device.queue.writeBuffer(indirectDrawBuffer, 0, indirectDrawView.buffer);

    // `pc.num_points` number of quads
    // const quadView = new Uint32Array(pc.num_points * 6);
    const quadBytes = pc.num_points * 6 * 4;
    const quadBuffer = createBuffer(
        device,
        "quad buffer",
        quadBytes,
        GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.VERTEX,
    );

    const splatParamsView = new Float32Array([
        1.0, //gaussian_scaling
        pc.sh_deg, // sh_deg
    ]);
    const splatParamsBuffer = createBuffer(
        device,
        "splat params buffer",
        splatParamsView.byteLength, // simpler
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        splatParamsView.buffer,
    );

    // ===============================================
    //    Create Compute Pipeline and Bind Groups
    // ===============================================
    const preprocessDataDumpBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" }, // PC_sh_buffer
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" }, // PC_gaussian_3d_buffer
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" }, // quadBuffer
            },
        ],
    });

    const uniformsBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "uniform" }, // splatParams
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "uniform" }, // camUnifs
            },
        ],
    });

    const sortBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
        ],
    });

    const preprocessPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            preprocessDataDumpBindGroupLayout, // group(0)
            uniformsBindGroupLayout, // group(1)
            sortBindGroupLayout, // group(2)
        ],
    });
    const preprocessPipeline = device.createComputePipeline({
        label: "preprocess",
        layout: preprocessPipelineLayout,
        compute: {
            module: device.createShaderModule({ code: preprocessWGSL }),
            entryPoint: "preprocess",
            constants: {
                workgroupSize: C.histogram_wg_size,
                sortKeyPerThread: c_histogram_block_rows,
            },
        },
    });

    const preproocessDataDumpBindGroup = device.createBindGroup({
        label: "preprocess data dump compute group",
        layout: preprocessDataDumpBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: pc.sh_buffer } },
            { binding: 1, resource: { buffer: pc.gaussian_3d_buffer } },
            { binding: 2, resource: { buffer: quadBuffer } },
        ],
    });

    const uniformsBindGroup = device.createBindGroup({
        label: "uniforms bind group",
        layout: uniformsBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: camera_buffer } },
            { binding: 1, resource: { buffer: splatParamsBuffer } },
        ],
    });

    const sort_bind_group = device.createBindGroup({
        label: "sort",
        layout: sortBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
            {
                binding: 1,
                resource: { buffer: sorter.ping_pong[0].sort_depths_buffer },
            },
            {
                binding: 2,
                resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
            },
            {
                binding: 3,
                resource: { buffer: sorter.sort_dispatch_indirect_buffer },
            },
        ],
    });

    // ===============================================
    //    Create Render Pipeline and Bind Groups
    // ===============================================

    const gaussianRenderPipeline = device.createRenderPipeline({
        label: "gaussian render pipeline",
        layout: "auto", // simplified set
        vertex: {
            module: device.createShaderModule({
                label: "vert shader",
                code: renderWGSL,
            }),
            entryPoint: "vs_main",
            buffers: [],
        },
        fragment: {
            module: device.createShaderModule({
                label: "frag shader",
                code: renderWGSL,
            }),
            entryPoint: "fs_main",
            targets: [
                {
                    format: presentation_format,
                    blend: {
                        color: {
                            operation: "add",
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            operation: "add",
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                    },
                },
            ],
        },
    });

    const gaussianRenderingBindGroup = device.createBindGroup({
        label: "gaussian rendering bind group",
        layout: gaussianRenderPipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
            }, // sortedIndices
            { binding: 1, resource: { buffer: quadBuffer } }, // quadBuffer
            { binding: 2, resource: { buffer: camera_buffer } }, // camUnifs
        ],
    });

    let frame = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
        // ===============================================
        //    Command Encoder Functions
        // ===============================================
        encoder.copyBufferToBuffer(
            nullingBuffer,
            0,
            sorter.sort_info_buffer,
            0,
            4,
        );
        encoder.copyBufferToBuffer(
            nullingBuffer,
            0,
            sorter.sort_dispatch_indirect_buffer,
            0,
            4,
        );

        const preprocessComputePass = encoder.beginComputePass({
            label: "preprocess compute pass",
        });
        preprocessComputePass.setPipeline(preprocessPipeline);
        preprocessComputePass.setBindGroup(0, preproocessDataDumpBindGroup);
        preprocessComputePass.setBindGroup(1, uniformsBindGroup);
        preprocessComputePass.setBindGroup(2, sort_bind_group);

        preprocessComputePass.dispatchWorkgroups(
            Math.ceil(pc.num_points / C.histogram_wg_size),
        ); //divup

        preprocessComputePass.end();

        sorter.sort(encoder);

        encoder.copyBufferToBuffer(
            sorter.sort_info_buffer, // source
            0, // sourceOffset
            indirectDrawBuffer, // destination
            4, // destinationOffset
            4, // size
        ); // indirect draw call ready!

        const renderPass = encoder.beginRenderPass({
            label: "render pass",
            colorAttachments: [
                {
                    view: texture_view,
                    loadOp: "clear",
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    storeOp: "store",
                },
            ],
        });
        renderPass.setPipeline(gaussianRenderPipeline);
        renderPass.setBindGroup(0, gaussianRenderingBindGroup);

        renderPass.drawIndirect(indirectDrawBuffer, 0);
        renderPass.end();
    };

    // ===============================================
    //    Return Render Object
    // ===============================================
    return {
        frame,
        camera_buffer,
        splatParamsBuffer,
    };
}
