import { PointCloud } from "../utils/load";
import preprocessWGSL from "../shaders/preprocess.wgsl";
import renderWGSL from "../shaders/gaussian.wgsl";
import { get_sorter, c_histogram_block_rows, C } from "../sort/sort";
import { Renderer } from "./renderer";

export interface GaussianRenderer extends Renderer {}

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

    const nulling_data = new Uint32Array([0]);

    const preprocessPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            device.createBindGroupLayout({ entries: [] }), // group(0)
            device.createBindGroupLayout({ entries: [] }), // group(1)
            device.createBindGroupLayout({
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
            }),
        ],
    });

    // ===============================================
    //    Create Compute Pipeline and Bind Groups
    // ===============================================
    const preprocess_pipeline = device.createComputePipeline({
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

    const sort_bind_group = device.createBindGroup({
        label: "sort",
        layout: preprocess_pipeline.getBindGroupLayout(2),
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

    // ===============================================
    //    Command Encoder Functions
    // ===============================================

    // ===============================================
    //    Return Render Object
    // ===============================================
    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            sorter.sort(encoder);
        },
        camera_buffer,
    };
}
