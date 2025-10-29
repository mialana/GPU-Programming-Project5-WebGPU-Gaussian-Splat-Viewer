import { load } from "../utils/load";
import { Pane } from "tweakpane";
import * as TweakpaneFileImportPlugin from "tweakpane-plugin-file-import";
import {
    default as get_renderer_gaussian,
    GaussianRenderer,
} from "./gaussian-renderer";
import { default as get_renderer_pointcloud } from "./point-cloud-renderer";
import { Camera, load_camera_presets } from "../camera/camera";
import { CameraControl } from "../camera/camera-control";
import {
    log,
    logSeparator,
    time,
    timeReturn,
    timeLog,
} from "../utils/simple-console";

export interface Renderer {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void;
    camera_buffer: GPUBuffer;
}

export default async function init(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
) {
    let ply_file_loaded = false;
    let cam_file_loaded = false;
    let renderers: { pointcloud?: Renderer; gaussian?: Renderer } = {};
    let gaussian_renderer: GaussianRenderer | undefined;
    let pointcloud_renderer: Renderer | undefined;
    let renderer: Renderer | undefined;
    let cameras;

    const camera = new Camera(canvas, device);
    const control = new CameraControl(camera);

    const observer = new ResizeObserver(() => {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;

        camera.on_update_canvas();
    });
    observer.observe(canvas);

    const presentation_format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentation_format,
        alphaMode: "opaque",
    });

    // Tweakpane: easily adding tweak control for parameters.
    const params = {
        fps: 0.0,
        gaussian_multiplier: 1,
        show_logs: true,
        renderer: "gaussian",
        ply_file: "",
        cam_file: "",
    };

    const pane = new Pane({
        title: "Config",
        expanded: true,
    });
    pane.registerPlugin(TweakpaneFileImportPlugin);
    {
        pane.addMonitor(params, "fps", {
            readonly: true,
        });
    }

    // stop view from changing when gaussian_multiplier is set with number keys
    pane.element.addEventListener("keydown", (event) => {
        event.stopPropagation();
    });

    async function load_ply(file: File) {
        if (file) {
            log(`loading: ${file.name}`);
            time();

            let pc;
            try {
                pc = await load(file, device);
            } catch (err) {
                if (err instanceof RangeError) {
                    const msgParts = err.message.split("\n");
                    msgParts.forEach((part) => {
                        log(part, 2); // log as a warning, but current scene can continue rendering
                    });
                    logSeparator();
                    return;
                }
            }

            pointcloud_renderer = get_renderer_pointcloud(
                pc,
                device,
                presentation_format,
                camera.uniform_buffer,
            );
            gaussian_renderer = get_renderer_gaussian(
                pc,
                device,
                presentation_format,
                camera.uniform_buffer,
            );
            renderers = {
                pointcloud: pointcloud_renderer,
                gaussian: gaussian_renderer,
            };
            renderer = renderers[params.renderer];
            ply_file_loaded = true;

            timeLog(`${file.name} load time`);
        } else {
            log(`invalid ply file.`);
            ply_file_loaded = false;
        }
        logSeparator();
    }

    async function load_cam(file: File) {
        if (file) {
            log(`loading: ${file.name}`);
            cameras = await load_camera_presets(file);
            camera.set_preset(0, cameras[0]);
            cam_file_loaded = true;
        } else {
            log(`invalid camera json.`);
            cam_file_loaded = false;
        }
        logSeparator();
    }
    {
        // load a default ply hosted on site
        async function loadDefaults(
            plyUrl: string = "resources/crochet/crochet.ply",
            camUrl: string = "resources/crochet/cameras.json",
        ) {
            try {
                const res = await fetch(plyUrl, { priority: "high" });
                if (!res.ok) throw new Error(`Failed to fetch ${plyUrl}`);
                const blob = await res.blob();
                const file = new File([blob], plyUrl.split("/").pop(), {
                    type: "application/octet-stream",
                });
                await load_ply(file);
            } catch (err) {
                console.error("Error loading default PLY:", err);
            }

            try {
                const res = await fetch(camUrl);
                if (!res.ok) throw new Error(`Failed to fetch ${camUrl}`);
                const blob = await res.blob();
                const file = new File([blob], "cameras.json", {
                    type: "application/octet-stream",
                });
                await load_cam(file);
            } catch (err) {
                console.error("Error loading default camera:", err);
            }
        }

        await loadDefaults();
    }

    {
        pane.addInput(params, "show_logs", {
            label: "show logs",
        }).on("change", () => {
            document
                .getElementById("log-container")
                .classList.toggle("hidden-class");
        });
    }
    {
        pane.addInput(params, "renderer", {
            options: {
                pointcloud: "pointcloud",
                gaussian: "gaussian",
            },
        }).on("change", (e) => {
            renderer = renderers[e.value];
        });
    }
    {
        pane.addInput(params, "ply_file", {
            label: "PLY File",
            view: "file-input",
            lineCount: 1,
            filetypes: [".ply"],
            invalidFiletypeMessage: "We can't accept those filetypes!",
        }).on("change", (file: any) => load_ply(file.value));
    }
    {
        pane.addInput(params, "cam_file", {
            label: "cam JSON",
            view: "file-input",
            lineCount: 1,
            filetypes: [".json"],
            invalidFiletypeMessage: "We can't accept those filetypes!",
        }).on("change", (file: any) => load_cam(file.value));
    }
    {
        pane.addInput(params, "gaussian_multiplier", {
            label: "gaussian multiplier",
            min: 0,
            max: 1.5,
        }).on("change", (e) => {
            //TODO: Bind constants to the gaussian renderer.
            if (!gaussian_renderer) {
                return;
            }
            device.queue.writeBuffer(
                gaussian_renderer.splatParamsBuffer,
                0,
                new Float32Array([params.gaussian_multiplier]),
            );
        });
    }

    document.addEventListener("keydown", (event) => {
        switch (event.key) {
            case "0":
            case "1":
            case "2":
            case "3":
            case "4":
            case "5":
            case "6":
            case "7":
            case "8":
            case "9":
                const i = parseInt(event.key);
                camera.set_preset(i, cameras[i]);
                break;
        }
    });

    function frame() {
        if (ply_file_loaded && cam_file_loaded) {
            params.fps = (1.0 / timeReturn()) * 1000.0;
            time();
            const encoder = device.createCommandEncoder();
            const texture_view = context.getCurrentTexture().createView();
            renderer.frame(encoder, texture_view);
            device.queue.submit([encoder.finish()]);
        }
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}
