import rawPlugin from "vite-raw-plugin";
import { defineConfig } from "vite";

export default defineConfig({
    build: {
        target: "esnext",
    },
    base: "",
    plugins: [
        rawPlugin({
            fileRegex: /\.wgsl$/,
        }),
    ],
});
