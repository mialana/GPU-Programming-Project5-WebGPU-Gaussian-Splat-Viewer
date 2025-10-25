export function decodeHeader(
    plyArrayBuffer: ArrayBuffer,
    isCompressed: boolean = false,
): [number, Record<string, string>, DataView] {
    /* decodes the .ply file header and returns a tuple of:
     * - vertexCount: number of vertices in the point cloud
     * - propertyTypes: a map from property names to their types
     * - vertexData: a DataView of the vertex data
     */

    const decoder = new TextDecoder();
    let headerOffset = 0;
    let headerText = "";

    while (true) {
        const headerChunk = new Uint8Array(plyArrayBuffer, headerOffset, 50);
        headerText += decoder.decode(headerChunk);
        headerOffset += 50;

        if (headerText.includes("end_header")) {
            break;
        }
    }

    const headerLines = headerText.split("\n");

    let vertexCount = 0;
    let propertyTypes: Record<string, string> = {};

    for (let i = 0; i < headerLines.length; i++) {
        const line = headerLines[i].trim();
        if (line.startsWith("element vertex")) {
            const vertexCountMatch = line.match(/\d+/);
            if (vertexCountMatch) {
                vertexCount = parseInt(vertexCountMatch[0]);
            }
        } else if (line.startsWith("property")) {
            const propertyMatch = line.match(/(\w+)\s+(\w+)\s+(\w+)/);
            if (propertyMatch) {
                const propertyType = propertyMatch[2];
                const propertyName = propertyMatch[3];
                propertyTypes[propertyName] = propertyType;
            }
        } else if (line === "end_header") {
            break;
        }
    }

    let vertexByteOffset =
        headerText.indexOf("end_header") + "end_header".length + 1;

    if (isCompressed) {
        // Read chunk count from header
        const chunkLine = headerLines.find((l) =>
            l.startsWith("element chunk"),
        );
        const chunkCount = chunkLine ? parseInt(chunkLine.match(/\d+/)![0]) : 0;
        const CHUNK_SIZE = 72; // 18 floats × 4 bytes
        vertexByteOffset += chunkCount * CHUNK_SIZE;
    }

    const vertexData = new DataView(plyArrayBuffer, vertexByteOffset);

    return [vertexCount, propertyTypes, vertexData];
}

export function readRawVertex(
    offset: number,
    vertexData: DataView,
    propertyTypes: Record<string, string>,
): [number, Record<string, number>] {
    /* reads a single vertex from the vertexData DataView and returns a tuple of:
     * - offset: the offset of the next vertex in the vertexData DataView
     * - rawVertex: a map from property names to their values
     */
    let rawVertex: Record<string, number> = {};

    for (const property in propertyTypes) {
        const propertyType = propertyTypes[property];
        if (propertyType === "float") {
            rawVertex[property] = vertexData.getFloat32(offset, true);
            offset += Float32Array.BYTES_PER_ELEMENT;
        } else if (propertyType === "uchar") {
            rawVertex[property] = vertexData.getUint8(offset) / 255.0;
            offset += Uint8Array.BYTES_PER_ELEMENT;
        } else if (propertyType === "uint") {
            rawVertex[property] = vertexData.getUint32(offset, true);
            offset += Uint32Array.BYTES_PER_ELEMENT;
        }
    }

    return [offset, rawVertex];
}

// ────────────────────────────────────────────────
// Compressed PLY: read all chunk bounds
// ────────────────────────────────────────────────
export function readChunks(
    plyArrayBuffer: ArrayBuffer,
    headerText: string,
): {
    minPos: number[];
    maxPos: number[];
    minScale: number[];
    maxScale: number[];
    minColor: number[];
    maxColor: number[];
}[] {
    const chunkLine = headerText
        .split("\n")
        .find((l) => l.startsWith("element chunk"));
    const chunkCount = chunkLine ? parseInt(chunkLine.match(/\d+/)![0]) : 0;
    const CHUNK_SIZE = 72;

    const start = headerText.indexOf("end_header") + "end_header".length + 1;
    const view = new DataView(plyArrayBuffer, start);
    const chunks = [];

    for (let i = 0; i < chunkCount; i++) {
        const base = i * CHUNK_SIZE;
        const f = (j: number) => view.getFloat32(base + j * 4, true);
        chunks.push({
            minPos: [f(0), f(1), f(2)],
            maxPos: [f(3), f(4), f(5)],
            minScale: [f(6), f(7), f(8)],
            maxScale: [f(9), f(10), f(11)],
            minColor: [f(12), f(13), f(14)],
            maxColor: [f(15), f(16), f(17)],
        });
    }

    return chunks;
}

// ────────────────────────────────────────────────
// Compressed PLY: unpack a single vertex (16 bytes)
// ────────────────────────────────────────────────
export function readRawCompressedVertex(
    vertexIndex: number,
    vertexData: DataView,
    chunks: ReturnType<typeof readChunks>,
): {
    position: [number, number, number];
    scale: [number, number, number];
    rotation: [number, number, number, number];
    color: [number, number, number, number];
} {
    const CHUNK_SPLAT_CAP = 256;
    const VERTEX_SIZE = 16;
    const chunkIndex = Math.floor(vertexIndex / CHUNK_SPLAT_CAP);
    const chunk = chunks[chunkIndex];
    const offset = vertexIndex * VERTEX_SIZE;

    const pos = vertexData.getUint32(offset, true);
    const rot = vertexData.getUint32(offset + 4, true);
    const scl = vertexData.getUint32(offset + 8, true);
    const col = vertexData.getUint32(offset + 12, true);

    // --- Position (11,10,11) ---
    const xBits = (pos >> 21) & 0x7ff;
    const yBits = (pos >> 11) & 0x3ff;
    const zBits = pos & 0x7ff;
    const nx = xBits / 2047.0,
        ny = yBits / 1023.0,
        nz = zBits / 2047.0;
    const position: [number, number, number] = [
        chunk.minPos[0] + nx * (chunk.maxPos[0] - chunk.minPos[0]),
        chunk.minPos[1] + ny * (chunk.maxPos[1] - chunk.minPos[1]),
        chunk.minPos[2] + nz * (chunk.maxPos[2] - chunk.minPos[2]),
    ];

    // --- Rotation (2+10+10+10) ---
    const omitted = (rot >> 30) & 0x3;
    const r0 = ((rot >> 20) & 0x3ff) / 511.5 - 1;
    const r1 = ((rot >> 10) & 0x3ff) / 511.5 - 1;
    const r2 = (rot & 0x3ff) / 511.5 - 1;
    const q: [number, number, number, number] = [0, 0, 0, 0];
    let t = 0;
    for (let j = 0; j < 4; j++) {
        if (j === omitted) continue;
        q[j] = [r0, r1, r2][t++];
    }
    q[omitted] = Math.sqrt(
        Math.max(0, 1 - (q[0] ** 2 + q[1] ** 2 + q[2] ** 2)),
    );

    // --- Scale (11,10,11) ---
    const sxBits = (scl >> 21) & 0x7ff;
    const syBits = (scl >> 11) & 0x3ff;
    const szBits = scl & 0x7ff;
    const sx = sxBits / 2047.0,
        sy = syBits / 1023.0,
        sz = szBits / 2047.0;
    const scale: [number, number, number] = [
        chunk.minScale[0] + sx * (chunk.maxScale[0] - chunk.minScale[0]),
        chunk.minScale[1] + sy * (chunk.maxScale[1] - chunk.minScale[1]),
        chunk.minScale[2] + sz * (chunk.maxScale[2] - chunk.minScale[2]),
    ];

    // --- Color (8,8,8,8) ---
    const r = (col >> 24) & 0xff;
    const g = (col >> 16) & 0xff;
    const b = (col >> 8) & 0xff;
    const a = col & 0xff;
    const color: [number, number, number, number] = [
        r / 255,
        g / 255,
        b / 255,
        a / 255,
    ];

    return { position, scale, rotation: q, color };
}

export function nShCoeffs(sphericalHarmonicsDegree: number): number {
    /* returns the expected number of spherical harmonics coefficients */
    if (sphericalHarmonicsDegree === 0) {
        return 1;
    } else if (sphericalHarmonicsDegree === 1) {
        return 4;
    } else if (sphericalHarmonicsDegree === 2) {
        return 9;
    } else if (sphericalHarmonicsDegree === 3) {
        return 16;
    } else {
        throw new Error(`Unsupported SH degree: ${sphericalHarmonicsDegree}`);
    }
}
