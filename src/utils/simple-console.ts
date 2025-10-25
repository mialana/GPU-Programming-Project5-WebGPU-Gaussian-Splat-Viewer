const html_log = document.querySelector("#log") as HTMLDivElement;

const colorToLevelMap = ["green", "orange", "red"];

export async function log(msg: string, level = 0) {
    console.log(`%c${msg}`, `color: ${colorToLevelMap[level]};`);
    const p = document.createElement("p");
    p.innerText = msg;
    if (level > 0) {
        p.style.color = colorToLevelMap[level];
    }
    html_log.appendChild(p);
}

export function logSeparator() {
    const hr = document.createElement("hr");
    html_log.appendChild(hr);
}

let t: number;

export function time() {
    t = performance.now();
}

export function timeLog(prefix_text: string) {
    const d = performance.now() - t;
    log(`${prefix_text}: ${d.toFixed(0)} ms`);
}

export function timeReturn() {
    const d = performance.now() - t;
    return d;
}
