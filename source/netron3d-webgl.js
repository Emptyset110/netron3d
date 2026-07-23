
// netron3d-webgl — a tiny instanced-cube + lines engine.
//
// Enough WebGL2 to draw thousands of little cubes (the cells of a tensor) and a
// set of connection lines between them, with a travelling pulse that shows which
// way the data flows, an orbit camera, and a 3D→screen projection so the caller
// can pin HTML size-labels to tensors. Deliberately small and dependency-free:
// netron3d is about *what* the geometry means; this is the plumbing.

// --- 4x4 matrices (column-major, WebGL order) --------------------------------

const mat4 = {};

mat4.perspective = (fovy, aspect, near, far) => {
    const f = 1 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ]);
};

mat4.lookAt = (eye, center, up) => {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const norm = (v) => {
        const l = Math.hypot(v[0], v[1], v[2]) || 1;
        return [v[0] / l, v[1] / l, v[2] / l];
    };
    const cross = (a, b) => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = norm(sub(eye, center));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
};

mat4.multiply = (a, b) => {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            out[c * 4 + r] =
                a[0 * 4 + r] * b[c * 4 + 0] +
                a[1 * 4 + r] * b[c * 4 + 1] +
                a[2 * 4 + r] * b[c * 4 + 2] +
                a[3 * 4 + r] * b[c * 4 + 3];
        }
    }
    return out;
};

// --- one unit cube, with per-face normals ------------------------------------

const CUBE = (() => {
    const faces = [
        { n: [0, 0, 1], v: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]] },
        { n: [0, 0, -1], v: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]] },
        { n: [0, 1, 0], v: [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]] },
        { n: [0, -1, 0], v: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]] },
        { n: [1, 0, 0], v: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]] },
        { n: [-1, 0, 0], v: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]] },
    ];
    const positions = [];
    const normals = [];
    const indices = [];
    faces.forEach((face, f) => {
        for (const corner of face.v) {
            positions.push(...corner);
            normals.push(...face.n);
        }
        const o = f * 4;
        indices.push(o, o + 1, o + 2, o, o + 2, o + 3);
    });
    return {
        positions: new Float32Array(positions),
        normals: new Float32Array(normals),
        indices: new Uint16Array(indices),
    };
})();

const CUBE_VERT = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aOffset;
layout(location=3) in vec3 aScale;
layout(location=4) in vec3 aColor;
uniform mat4 uProj;
uniform mat4 uView;
out vec3 vNormal;
out vec3 vColor;
void main() {
  vec3 world = aPos * aScale + aOffset;
  gl_Position = uProj * uView * vec4(world, 1.0);
  vNormal = aNormal;
  vColor = aColor;
}`;

const CUBE_FRAG = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vColor;
uniform vec3 uLight;
out vec4 outColor;
void main() {
  float d = max(dot(normalize(vNormal), normalize(uLight)), 0.0);
  outColor = vec4(vColor * (0.35 + 0.65 * d), 1.0);
}`;

// Connection lines. `aAlong` runs 0→1 from source to destination, so a pulse
// can travel along it and show the direction of dataflow.
const LINE_VERT = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aAlong;
uniform mat4 uProj;
uniform mat4 uView;
out vec3 vColor;
out float vAlong;
void main() {
  gl_Position = uProj * uView * vec4(aPos, 1.0);
  vColor = aColor;
  vAlong = aAlong;
}`;

const LINE_FRAG = `#version 300 es
precision highp float;
in vec3 vColor;
in float vAlong;
uniform float uTime;
out vec4 outColor;
void main() {
  float p = fract(vAlong - uTime * 0.5);
  float pulse = smoothstep(0.0, 0.06, p) * (1.0 - smoothstep(0.06, 0.22, p));
  outColor = vec4(vColor * 0.5 + vColor * pulse * 1.8, 1.0);
}`;

function compile(gl, type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(`netron3d shader: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
}

function link(gl, vertSrc, fragSrc) {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertSrc));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragSrc));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(`netron3d link: ${gl.getProgramInfoLog(program)}`);
    }
    return program;
}

// Create an engine on a canvas. Returns a handle, or null if WebGL2 is absent.
export function createEngine(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) {
        return null;
    }

    const cubeProgram = link(gl, CUBE_VERT, CUBE_FRAG);
    const lineProgram = link(gl, LINE_VERT, LINE_FRAG);

    const arrayBuffer = (loc, data, size, divisor, vao) => {
        gl.bindVertexArray(vao);
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        if (divisor) {
            gl.vertexAttribDivisor(loc, 1);
        }
        return b;
    };

    // Cube VAO.
    const cubeVao = gl.createVertexArray();
    arrayBuffer(0, CUBE.positions, 3, 0, cubeVao);
    arrayBuffer(1, CUBE.normals, 3, 0, cubeVao);
    const offsetBuf = arrayBuffer(2, new Float32Array(0), 3, 1, cubeVao);
    const scaleBuf = arrayBuffer(3, new Float32Array(0), 3, 1, cubeVao);
    const cubeColorBuf = arrayBuffer(4, new Float32Array(0), 3, 1, cubeVao);
    gl.bindVertexArray(cubeVao);
    const cubeIndex = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeIndex);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, CUBE.indices, gl.STATIC_DRAW);

    // Line VAO.
    const lineVao = gl.createVertexArray();
    const linePosBuf = arrayBuffer(0, new Float32Array(0), 3, 0, lineVao);
    const lineColorBuf = arrayBuffer(1, new Float32Array(0), 3, 0, lineVao);
    const lineAlongBuf = arrayBuffer(2, new Float32Array(0), 1, 0, lineVao);

    const uniforms = (program) => ({
        proj: gl.getUniformLocation(program, 'uProj'),
        view: gl.getUniformLocation(program, 'uView'),
        light: gl.getUniformLocation(program, 'uLight'),
        time: gl.getUniformLocation(program, 'uTime'),
    });
    const cubeU = uniforms(cubeProgram);
    const lineU = uniforms(lineProgram);

    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.051, 0.067, 0.09, 1);

    const state = {
        cubeCount: 0,
        lineCount: 0,
        target: [0, 0, 0],
        yaw: 0.7,
        pitch: 0.5,
        dist: 40,
        dragging: false,
        auto: true,
        raf: 0,
        t0: 0,
        disposed: false,
        viewProj: mat4.perspective(1, 1, 0.1, 100),
        onFrame: null,
    };

    const setInstances = (s) => {
        state.cubeCount = s.offsets.length / 3;
        gl.bindBuffer(gl.ARRAY_BUFFER, offsetBuf);
        gl.bufferData(gl.ARRAY_BUFFER, s.offsets, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuf);
        gl.bufferData(gl.ARRAY_BUFFER, s.scales, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeColorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, s.colors, gl.DYNAMIC_DRAW);
        if (s.target) {
            state.target = s.target;
        }
        if (s.distance) {
            state.dist = s.distance;
        }
    };

    const setLines = (l) => {
        state.lineCount = l.positions.length / 3;
        gl.bindBuffer(gl.ARRAY_BUFFER, linePosBuf);
        gl.bufferData(gl.ARRAY_BUFFER, l.positions, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineColorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, l.colors, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineAlongBuf);
        gl.bufferData(gl.ARRAY_BUFFER, l.along, gl.DYNAMIC_DRAW);
    };

    // World point → CSS-pixel screen coordinate (for HTML labels).
    const project = (p) => {
        const m = state.viewProj;
        const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
        const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
        const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
        if (w <= 0) {
            return { x: 0, y: 0, visible: false };
        }
        return {
            x: (x / w * 0.5 + 0.5) * canvas.clientWidth,
            y: (1 - (y / w * 0.5 + 0.5)) * canvas.clientHeight,
            visible: true,
        };
    };

    const resize = () => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        gl.viewport(0, 0, canvas.width, canvas.height);
    };

    const render = (now) => {
        if (state.disposed) {
            return;
        }
        if (!state.t0) {
            state.t0 = now || 0;
        }
        const time = ((now || 0) - state.t0) / 1000;

        resize();
        if (state.auto && !state.dragging) {
            state.yaw += 0.0035;
        }
        const aspect = canvas.width / canvas.height || 1;
        const proj = mat4.perspective(0.9, aspect, 0.1, 4000);
        const cp = Math.cos(state.pitch);
        const eye = [
            state.target[0] + state.dist * cp * Math.sin(state.yaw),
            state.target[1] + state.dist * Math.sin(state.pitch),
            state.target[2] + state.dist * cp * Math.cos(state.yaw),
        ];
        const view = mat4.lookAt(eye, state.target, [0, 1, 0]);
        state.viewProj = mat4.multiply(proj, view);

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(cubeProgram);
        gl.uniformMatrix4fv(cubeU.proj, false, proj);
        gl.uniformMatrix4fv(cubeU.view, false, view);
        gl.uniform3f(cubeU.light, 0.4, 0.9, 0.6);
        gl.bindVertexArray(cubeVao);
        gl.drawElementsInstanced(gl.TRIANGLES, CUBE.indices.length, gl.UNSIGNED_SHORT, 0, state.cubeCount);

        if (state.lineCount > 0) {
            gl.useProgram(lineProgram);
            gl.uniformMatrix4fv(lineU.proj, false, proj);
            gl.uniformMatrix4fv(lineU.view, false, view);
            gl.uniform1f(lineU.time, time);
            gl.bindVertexArray(lineVao);
            gl.drawArrays(gl.LINES, 0, state.lineCount);
        }

        if (state.onFrame) {
            state.onFrame(project);
        }
        state.raf = requestAnimationFrame(render);
    };

    let px = 0;
    let py = 0;
    canvas.addEventListener('pointerdown', (e) => {
        state.dragging = true;
        state.auto = false;
        px = e.clientX;
        py = e.clientY;
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
        if (!state.dragging) {
            return;
        }
        state.yaw += (e.clientX - px) * 0.008;
        state.pitch = Math.max(-1.4, Math.min(1.4, state.pitch + (e.clientY - py) * 0.008));
        px = e.clientX;
        py = e.clientY;
    });
    canvas.addEventListener('pointerup', () => { state.dragging = false; });
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        state.dist = Math.max(4, Math.min(2000, state.dist * (1 + Math.sign(e.deltaY) * 0.1)));
    }, { passive: false });

    state.raf = requestAnimationFrame(render);

    return {
        setInstances,
        setLines,
        project,
        onFrame(cb) { state.onFrame = cb; },
        dispose() {
            state.disposed = true;
            state.onFrame = null;
            cancelAnimationFrame(state.raf);
        },
    };
}
