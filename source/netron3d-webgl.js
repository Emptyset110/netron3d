
// netron3d-webgl — a tiny instanced-cube engine.
//
// Enough WebGL2 to draw thousands of little cubes, each an element of a tensor,
// with an orbit camera. Deliberately small and dependency-free: netron3d is
// about *what* the cubes mean (the kernels), and this is just the plumbing that
// puts them in 3D. No matrix library, no scene graph — one instanced draw call.

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

// --- one unit cube, with per-face normals ------------------------------------

const CUBE = (() => {
    // 6 faces × 4 corners. Positions centred on the origin, side 1.
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

const VERT = `#version 300 es
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

const FRAG = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vColor;
uniform vec3 uLight;
out vec4 outColor;
void main() {
  float d = max(dot(normalize(vNormal), normalize(uLight)), 0.0);
  outColor = vec4(vColor * (0.35 + 0.65 * d), 1.0);
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

// Create an engine on a canvas. `instances` is { offsets, scales, colors } —
// three Float32Arrays of length 3*N. Returns a handle with dispose().
export function createEngine(canvas) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) {
        return null;
    }
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(`netron3d link: ${gl.getProgramInfoLog(program)}`);
    }

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const buffer = (data, loc, size, divisor) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, divisor ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
        if (divisor) {
            gl.vertexAttribDivisor(loc, 1);
        }
        return b;
    };

    buffer(CUBE.positions, 0, 3, 0);
    buffer(CUBE.normals, 1, 3, 0);
    const offsetBuf = buffer(new Float32Array(0), 2, 3, 1);
    const scaleBuf = buffer(new Float32Array(0), 3, 3, 1);
    const colorBuf = buffer(new Float32Array(0), 4, 3, 1);

    const index = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, CUBE.indices, gl.STATIC_DRAW);

    const uProj = gl.getUniformLocation(program, 'uProj');
    const uView = gl.getUniformLocation(program, 'uView');
    const uLight = gl.getUniformLocation(program, 'uLight');

    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.051, 0.067, 0.09, 1);

    const state = {
        count: 0,
        target: [0, 0, 0],
        yaw: 0.7,
        pitch: 0.5,
        dist: 40,
        dragging: false,
        auto: true,
        raf: 0,
        disposed: false,
    };

    const setInstances = (instances) => {
        state.count = instances.offsets.length / 3;
        gl.bindBuffer(gl.ARRAY_BUFFER, offsetBuf);
        gl.bufferData(gl.ARRAY_BUFFER, instances.offsets, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuf);
        gl.bufferData(gl.ARRAY_BUFFER, instances.scales, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, instances.colors, gl.DYNAMIC_DRAW);
        if (instances.target) {
            state.target = instances.target;
        }
        if (instances.distance) {
            state.dist = instances.distance;
        }
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

    const render = () => {
        if (state.disposed) {
            return;
        }
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

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(program);
        gl.uniformMatrix4fv(uProj, false, proj);
        gl.uniformMatrix4fv(uView, false, view);
        gl.uniform3f(uLight, 0.4, 0.9, 0.6);
        gl.bindVertexArray(vao);
        gl.drawElementsInstanced(gl.TRIANGLES, CUBE.indices.length, gl.UNSIGNED_SHORT, 0, state.count);

        state.raf = requestAnimationFrame(render);
    };

    // Orbit controls.
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
        dispose() {
            state.disposed = true;
            cancelAnimationFrame(state.raf);
        },
    };
}
