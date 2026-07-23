
// netron3d — a real neural network, drawn in 3D.
//
// Netron lays any model out as a clean 2D graph. This adds a second, opt-in
// rendering: for architectures we recognise, a hand-laid 3D scene sized from
// the model's own numbers. It is deliberately NOT a general graph-to-3D layout
// (which reads worse than 2D). It detects a *family* and draws a purpose-built
// scene for it; anything unrecognised returns false so the caller keeps Netron's
// 2D graph, which is the right rendering for it.
//
// This module is self-contained: it injects its own styles, builds plain DOM,
// and reaches for no external asset or library.

const netron3d = {};

// --- operator signature ------------------------------------------------------

// Normalise a graph to a lowercase list of operator names. Accepts either a
// plain string array (for tests) or a Netron model graph whose nodes carry a
// `type` (a string, or an object with `.name`).
netron3d.opTypes = (graph) => {
    if (Array.isArray(graph)) {
        return graph.map((s) => String(s).toLowerCase());
    }
    const nodes = (graph && graph.nodes) || [];
    return nodes.map((node) => {
        const type = node && node.type;
        const name = (type && (type.name || type)) || '';
        return String(name).toLowerCase();
    });
};

// --- family detection --------------------------------------------------------

// Identify the architecture family from the operator signature, and estimate
// the few numbers a scene needs. Estimates are marked as such: a graph gives
// shapes, not always the clean layer count a config would, so the caller can
// present them honestly.
netron3d.detect = (graph, label) => {
    const ops = netron3d.opTypes(graph);
    const count = (predicate) => ops.filter(predicate).length;

    const attention = count((o) => o.includes('attention') || o.includes('multihead'));
    const softmax = count((o) => o.includes('softmax'));
    const matmul = count((o) => o.includes('matmul') || o.includes('gemm'));
    const norm = count((o) =>
        o.includes('layernorm') || o.includes('rmsnorm') || o.includes('skiplayernorm'));

    // An explicit attention op is decisive; otherwise the softmax + matmul + norm
    // pattern, repeated, is a transformer's fingerprint.
    const isTransformer = attention > 0 || (softmax > 0 && matmul > 1 && norm > 1);
    if (!isTransformer) {
        return { family: 'unknown', dims: {}, label: label || '' };
    }

    // One attention op per block, or a pair of norms per block — whichever the
    // graph exposes. Left undefined when neither is legible.
    const layers = Math.max(attention, Math.round(norm / 2)) || 0;
    const dims = layers > 0 ? { n_layers: layers, estimated: true } : {};
    return { family: 'transformer', dims, label: label || '' };
};

// --- rendering ---------------------------------------------------------------

netron3d._styleId = 'netron3d-style';

netron3d._css = `
  .n3d-scene{position:relative;width:100%;height:100%;overflow:hidden;background:#0d1117;
    display:flex;align-items:center;justify-content:center;perspective:1500px;cursor:grab;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d1d9}
  .n3d-scene:active{cursor:grabbing}
  .n3d-caption{position:absolute;top:10px;left:12px;font-size:12px;line-height:1.5;z-index:2}
  .n3d-caption b{color:#e6edf3} .n3d-caption .sub{color:#8b949e}
  .n3d-stage{transform-style:preserve-3d;animation:n3d-spin 26s linear infinite}
  @keyframes n3d-spin{from{transform:rotateX(-16deg) rotateY(0deg)}
    to{transform:rotateX(-16deg) rotateY(360deg)}}
  .n3d-io,.n3d-layer,.n3d-ellipsis{width:230px;margin:7px auto;transform-style:preserve-3d}
  .n3d-io{height:26px;display:flex;align-items:center;justify-content:center;font-size:11px;
    color:#8b949e;background:#161b22;border:1px solid #30363d;border-radius:5px}
  .n3d-layer{height:34px;display:flex;align-items:stretch;gap:6px;padding:5px;
    background:linear-gradient(180deg,#1b2230,#141a24);border:1px solid #30363d;
    border-radius:6px;box-shadow:0 10px 24px rgba(0,0,0,.35)}
  .n3d-attn{flex:2;display:flex;flex-direction:column;justify-content:center;gap:3px;
    background:#12233b;border:1px solid #1f6feb55;border-radius:4px;padding:0 6px}
  .n3d-mlp{flex:1;display:flex;align-items:center;justify-content:center;
    background:#22182f;border:1px solid #a371f755;border-radius:4px}
  .n3d-lbl{font-size:9px;color:#8b949e;letter-spacing:.04em}
  .n3d-heads{display:flex;gap:2px}
  .n3d-h{width:5px;height:12px;background:#58a6ff;border-radius:1px;opacity:.85}
  .n3d-ellipsis{text-align:center;color:#8b949e;font-size:12px;padding:6px 0}
`;

netron3d._ensureStyle = (element) => {
    const doc = element.ownerDocument || document;
    if (doc.getElementById(netron3d._styleId)) {
        return;
    }
    const style = doc.createElement('style');
    style.id = netron3d._styleId;
    style.textContent = netron3d._css;
    (doc.head || doc.documentElement).appendChild(style);
};

// Let the reader take over the rotation with a drag.
netron3d._enableDrag = (scene, stage) => {
    let rx = -16;
    let ry = 0;
    let dragging = false;
    let px = 0;
    let py = 0;
    scene.addEventListener('pointerdown', (e) => {
        dragging = true;
        px = e.clientX;
        py = e.clientY;
        stage.style.animation = 'none';
        scene.setPointerCapture(e.pointerId);
    });
    scene.addEventListener('pointermove', (e) => {
        if (!dragging) {
            return;
        }
        ry += (e.clientX - px) * 0.5;
        rx -= (e.clientY - py) * 0.5;
        px = e.clientX;
        py = e.clientY;
        stage.style.transform = `rotateX(${rx}deg) rotateY(${ry}deg)`;
    });
    scene.addEventListener('pointerup', () => { dragging = false; });
};

netron3d.families = {};

// The transformer family: embedding, a stack of attention+MLP blocks, unembedding.
netron3d.families.transformer = {
    name: 'transformer',
    render(element, spec) {
        netron3d._ensureStyle(element);
        const doc = element.ownerDocument || document;
        const dims = (spec && spec.dims) || {};
        const exact = dims.n_layers !== undefined && dims.n_layers !== null;
        const estimated = Boolean(dims.estimated);
        const nLayers = Number(dims.n_layers) || 8;
        const nHeads = Math.min(Math.max(Number(dims.n_heads) || 12, 1), 16);
        const nEmbd = dims.n_embd;
        const label = (spec && spec.label) || 'GPT-style transformer';

        const MAX_SLABS = 14;
        const slab = () => {
            const heads = new Array(nHeads).fill('<i class="n3d-h"></i>').join('');
            return `<div class="n3d-layer">` +
                `<div class="n3d-attn"><span class="n3d-lbl">attn</span>` +
                `<div class="n3d-heads">${heads}</div></div>` +
                `<div class="n3d-mlp"><span class="n3d-lbl">mlp</span></div></div>`;
        };
        let slabs;
        if (nLayers <= MAX_SLABS) {
            slabs = new Array(nLayers).fill(0).map(slab).join('');
        } else {
            const head = new Array(MAX_SLABS - 4).fill(0).map(slab).join('');
            const tail = new Array(3).fill(0).map(slab).join('');
            slabs = head + `<div class="n3d-ellipsis">⋮ ${nLayers} blocks</div>` + tail;
        }

        let sub;
        if (exact) {
            sub = `${nLayers} layers · ${nHeads} heads`;
            if (nEmbd) {
                sub += ` · d_model ${Number(nEmbd)}`;
            }
            if (estimated) {
                sub += ' (estimated from graph)';
            }
        } else {
            sub = 'decoder-only transformer — illustrative depth';
        }

        const safeLabel = String(label).replace(/[&<>]/g, (c) =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

        element.innerHTML =
            `<div class="n3d-scene"><div class="n3d-caption"><b>${safeLabel}</b><br>` +
            `<span class="sub">${sub}</span><br><span class="sub">drag to rotate</span></div>` +
            `<div class="n3d-stage">` +
            `<div class="n3d-io">token + position embedding</div>${slabs}` +
            `<div class="n3d-io">final norm → unembedding</div></div></div>`;

        const scene = element.querySelector('.n3d-scene');
        const stage = element.querySelector('.n3d-stage');
        netron3d._enableDrag(scene, stage);
        return true;
    }
};

// Render `spec` into `element`. Returns false when no family matches, so the
// caller keeps Netron's 2D graph.
netron3d.render = (element, spec) => {
    const family = spec && netron3d.families[spec.family];
    if (!family) {
        return false;
    }
    return family.render(element, spec);
};

export const { detect, families, opTypes, render } = netron3d;
export default netron3d;
