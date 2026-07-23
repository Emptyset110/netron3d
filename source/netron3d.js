
// netron3d — a real neural network, drawn in 3D.
//
// Netron lays any model out as a clean 2D graph. This adds a second, opt-in
// rendering that treats the model's *tensors as the 3D objects*: each weight
// matrix and activation is a grid of cells at its real (downsampled) dimensions,
// arranged to show the data flow up through the network — the way llm-viz draws
// a transformer, but sized from this model's own numbers and rendered from our
// own WebGL.
//
// It is deliberately NOT a general graph-to-3D layout (which reads worse than
// 2D). It detects a *family* and draws a purpose-built volumetric scene for it;
// anything unrecognised returns false so the caller keeps Netron's 2D graph.

import { createEngine } from './netron3d-webgl.js';

const netron3d = {};

// --- operator signature ------------------------------------------------------

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

netron3d.detect = (graph, label) => {
    const ops = netron3d.opTypes(graph);
    const count = (predicate) => ops.filter(predicate).length;

    const attention = count((o) => o.includes('attention') || o.includes('multihead'));
    const softmax = count((o) => o.includes('softmax'));
    const matmul = count((o) => o.includes('matmul') || o.includes('gemm'));
    const norm = count((o) =>
        o.includes('layernorm') || o.includes('rmsnorm') || o.includes('skiplayernorm'));

    const isTransformer = attention > 0 || (softmax > 0 && matmul > 1 && norm > 1);
    if (!isTransformer) {
        return { family: 'unknown', dims: {}, label: label || '' };
    }
    const layers = Math.max(attention, Math.round(norm / 2)) || 0;
    const dims = layers > 0 ? { n_layers: layers, estimated: true } : {};
    return { family: 'transformer', dims, label: label || '' };
};

// --- the transformer scene ---------------------------------------------------
//
// Every tensor is a grid of cells. Real dimensions are downsampled to a legible
// number of cells per axis (a 3072-wide matrix would be a wall); the caption
// says so, and the *shape* — which matrix is tall, which is wide, how the heads
// stack — is preserved. Coordinates: Y is up (the flow), X is width/features,
// Z is depth (tokens / heads).

const CAP = { dmodel: 16, seq: 8, dhead: 8, dff: 16, vocab: 16 };
const CELL = 0.72;   // cube size; step between cells is 1
const GAP = 4;       // space between the residual spine and the side blocks

const COLOR = {
    embed: [0.35, 0.80, 0.45],
    residual: [0.15, 0.75, 0.70],
    qkv: [0.25, 0.55, 0.95],
    scores: [0.20, 0.85, 0.90],
    mlp: [0.70, 0.45, 0.95],
};

// A little smooth variation per cell, so a slab reads as data rather than a
// flat painted block. Purely visual — we do not have the real weights loaded.
const shimmer = (r, c) => 0.68 + 0.32 * (0.5 + 0.5 * Math.sin(r * 0.7) * Math.cos(c * 0.9));

// Emit an R×C grid of cells into `acc`, starting at `base`, stepping `u` per
// column and `v` per row (both length-3 world vectors).
function addSlab(acc, rows, cols, base, u, v, color) {
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            acc.offsets.push(
                base[0] + u[0] * c + v[0] * r,
                base[1] + u[1] * c + v[1] * r,
                base[2] + u[2] * c + v[2] * r);
            acc.scales.push(CELL, CELL, CELL);
            const k = shimmer(r, c);
            acc.colors.push(color[0] * k, color[1] * k, color[2] * k);
        }
    }
}

netron3d.transformerScene = (dims) => {
    const d = dims || {};
    const real = {
        dmodel: Number(d.n_embd) || 0,
        heads: Number(d.n_heads) || 0,
        layers: Number(d.n_layers) || 0,
    };
    const DM = Math.min(real.dmodel || 12, CAP.dmodel);
    const SEQ = CAP.seq;
    const DH = Math.min(real.dmodel && real.heads ? Math.round(real.dmodel / real.heads) : 8, CAP.dhead);
    const DF = Math.min(real.dmodel ? real.dmodel * 4 : 16, CAP.dff);
    const VOC = CAP.vocab;

    // How many layer bands to actually draw. Scales with real depth so a deep
    // stack looks deep, capped so a 96-layer model stays legible; the caption
    // always states the true count.
    const shown = Math.min(real.layers || 6, 12);
    const BAND = DM + 7;

    const acc = { offsets: [], scales: [], colors: [] };
    const X = [1, 0, 0];
    const Y = [0, 1, 0];
    const Z = [0, 0, 1];

    const centered = (n) => -n / 2;

    // Token embedding at the base.
    addSlab(acc, SEQ, DM, [centered(DM), -BAND, centered(SEQ)], X, Z, COLOR.embed);

    for (let b = 0; b < shown; b++) {
        const y = b * BAND;
        // Residual stream: a horizontal seq × d_model slab (the spine).
        addSlab(acc, SEQ, DM, [centered(DM), y, centered(SEQ)], X, Z, COLOR.residual);

        // Attention: Q, K, V weight matrices (d_model rows × d_head cols),
        // vertical, stacked in depth, on the −X side.
        const attnX = centered(DM) - GAP - DH;
        [-2.5, 0, 2.5].forEach((dz) => {
            addSlab(acc, DM, DH, [attnX, y + 1, dz], X, Y, COLOR.qkv);
        });
        // The attention score matrix (seq × seq), above the QKV.
        addSlab(acc, SEQ, SEQ, [attnX, y + DM + 2, 0], X, Y, COLOR.scores);

        // MLP: up-projection (d_model × d_ff) and down-projection, vertical,
        // on the +X side.
        const mlpX = DM / 2 + GAP;
        addSlab(acc, DM, DF, [mlpX, y + 1, -1.5], X, Y, COLOR.mlp);
        addSlab(acc, DF, DM, [mlpX, y + 1, DF + 1], X, Y, COLOR.mlp);
    }

    // Unembedding at the top.
    addSlab(acc, VOC, DM, [centered(DM), shown * BAND, centered(VOC)], X, Z, COLOR.embed);

    const height = (shown + 1) * BAND;
    return {
        offsets: new Float32Array(acc.offsets),
        scales: new Float32Array(acc.scales),
        colors: new Float32Array(acc.colors),
        target: [0, height / 2 - BAND / 2, 0],
        distance: Math.max(height, DM + 2 * (GAP + DF)) * 1.15,
        cells: acc.offsets.length / 3,
        downsampled:
            (real.dmodel > CAP.dmodel) ||
            (real.dmodel && Math.round(real.dmodel / (real.heads || 1)) > CAP.dhead),
        shown,
    };
};

// --- rendering ---------------------------------------------------------------

netron3d._styleId = 'netron3d-style';
netron3d._css = `
  .n3d-root{position:relative;width:100%;height:100%;background:#0d1117;overflow:hidden;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d1d9}
  .n3d-canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
  .n3d-canvas:active{cursor:grabbing}
  .n3d-caption{position:absolute;top:10px;left:12px;font-size:12px;line-height:1.5;
    z-index:2;pointer-events:none}
  .n3d-caption b{color:#e6edf3} .n3d-caption .sub{color:#8b949e}
  .n3d-legend{position:absolute;bottom:10px;left:12px;font-size:11px;color:#8b949e;
    z-index:2;pointer-events:none;line-height:1.7}
  .n3d-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
  .n3d-fallback{padding:1rem;color:#8b949e;font-size:12px}
`;

netron3d._ensureStyle = (doc) => {
    if (doc.getElementById(netron3d._styleId)) {
        return;
    }
    const style = doc.createElement('style');
    style.id = netron3d._styleId;
    style.textContent = netron3d._css;
    (doc.head || doc.documentElement).appendChild(style);
};

const swatch = (rgb, text) => {
    const hex = rgb.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
    return `<span><i style="background:#${hex}"></i>${text}</span>`;
};

netron3d.families = {};
netron3d.families.transformer = {
    name: 'transformer',
    render(element, spec) {
        const doc = element.ownerDocument || document;
        netron3d._ensureStyle(doc);

        // Tear down a previous scene on this element before drawing another.
        if (element._netron3d) {
            element._netron3d.dispose();
            element._netron3d = null;
        }
        element.innerHTML = '';

        const dims = (spec && spec.dims) || {};
        const scene = netron3d.transformerScene(dims);
        const label = (spec && spec.label) || 'GPT-style transformer';
        const exact = dims.n_layers !== undefined && dims.n_layers !== null;

        const root = doc.createElement('div');
        root.className = 'n3d-root';
        const canvas = doc.createElement('canvas');
        canvas.className = 'n3d-canvas';
        root.appendChild(canvas);
        element.appendChild(root);

        const engine = createEngine(canvas);
        if (!engine) {
            root.innerHTML =
                '<div class="n3d-fallback">This view needs WebGL2, which is not ' +
                'available here. Netron\'s 2D graph is the fallback.</div>';
            return false;
        }
        engine.setInstances(scene);
        element._netron3d = engine;

        let sub;
        if (exact) {
            sub = `${dims.n_layers} layers`;
            if (dims.n_heads) {
                sub += ` · ${dims.n_heads} heads`;
            }
            if (dims.n_embd) {
                sub += ` · d_model ${dims.n_embd}`;
            }
            if (dims.estimated) {
                sub += ' (estimated from graph)';
            }
        } else {
            sub = 'decoder-only transformer — illustrative dimensions';
        }
        const note = scene.downsampled
            ? '<br><span class="sub">dimensions downsampled to cells · drag to rotate, scroll to zoom</span>'
            : '<br><span class="sub">drag to rotate, scroll to zoom</span>';

        const caption = doc.createElement('div');
        caption.className = 'n3d-caption';
        caption.innerHTML =
            `<b>${String(label).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))}</b>` +
            `<br><span class="sub">${sub}</span>${note}`;
        root.appendChild(caption);

        const legend = doc.createElement('div');
        legend.className = 'n3d-legend';
        legend.innerHTML =
            swatch(COLOR.residual, 'residual') + ' &nbsp; ' +
            swatch(COLOR.qkv, 'Q·K·V') + ' &nbsp; ' +
            swatch(COLOR.scores, 'attention') + ' &nbsp; ' +
            swatch(COLOR.mlp, 'MLP') + ' &nbsp; ' +
            swatch(COLOR.embed, 'embed / unembed');
        root.appendChild(legend);

        return true;
    }
};

netron3d.render = (element, spec) => {
    const family = spec && netron3d.families[spec.family];
    if (!family) {
        return false;
    }
    return family.render(element, spec);
};

export const { detect, families, opTypes, render, transformerScene } = netron3d;
export default netron3d;
