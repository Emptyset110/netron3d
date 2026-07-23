
// netron3d — a real neural network, drawn in 3D.
//
// Netron lays any model out as a clean 2D graph. This adds a second, opt-in
// rendering that treats the model's *tensors as the 3D objects*: each weight
// matrix and activation is a grid of cells at its real (downsampled) dimensions,
// connected by dataflow lines with a travelling pulse, and labelled with its
// true size — the way llm-viz draws a transformer, but sized from this model's
// own numbers and rendered from our own WebGL.
//
// Not a general graph-to-3D layout (which reads worse than 2D). It detects a
// *family* and draws a purpose-built scene; anything unrecognised returns false
// so the caller keeps Netron's 2D graph.

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
    if (isTransformer) {
        const layers = Math.max(attention, Math.round(norm / 2)) || 0;
        const dims = layers > 0 ? { n_layers: layers, estimated: true } : {};
        return { family: 'transformer', dims, label: label || '' };
    }

    // A convolutional net: Conv ops and no attention. The stage count is
    // estimated from how many convs there are.
    const conv = count((o) => o.includes('conv'));
    if (conv > 0) {
        const stages = Math.max(2, Math.min(conv, 7));
        return { family: 'cnn', dims: { n_stages: stages, estimated: true }, label: label || '' };
    }
    return { family: 'unknown', dims: {}, label: label || '' };
};

// --- the transformer scene ---------------------------------------------------

const CAP = { dmodel: 16, seq: 8, dhead: 8, dff: 16, vocab: 16 };
const CELL = 0.72;
const GAP = 4;

const COLOR = {
    embed: [0.35, 0.80, 0.45],
    residual: [0.15, 0.75, 0.70],
    qkv: [0.25, 0.55, 0.95],
    scores: [0.20, 0.85, 0.90],
    mlp: [0.70, 0.45, 0.95],
    flow: [0.55, 0.90, 0.85],
};

const shimmer = (r, c) => 0.68 + 0.32 * (0.5 + 0.5 * Math.sin(r * 0.7) * Math.cos(c * 0.9));

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
        vocab: Number(d.n_vocab) || 0,
    };
    const DM = Math.min(real.dmodel || 12, CAP.dmodel);
    const SEQ = CAP.seq;
    const DH = Math.min(real.dmodel && real.heads ? Math.round(real.dmodel / real.heads) : 8, CAP.dhead);
    const DF = Math.min(real.dmodel ? real.dmodel * 4 : 16, CAP.dff);
    const VOC = CAP.vocab;
    const shown = Math.min(real.layers || 6, 12);
    const BAND = DM + 7;

    const acc = { offsets: [], scales: [], colors: [] };
    const edges = { positions: [], colors: [], along: [] };
    const labels = [];
    const X = [1, 0, 0];
    const Y = [0, 1, 0];
    const Z = [0, 0, 1];
    const centered = (n) => -n / 2;

    // A dataflow edge, oriented source → destination (the pulse runs 0 → 1).
    const addEdge = (p0, p1) => {
        edges.positions.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
        edges.colors.push(...COLOR.flow, ...COLOR.flow);
        edges.along.push(0, 1);
    };

    const attnX = centered(DM) - GAP - DH;
    const mlpX = DM / 2 + GAP;
    const spine = (y) => [0, y, 0];
    const attnCenter = (y) => [attnX + DH / 2, y + DM / 2, 0];
    const mlpCenter = (y) => [mlpX + DF / 2, y + DM / 2, 1.5];

    // Token embedding at the base.
    addSlab(acc, SEQ, DM, [centered(DM), -BAND, centered(SEQ)], X, Z, COLOR.embed);
    const embedC = [0, -BAND, 0];

    for (let b = 0; b < shown; b++) {
        const y = b * BAND;
        // Residual stream: the horizontal seq × d_model spine.
        addSlab(acc, SEQ, DM, [centered(DM), y, centered(SEQ)], X, Z, COLOR.residual);
        // Attention: Q, K, V weight matrices stacked by head; the score grid above.
        [-2.5, 0, 2.5].forEach((dz) => addSlab(acc, DM, DH, [attnX, y + 1, dz], X, Y, COLOR.qkv));
        addSlab(acc, SEQ, SEQ, [attnX, y + DM + 2, 0], X, Y, COLOR.scores);
        // MLP: up- and down-projection matrices.
        addSlab(acc, DM, DF, [mlpX, y + 1, -1.5], X, Y, COLOR.mlp);
        addSlab(acc, DF, DM, [mlpX, y + 1, DF + 1], X, Y, COLOR.mlp);

        // Dataflow: up the residual spine, and the read/write of each sublayer.
        if (b === 0) {
            addEdge(embedC, spine(0));
        } else {
            addEdge(spine((b - 1) * BAND), spine(y));
        }
        addEdge(spine(y), attnCenter(y));      // residual → attention (read)
        addEdge(attnCenter(y), spine(y + 2));  // attention → residual (write)
        addEdge(spine(y), mlpCenter(y));       // residual → MLP
        addEdge(mlpCenter(y), spine(y + 2));   // MLP → residual
    }

    const topY = shown * BAND;
    addSlab(acc, VOC, DM, [centered(DM), topY, centered(VOC)], X, Z, COLOR.embed);
    addEdge(spine((shown - 1) * BAND), spine(topY));
    const unembedC = [0, topY, 0];

    // Size labels — the *real* dimensions, not the downsampled cell counts, so
    // the true tensor sizes are legible. Only emitted when the dims are known.
    const exact = real.dmodel > 0;
    if (exact) {
        const dhead = real.heads ? Math.round(real.dmodel / real.heads) : null;
        const dff = real.dmodel * 4;
        const voc = real.vocab ? real.vocab.toLocaleString() : 'vocab';
        labels.push({ pos: spine(0), text: `residual · seq × ${real.dmodel}` });
        labels.push({ pos: attnCenter(0), text: `Wq·k·v · ${real.dmodel}${dhead ? ` × ${dhead}` : ''}` });
        labels.push({ pos: mlpCenter(0), text: `W_mlp · ${real.dmodel} × ${dff}` });
        labels.push({ pos: embedC, text: `embedding · ${voc} × ${real.dmodel}` });
        labels.push({ pos: unembedC, text: `unembedding · ${real.dmodel} × ${voc}` });
    } else {
        labels.push({ pos: spine(0), text: 'residual stream' });
        labels.push({ pos: attnCenter(0), text: 'Q · K · V' });
        labels.push({ pos: mlpCenter(0), text: 'MLP' });
    }

    const height = (shown + 1) * BAND;
    return {
        offsets: new Float32Array(acc.offsets),
        scales: new Float32Array(acc.scales),
        colors: new Float32Array(acc.colors),
        edges: {
            positions: new Float32Array(edges.positions),
            colors: new Float32Array(edges.colors),
            along: new Float32Array(edges.along),
        },
        labels,
        target: [0, height / 2 - BAND / 2, 0],
        distance: Math.max(height, DM + 2 * (GAP + DF)) * 1.2,
        cells: acc.offsets.length / 3,
        downsampled:
            (real.dmodel > CAP.dmodel) ||
            (real.dmodel && Math.round(real.dmodel / (real.heads || 1)) > CAP.dhead),
        shown,
    };
};

// --- the convolutional scene -------------------------------------------------
//
// A different 3D interpretation entirely: not a stack of matrices but a funnel
// of feature-map *volumes* that shrink spatially and deepen in channels as the
// data flows through — the shape that makes a CNN a CNN. Each stage is a stack
// of channel slices (H × W grids). Feature-map shapes are illustrative until
// read from the parsed model, and the caption says so.

const CNN_STAGE_GAP = 14;

const cnnColor = (t) => [0.28 + t * 0.40, 0.58 - t * 0.10, 0.95 - t * 0.28];

netron3d.cnnScene = (dims) => {
    const d = dims || {};
    const stages = Math.max(2, Math.min(Number(d.n_stages) || 5, 7));

    const acc = { offsets: [], scales: [], colors: [] };
    const edges = { positions: [], colors: [], along: [] };
    const labels = [];
    const addEdge = (p0, p1) => {
        edges.positions.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
        edges.colors.push(...COLOR.flow, ...COLOR.flow);
        edges.along.push(0, 1);
    };

    const centers = [];
    let maxH = 0;
    for (let s = 0; s < stages; s++) {
        const t = stages > 1 ? s / (stages - 1) : 0;
        const H = Math.max(3, Math.round(9 - t * 5));   // spatial shrinks
        const W = H;
        const C = Math.min(2 + Math.round(t * 6), 8);   // channel slices grow
        const cx = s * CNN_STAGE_GAP;
        for (let ch = 0; ch < C; ch++) {
            const base = [cx - W / 2, -H / 2, (ch - (C - 1) / 2) * 1.7];
            addSlab(acc, H, W, base, [1, 0, 0], [0, 1, 0], cnnColor(t));
        }
        const center = [cx, 0, 0];
        centers.push(center);
        if (s > 0) {
            addEdge(centers[s - 1], center);
        }
        const text = s === 0 ? 'input' : (s === stages - 1 ? 'head' : `conv stage ${s}`);
        labels.push({ pos: [cx, H / 2 + 2.5, 0], text });
        maxH = Math.max(maxH, H);
    }

    const width = (stages - 1) * CNN_STAGE_GAP;
    return {
        offsets: new Float32Array(acc.offsets),
        scales: new Float32Array(acc.scales),
        colors: new Float32Array(acc.colors),
        edges: {
            positions: new Float32Array(edges.positions),
            colors: new Float32Array(edges.colors),
            along: new Float32Array(edges.along),
        },
        labels,
        target: [width / 2, 0, 0],
        distance: Math.max(width, maxH * 2) * 1.6,
        cells: acc.offsets.length / 3,
        stages,
        illustrative: true,
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
    z-index:3;pointer-events:none}
  .n3d-caption b{color:#e6edf3} .n3d-caption .sub{color:#8b949e}
  .n3d-legend{position:absolute;bottom:10px;left:12px;font-size:11px;color:#8b949e;
    z-index:3;pointer-events:none;line-height:1.7}
  .n3d-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
  .n3d-labels{position:absolute;inset:0;z-index:2;pointer-events:none;overflow:hidden}
  .n3d-label{position:absolute;left:0;top:0;font-size:10px;color:#e6edf3;white-space:nowrap;
    background:rgba(13,17,23,.72);border:1px solid #30363d;border-radius:4px;padding:1px 5px;
    transform:translate(-9999px,-9999px);will-change:transform}
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

const escapeHtml = (s) =>
    String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// Common plumbing for any family: canvas, engine, screen-pinned labels, caption
// and legend. `meta` is { label, subtitleHtml, legendHtml }.
netron3d._mount = (element, scene, meta) => {
    const doc = element.ownerDocument || document;
    netron3d._ensureStyle(doc);

    if (element._netron3d) {
        element._netron3d.dispose();
        element._netron3d = null;
    }
    element.innerHTML = '';

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
    engine.setLines(scene.edges);
    element._netron3d = engine;

    // Labels pinned to their tensors each frame by projecting the 3D anchor.
    const labelLayer = doc.createElement('div');
    labelLayer.className = 'n3d-labels';
    root.appendChild(labelLayer);
    const labelEls = scene.labels.map((lb) => {
        const el = doc.createElement('div');
        el.className = 'n3d-label';
        el.textContent = lb.text;
        labelLayer.appendChild(el);
        return el;
    });
    engine.onFrame((project) => {
        for (let i = 0; i < scene.labels.length; i++) {
            const p = project(scene.labels[i].pos);
            const el = labelEls[i];
            if (!p.visible) {
                el.style.display = 'none';
                continue;
            }
            el.style.display = 'block';
            el.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
        }
    });

    const caption = doc.createElement('div');
    caption.className = 'n3d-caption';
    caption.innerHTML = `<b>${escapeHtml(meta.label)}</b><br>${meta.subtitleHtml}`;
    root.appendChild(caption);

    const legend = doc.createElement('div');
    legend.className = 'n3d-legend';
    legend.innerHTML = meta.legendHtml;
    root.appendChild(legend);

    return true;
};

netron3d.families = {};

netron3d.families.transformer = {
    name: 'transformer',
    render(element, spec) {
        const dims = (spec && spec.dims) || {};
        const scene = netron3d.transformerScene(dims);
        const exact = dims.n_layers !== undefined && dims.n_layers !== null;

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
            ? 'cells downsampled · labels show true sizes · drag to rotate, scroll to zoom'
            : 'drag to rotate, scroll to zoom';

        return netron3d._mount(element, scene, {
            label: (spec && spec.label) || 'GPT-style transformer',
            subtitleHtml: `<span class="sub">${sub}</span><br><span class="sub">${note}</span>`,
            legendHtml:
                swatch(COLOR.residual, 'residual') + ' &nbsp; ' +
                swatch(COLOR.qkv, 'Q·K·V') + ' &nbsp; ' +
                swatch(COLOR.scores, 'attention') + ' &nbsp; ' +
                swatch(COLOR.mlp, 'MLP') + ' &nbsp; ' +
                swatch(COLOR.embed, 'embed / unembed') + ' &nbsp; ' +
                swatch(COLOR.flow, 'dataflow'),
        });
    }
};

netron3d.families.cnn = {
    name: 'cnn',
    render(element, spec) {
        const dims = (spec && spec.dims) || {};
        const scene = netron3d.cnnScene(dims);
        const sub = `${scene.stages} stages` +
            (dims.estimated ? ' (estimated from graph)' : '');
        return netron3d._mount(element, scene, {
            label: (spec && spec.label) || 'convolutional network',
            subtitleHtml:
                `<span class="sub">${sub}</span><br>` +
                '<span class="sub">feature-map shapes illustrative until read from the ' +
                'model · drag to rotate, scroll to zoom</span>',
            legendHtml:
                swatch(cnnColor(0), 'shallow / large') + ' &nbsp; ' +
                swatch(cnnColor(1), 'deep / small') + ' &nbsp; ' +
                swatch(COLOR.flow, 'dataflow'),
        });
    }
};

netron3d.render = (element, spec) => {
    const family = spec && netron3d.families[spec.family];
    if (!family) {
        return false;
    }
    return family.render(element, spec);
};

export const { cnnScene, detect, families, opTypes, render, transformerScene } = netron3d;
export default netron3d;
