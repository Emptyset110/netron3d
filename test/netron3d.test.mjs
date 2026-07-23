// Family detection tests for netron3d.
//
//   node test/netron3d.test.mjs
//
// Pure logic only — the renderer needs a DOM and is exercised by the browser
// demo (test/netron3d.html), not here.

import assert from 'node:assert';
import { detect, opTypes, render, families, transformerScene, cnnScene } from '../source/netron3d.js';

let passed = 0;
const test = (name, fn) => {
    fn();
    passed += 1;
    process.stdout.write(`ok   ${name}\n`);
};

test('reads op types from a Netron-style graph', () => {
    const graph = { nodes: [{ type: { name: 'MatMul' } }, { type: 'Softmax' }] };
    assert.deepStrictEqual(opTypes(graph), ['matmul', 'softmax']);
});

test('detects a transformer from an explicit attention op', () => {
    const result = detect(['MatMul', 'Attention', 'LayerNormalization', 'Gelu']);
    assert.strictEqual(result.family, 'transformer');
});

test('detects a transformer from the softmax+matmul+norm pattern', () => {
    const ops = [];
    for (let i = 0; i < 6; i++) {
        ops.push('MatMul', 'Softmax', 'MatMul', 'LayerNormalization', 'LayerNormalization');
    }
    const result = detect(ops, 'demo');
    assert.strictEqual(result.family, 'transformer');
    // Six norm-pairs → an estimated six layers, marked as estimated.
    assert.strictEqual(result.dims.n_layers, 6);
    assert.strictEqual(result.dims.estimated, true);
    assert.strictEqual(result.label, 'demo');
});

test('detects a plain CNN as the cnn family, not a transformer', () => {
    const ops = ['Conv', 'Relu', 'MaxPool', 'Conv', 'Relu', 'Gemm'];
    const result = detect(ops);
    assert.strictEqual(result.family, 'cnn');
    assert.strictEqual(result.dims.n_stages, 2);
});

test('leaves an architecture with neither attention nor conv unknown', () => {
    assert.strictEqual(detect(['Relu', 'Add', 'Gemm']).family, 'unknown');
});

test('render() returns false for an unknown family (caller falls back to 2D)', () => {
    // No DOM here; an unknown family must bail out before touching `element`.
    assert.strictEqual(render(null, { family: 'unknown' }), false);
    assert.strictEqual(render(null, null), false);
});

test('the transformer family is registered', () => {
    assert.ok(families.transformer);
    assert.strictEqual(families.transformer.name, 'transformer');
});

test('builds a tensor scene sized from real dimensions', () => {
    const scene = transformerScene({ n_layers: 34, n_heads: 16, n_embd: 3072 });
    // Cells are real geometry: three matching buffers of 3 floats each.
    assert.ok(scene.cells > 1000);
    assert.strictEqual(scene.offsets.length, scene.cells * 3);
    assert.strictEqual(scene.offsets.length, scene.scales.length);
    assert.strictEqual(scene.offsets.length, scene.colors.length);
    // A 3072-wide model is downsampled to cells; that is stated, not hidden.
    assert.strictEqual(scene.downsampled, true);
});

test('carries dataflow edges and real-size labels', () => {
    const scene = transformerScene({ n_layers: 6, n_heads: 6, n_embd: 384, n_vocab: 50257 });
    // Edges are pairs of vertices; the `along` param has one value per vertex.
    assert.ok(scene.edges.positions.length > 0);
    assert.strictEqual(scene.edges.along.length, scene.edges.positions.length / 3);
    // Labels state the *true* dimensions, not the downsampled cell counts.
    assert.ok(scene.labels.length >= 4);
    assert.ok(scene.labels.some((l) => l.text.includes('384')));
    assert.ok(scene.labels.some((l) => l.text.includes('W_mlp')));
});

test('the cnn family builds a funnel of feature-map volumes', () => {
    const scene = cnnScene({ n_stages: 5 });
    assert.ok(scene.cells > 100);
    assert.strictEqual(scene.stages, 5);
    assert.ok(scene.labels.some((l) => l.text === 'input'));
    assert.ok(scene.labels.some((l) => l.text === 'head'));
    assert.strictEqual(scene.illustrative, true);
});

test('depth scales with the layer count', () => {
    const shallow = transformerScene({ n_layers: 4, n_heads: 8, n_embd: 512 });
    const deep = transformerScene({ n_layers: 34, n_heads: 16, n_embd: 3072 });
    assert.ok(deep.shown > shallow.shown, 'a deeper model draws more bands');
});

process.stdout.write(`\n${passed} passed\n`);
