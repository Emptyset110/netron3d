# netron3d

**A real neural network, drawn in 3D.** A fork of
[Netron](https://github.com/lutzroeder/netron) that adds a family-aware 3D view.

[Netron](https://github.com/lutzroeder/netron) reads any model and lays its
computation graph out in clean 2D. [llm-viz](https://bbycroft.net/llm) renders a
transformer beautifully in 3D — but only its own preset demo networks; it takes
no model as input. netron3d is the missing middle: **this** model, its **actual**
dimensions, in **3D**.

## The idea

A general computation graph in 3D is *less* legible than in 2D — 3D node-link
diagrams suffer occlusion and depth ambiguity, which is exactly why Netron chose
2D. The appeal of llm-viz is not a graph algorithm; it is a *hand-crafted layout*
that knows it is looking at a transformer.

So netron3d does not auto-3D an arbitrary graph. It:

1. **identifies the family** — transformer, and more to come — from the model's
   operator signature (or, later, its config);
2. **draws a hand-laid 3D scene** for that family, **sized from the real model's
   numbers**.

Anything with no family template keeps Netron's 2D graph, which is the right
answer for it.

## Status

Founding slice, added on top of the Netron fork:

- `source/netron3d-webgl.js` — a tiny WebGL2 instanced-cube engine (one draw
  call, orbit camera), dependency-free.
- `source/netron3d.js` — the **transformer scene**: every tensor is a *grid of
  cells* at its real (downsampled) dimensions — the residual stream, the Q·K·V
  weight matrices and attention scores, the MLP projections, embedding and
  unembedding — laid out to show the flow up through the stack. The tensors ARE
  the geometry, not panels in perspective. `detect(graph)` → `{ family, dims }`;
  `transformerScene(dims)` → instance buffers; `render(element, spec)` → draws
  it, or returns `false` so the caller falls back to Netron's 2D graph.
- a second family, **cnn** — a funnel of feature-map *volumes* that shrink
  spatially and deepen in channels, a fundamentally different 3D reading from the
  transformer's block stack. Feature-map shapes are illustrative until read from
  the parsed model, and the caption says so.
- `test/netron3d.test.mjs` — detection + scene logic (`node test/netron3d.test.mjs`).
- `test/netron3d.html` — a standalone browser demo of the renderer.

## Design: tensors as 3D cells

The point is not to tilt a 2D diagram into 3D — it is to draw the *kernels*. Each
weight matrix and activation is a grid of cells sized by the model's real
dimensions (downsampled to a legible cell budget; the caption states the true
numbers). For a transformer:

- **residual stream** — the central `seq × d_model` spine the data flows up;
- **attention** — the `d_model × d_head` Q, K and V weight matrices stacked by
  head, and the `seq × seq` score grid;
- **MLP** — the `d_model × d_ff` up- and down-projection matrices;
- **embedding / unembedding** — top and bottom.

Depth scales with the layer count, so a 34-layer model reads as deeper than a
6-layer one. Rendered in WebGL2 with instanced cubes and an orbit camera.

Roadmap:

- wire a **3D toggle into Netron's viewer** (`source/view.js`) beside its 2D graph;
- more families: U-Net, ViT, diffusion U-Net (transformer and CNN are in);
- exact dimensions from the parsed model rather than estimates, and from a
  sibling config where present;
- load real weights so cells carry actual values, not a structural shimmer;
- richer per-tensor detail (labels on the wires, activation volumes at a chosen
  sequence length).

## Licence

MIT — Netron's, retained in `LICENSE`; additions under the same terms. See
`NOTICE` for the fork's provenance and its relationship to llm-viz.
