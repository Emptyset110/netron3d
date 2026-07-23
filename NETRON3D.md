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

- `source/netron3d.js` — family detection from a parsed graph's op signature,
  and a self-contained transformer 3D scene (CSS 3D, drag to rotate, no external
  asset). `detect(graph)` → `{ family, dims }`; `render(element, spec)` → draws
  it, or returns `false` so the caller falls back to Netron's 2D graph.
- `test/netron3d.test.mjs` — detection logic (`node test/netron3d.test.mjs`).
- `test/netron3d.html` — a standalone browser demo of the renderer.

Roadmap:

- wire a **3D toggle into Netron's viewer** (`source/view.js`) beside its 2D graph;
- more families: U-Net, ResNet/CNN, ViT, diffusion U-Net;
- exact dimensions from the parsed model rather than estimates, and from a
  sibling config where present;
- a WebGL renderer for scenes that outgrow CSS 3D.

## Licence

MIT — Netron's, retained in `LICENSE`; additions under the same terms. See
`NOTICE` for the fork's provenance and its relationship to llm-viz.
