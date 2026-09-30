# syzygy-lattice

**Orientation-keyed ASCII image encoding + computer vision that runs on the text. Receipted perception.**

Part of the SuperInstance fleet. Sibling of [Syzygy](https://github.com/SuperInstance/Syzygy)
(the fused single-pass sensor→text kernel) and inheritor of
[chiaroscuro](https://github.com/SuperInstance/quilt-canvas-tui)'s question —
*when a pixel becomes a character, what should the character know?*

Syzygy emits braille, edge-glyph, and tone as three **separate** channels.
syzygy-lattice fuses the second dimension **into** the tone character: the
local Sobel class chooses *which density ramp* a cell's brightness draws from,
so one printed glyph encodes **brightness band AND edge orientation** at once.
A human reads the shading; a machine reads the orientation; both read the same
character. The five ramps (flat / horizontal / vertical / diag+ / diag-) form
a lattice — nearest neighbors differ in exactly one coordinate — the same
discipline as a crystal lattice, hence the name and the nod to
[crystal-lattice](https://github.com/SuperInstance/crystal-lattice).

On top of the codec: **CV on the text, never the pixels** — connected
components over the braille dot grid, orientation histograms from the glyph
layer, band segmentation — and every detection is sealed as a receipt in the
`quilt/cell-receipt@v1` schema (same shape as quilt-canvas-tui's fabric).
A vision pipeline whose outputs are a hash chain: a forged *"I saw X at cell
(a,b)"* fails verification exactly like a fabricated memory claim in PoEM's
threat model ([arXiv 2608.16032](https://arxiv.org/abs/2608.16032)).

## Run it

```sh
node demo.mjs          # render five scenes as ASCII + run CV + seal the ledger
node --test test/lattice.pins.mjs   # 9 pins (T1–T6 codec/CV/ledger, T7–T9 click-ML)
```

No dependencies. Node ≥ 18. Everything is seeded and integer-leaning; the
diamond-scene lattice hashes to `e5ce1efa0777784a` on every machine.

## The three layers

| layer | bytes | what it is |
|---|---|---|
| braille | 1/cell | 2×4 threshold dots → U+2800+mask (Syzygy's 0001 discipline) |
| glyph | 1/cell | band (0..9) + orientation class (0..4) packed in one byte |
| receipts | 1/detection | sealed `DETECT` records, parent-linked, content-bound |

## Marks

This repo uses the fleet's four marks (see Syzygy's `docs/diffuse-by-marks.md`):

- **HEWN** — cut and proven by a passing pin right now.
- **SHAPED** — code exists, not yet pinned.
- **DRAWN** — chalked out, no code yet.
- **SCARF** — a ruling where sources contradicted, cited both sides.

| shard | mark | pins |
|---|---|---|
| `src/synth.mjs` — seeded scene synthesis | HEWN | T1 |
| `src/codec.mjs` — orientation-keyed lattice codec | HEWN | T2, T3 |
| `src/cv.mjs` — components / orientation / segmentation on text | HEWN | T4 (part) |
| `src/receipts.mjs` — vision ledger + decode path | HEWN | T5, T6 |
| `src/clicklearn.mjs` — click-supervised micro-ML (video feed + hidden clicker + hypothesis space) | HEWN | T7, T8, T9 |

## Known gaps (written down next to the claims)

- The ledger's chain rule is **public** — tamper-evident, not tamper-resistant.
  The fix already exists in the family: quilt-canvas-tui's `signing.mjs`
  (HMAC key outside the process). Port it here — that is the honest next shard.
- `synth.mjs` scenes are SDF sprites, not camera frames. A Syzygy NV12 ingest
  bridge (reading `syz_fused` outputs into this lattice) is DRAWN.
- The orientation ramp set (five ramps × ten bands) is hand-picked, not
  optimized for print contrast. A learned ramp is DRAWN, after Syzygy's 0007
  ("the tokenizer is not learnable" — same gap, admitted in both houses).
- No WASM port yet; the JS is the reference. Same bytes everywhere is pinned
  at the lattice-hash level, not at the compiler level.
- The click-ML learner only recovers rules in its declared hypothesis space;
  that boundary is pinned (T8's outside-space clicker is rejected), not hidden.
  Real video ingest: see the fleet's edge-ledger + scout-video-ingest notes —
  ffmpeg absent (apt-installable), PIL bridge available, synth streams HEWN.

## License

MIT. _The crab inherits the shell; the lattice inherits the light._
