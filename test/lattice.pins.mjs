// test/lattice.pins.mjs — FAIL-first pins for syzygy-lattice.
// T1 synthesis is deterministic (same seed, same FNV)
// T2 codec golden hash on the diamond scene
// T3 orientation-keyed ramps: an all-horizontal scene encodes majority h glyphs
// T4 CV-on-text: diamond scene yields exactly one large component, bbox near center
// T5 vision ledger: clean chain verifies; spliced/edited detection caught
// T6 decode path: braille-only reconstruction of the hbar scene keeps MAE sane
import { test } from "node:test";
import assert from "node:assert/strict";
import { synthFrame, scenes, fnv1a64 } from "../src/synth.mjs";
import { encodeFrame, renderGlyphs, RAMPS } from "../src/codec.mjs";
import { components, orientationField, bandSegment } from "../src/cv.mjs";
import { VisionLedger, decodeToLuma } from "../src/receipts.mjs";

test("T1 synthesis is deterministic", () => {
  const a = synthFrame(scenes.diamond, 7), b = synthFrame(scenes.diamond, 7);
  assert.deepEqual([...a.y], [...b.y]);
  assert.equal(a.jitter, b.jitter);
  assert.equal(fnv1a64(a.y), fnv1a64(b.y));
});

test("T2 codec golden hash on diamond scene", () => {
  const enc = encodeFrame(synthFrame(scenes.diamond, 7));
  assert.equal(enc.cols, 16);
  assert.equal(enc.rows, 4);
  assert.equal(enc.hash, "e5ce1efa0777784a", "golden lattice hash of the diamond scene, seed 7");
});

test("T3 orientation-keyed ramps: the lattice discriminates line direction", () => {
  const eh = encodeFrame(synthFrame(scenes.hbar, 7));
  const ev = encodeFrame(synthFrame(scenes.vbar, 7));
  const fh = orientationField(eh), fv = orientationField(ev);
  // flat interiors outnumber edge cells by construction, so the honest
  // property is DISCRIMINATION: each scene's own line class dominates the
  // other scene's by an order of magnitude.
  assert.ok(fh.votes.h > 8 * Math.max(1, fv.votes.h), `hbar h-votes ${fh.votes.h} should overwhelm vbar's ${fv.votes.h}`);
  assert.ok(fv.votes.v > 8 * Math.max(1, fh.votes.v), `vbar v-votes ${fv.votes.v} should overwhelm hbar's ${fh.votes.v}`);
  // and the glyphs drawn come from the orientation ramp, not the flat ramp
  const line = renderGlyphs(eh).split("\n")[1] ?? "";
  assert.ok([...line].some((ch) => RAMPS.h.includes(ch) && !RAMPS.flat.includes(ch)));
});

test("T4 components on diamond: one large component, exact bbox", () => {
  const enc = encodeFrame(synthFrame(scenes.diamond, 7));
  const comps = components(enc, 4);
  assert.equal(comps.length, 1, `expected the single diamond, got ${comps.length}`);
  // census-verified: dot-space bbox is [19,0,31,15] (seed 7, jitter −7 →
  // clipped at the right frame edge). Cell space is exact — a join that
  // forgets to grow the box leftward/upward fails here, which is how the
  // first version of components() was caught.
  assert.deepEqual(comps[0].bbox_dot, [19, 0, 31, 15]);
  assert.deepEqual(comps[0].bbox_cell, [9, 0, 15, 3]);
  assert.equal(comps[0].mass, 128);
});

test("T5 vision ledger: clean verifies; splice and edit caught", () => {
  const enc = encodeFrame(synthFrame(scenes.diamond, 7));
  const led = new VisionLedger();
  led.sealDetections(components(enc, 4));
  led.seal("SEGMENT", "bands", bandSegment(enc));
  led.seal("ORIENT", "field", orientationField(enc).votes);
  assert.equal(led.verifyChain().ok, true);
  // splice a phantom detection (PoEM: fabricated memory of seeing something)
  const phantom = led.seal("DETECT", "cells(0,0,1,1)", { kind: "component", mass: 99, bbox_cell: [0, 0, 1, 1] });
  led.receipts.splice(led.receipts.length - 2, 1); // remove the real one before it
  assert.equal(led.verifyChain().ok, false, "spliced phantom breaks linkage");
  // edit-in-place (FARMA: 'the result was already X')
  const led2 = new VisionLedger();
  led2.sealDetections(components(enc, 4));
  led2.receipts[0].result.mass = 999;
  assert.equal(led2.verifyChain().ok, false, "edited detection content breaks its receipt_id");
});

test("T6 decode: braille-only reconstruction of hbar keeps structure", () => {
  const enc = encodeFrame(synthFrame(scenes.hbar, 7));
  const dec = decodeToLuma(enc);
  // row 8 is inside the bar: decoded luma there must exceed row 0 (outside)
  const inside = dec.y[8 * dec.w + 16], outside = dec.y[0 * dec.w + 16];
  assert.ok(inside > outside, `inside(${inside}) should exceed outside(${outside})`);
});
