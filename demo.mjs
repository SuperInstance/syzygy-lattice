#!/usr/bin/env node
// demo.mjs — render five scenes through the lattice, run CV on the text,
// seal every detection as a receipt, and verify the ledger out loud.
import { synthFrame, scenes } from "./src/synth.mjs";
import { encodeFrame, renderGlyphs, renderBraille } from "./src/codec.mjs";
import { components, orientationField, bandSegment } from "./src/cv.mjs";
import { VisionLedger } from "./src/receipts.mjs";

const ledger = new VisionLedger();
console.log("syzygy-lattice demo — orientation-keyed ASCII encoding + CV on text + receipted perception\n");

for (const name of Object.keys(scenes)) {
  const frame = synthFrame(scenes[name], 7);
  const enc = encodeFrame(frame);
  const field = orientationField(enc);
  console.log(`── scene: ${name} ── lattice hash ${enc.hash}`);
  console.log(renderGlyphs(enc));
  console.log("");
  const comps = components(enc, 4);
  for (const c of comps) console.log(`  component mass=${c.mass} bbox_cell=[${c.bbox_cell}]`);
  console.log(`  orientation: ${field.majority_class} (${(field.majority_share * 100).toFixed(0)}% of weighted votes)`);
  const seg = bandSegment(enc);
  console.log(`  bands: high=${seg.high.cells} low=${seg.low.cells}`);
  ledger.seal("FRAME", name, { hash: enc.hash, orient: field.votes });
  ledger.sealDetections(comps);
  ledger.seal("SEGMENT", name, { high: seg.high.cells, low: seg.low.cells });
  console.log("");
}

const v = ledger.verifyChain();
console.log(`vision ledger: ${v.len} receipts, verify ${v.ok ? "CLEAN" : "FAILED"} — tip ${v.tip}`);
console.log("a forged 'I saw X' claim would break this chain. that is the point.");
