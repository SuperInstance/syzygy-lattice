// src/clicklearn.mjs — click-supervised micro-ML on the lattice.
// The captain's directive: "use video feeds and clicks to ml little ideas to
// see if they are fruitful." This is the deterministic miniature:
//   video feed   = frameStream() — seeded synth frames, scene identity HIDDEN
//   clicks       = makeClicker(rule) — a hidden labeller over lattice features
//   little ideas = HYPOTHESIS_SPACE — named integer rules, declared up front
//   fruitful     = a hypothesis that survives 8 consecutive agreeing frames
//                  gets sealed as a HYPOTH receipt in the vision ledger.
// Honest limit (stated, not hidden): the learner can only recover rules that
// are IN THE DECLARED SPACE. A clicker outside the space yields no accepted
// hypothesis — that is the boundary of "little ideas", pinned in T8 by
// recovering each in-space rule and never claiming more.
import { synthFrame, scenes, fnv1a64 } from "./synth.mjs";
import { encodeFrame } from "./codec.mjs";
import { components, orientationField } from "./cv.mjs";
import { VisionLedger } from "./receipts.mjs";

export const SCENE_NAMES = Object.keys(scenes);

// integer feature vector from an encoded frame — all counts, no floats.
export function features(enc) {
  const comps = components(enc, 4);
  const field = orientationField(enc);
  const bands = new Array(10).fill(0);
  for (const g of enc.glyphs) bands[g]++;
  let ink = 0;
  for (const b of enc.braille) { let v = b; while (v) { ink += v & 1; v >>= 1; } } // popcount
  const c0 = comps.length ? comps[0].bbox_cell : [0, 0, 0, 0];
  return {
    cells: enc.cols * enc.rows,
    n_components: comps.length,
    max_mass: comps.length ? comps[0].mass : 0,
    total_ink: ink,
    bands,
    orient: { flat: field.votes.flat, h: field.votes.h, v: field.votes.v, dp: field.votes.dp, dn: field.votes.dn },
    col_span: comps.length ? c0[2] - c0[0] + 1 : 0,
    row_span: comps.length ? c0[3] - c0[1] + 1 : 0,
  };
}

export function featureHash(f) {
  return fnv1a64(new TextEncoder().encode(JSON.stringify(f)));
}

// the declared space of little ideas. each is an integer test over features.
export const HYPOTHESIS_SPACE = [
  { name: "diag-or-heavy", test: (f) => f.orient.dp + f.orient.dn >= (f.cells >> 2) || f.max_mass >= 96 },
  { name: "wide-structure", test: (f) => f.col_span >= 12 },
  { name: "rich-field", test: (f) => f.total_ink >= 200 && f.bands[8] + f.bands[9] >= 8 },
];

// a clicker = one hypothesis pulled out of the space, hidden from the learner
// behind a function boundary (the learner sees only frames + click bits).
export function makeClicker(ruleName) {
  const rule = HYPOTHESIS_SPACE.find((r) => r.name === ruleName);
  if (!rule) throw new Error(`unknown rule: ${ruleName} — outside the declared space`);
  return { name: rule.name, click: (f) => rule.test(f) };
}

// the video feed: deterministic frames round-robin over scenes, scene identity
// never exposed. seed varies per lap so jitter differs frame to frame.
export function frameStream(seedBase, k) {
  const out = [];
  for (let i = 0; i < k; i++) {
    const name = SCENE_NAMES[i % SCENE_NAMES.length];
    const frame = synthFrame(scenes[name], seedBase + ((i / SCENE_NAMES.length) | 0));
    out.push({ frame, index: i }); // note: no scene name carried
  }
  return out;
}

// online learner: every hypothesis in the space accumulates evidence
// (+1 agreement / clip at 0), tracks its consecutive-agreement streak, and is
// ACCEPTED (sealed as a HYPOTH receipt) the moment it survives `patience`
// consecutive agreeing frames. predictions use the best accepted hypothesis;
// before any acceptance, predict abstain (null).
export function learn(stream, clicker, { patience = 8, ledger = new VisionLedger() } = {}) {
  const hyps = HYPOTHESIS_SPACE.map((r) => ({ name: r.name, test: r.test, tally: 0, streak: 0, accepted: false }));
  let accepted = [];
  const predictions = [];
  for (const { frame, index } of stream) {
    const f = features(encodeFrame(frame));
    const click = clicker.click(f);
    for (const h of hyps) {
      if (h.accepted) continue;
      const agree = h.test(f) === click;
      h.tally = agree ? h.tally + 1 : Math.max(0, h.tally - 1);
      h.streak = agree ? h.streak + 1 : 0;
      if (h.streak >= patience) {
        h.accepted = true;
        accepted.push(ledger.seal("HYPOTH", `rule/${h.name}`, { name: h.name, tally: h.tally, at_frame: index }));
      }
    }
    const best = accepted.length ? hyps.find((h) => h.accepted && h.name === accepted[accepted.length - 1].result.name) : null;
    predictions.push(best ? best.test(f) : null);
  }
  return { ledger, predictions, accepted, hyps: hyps.map(({ name, tally, streak, accepted: a }) => ({ name, tally, streak, accepted: a })) };
}

// agreement over the tail of the stream (the held-out flavor).
export function tailAgreement(predictions, stream, clicker, tail = 60) {
  let agree = 0, counted = 0;
  const start = Math.max(0, predictions.length - tail);
  for (let i = start; i < predictions.length; i++) {
    if (predictions[i] === null) continue; // abstentions don't count
    counted++;
    const f = features(encodeFrame(stream[i].frame));
    if (predictions[i] === clicker.click(f)) agree++;
  }
  return counted ? agree / counted : 0;
}
