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

// --- e-process learner (witness-validation math merged in) ---
// witness-validation's DESIGN separates three questions; the learner's
// "streak ≥ patience" rule answered none of them rigorously. This shard
// answers the statistical one the DESIGN prescribes: an E-PROCESS / test
// martingale. Generative model (pinned, seeded): the labeller clicks with
// probability p=3/4 on frames where the true rule fires, never otherwise.
// Null: a coin flip q=1/2 (realm-ml measured the fleet's judge at AUC 0.510
// — a coin flip wearing an oracle's clothes; here the null IS the coin).
// Per-frame likelihood ratio for hypothesis h = P(obs|h) / P(obs|null):
//   h fires, click     -> (3/4)/(1/2) = 3/2
//   h fires, no click  -> (1/4)/(1/2) = 1/2
//   h silent, click    -> 0/(1/2)     = 0      (h claimed impossible => killed)
//   h silent, no click -> 1/(1/2)     = 2
// e-value = running product, kept EXACT as BigInt num/den — no float ever
// crosses a decision. Accept at integer threshold num >= 32*den: Ville's
// inequality gives P(sup E >= 1/alpha) <= alpha under the null, so false
// acceptance is bounded by alpha = 1/32 per hypothesis (3/32 union over the
// declared space). Wrong rules die: killed outright the first time the
// labeller clicks where they stay silent, or ground down by 1/2 per lap on
// frames they fire but the truth does not.

// stochastic labeller: clicks w.p. pNum/pDen where the true rule fires.
// sampling is INTEGER: rand % pDen < pNum — no floats in the decision path.
export function makeEClicker(ruleName, pNum = 3, pDen = 4, seed = 20260930) {
  const rule = HYPOTHESIS_SPACE.find((r) => r.name === ruleName);
  if (!rule) throw new Error(`unknown rule: ${ruleName}`);
  let s = seed >>> 0;
  const rand = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0, s);
  return { name: rule.name, p: `${pNum}/${pDen}`, click: (f) => (rule.test(f) ? rand() % pDen < pNum : false) };
}

// the coin-flip null labeller realm-ml measured in the fleet's judge.
export function makeNullEClicker(seed = 777) {
  let s = seed >>> 0;
  const rand = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0, s);
  return { name: "null-coin", click: () => rand() % 2 === 0 };
}

// Acceptance rule (pinned, honest): h is accepted when its e-value is both
// >= 32 against the coin null AND >= 32x every OTHER live hypothesis's
// e-value (pairwise, cross-multiplied — BigInt, no floats). The second
// condition is what pins the observed misspecification leak: under the null
// Ville bounds false acceptance at alpha=1/32, but against a NESTED rule
// (impostor whose fire set covers the truth's) a bare threshold can let the
// impostor cross later — the ratio test makes the truth win because every
// impostor-only fire frame costs it 1/2 forever.
export function eLearn(stream, clicker, { thresholdNum = 32, thresholdDen = 1, ledger = new VisionLedger() } = {}) {
  const hyps = HYPOTHESIS_SPACE.map((r) => ({ name: r.name, test: r.test, num: 1n, den: 1n, killedAt: null, acceptedAt: null }));
  const accepted = [];
  const path = [];
  const T = BigInt(thresholdNum), TD = BigInt(thresholdDen);
  const PN = 3n, PD = 4n, QN = 1n, QD = 2n;
  const beatsAll = (h) => hyps.every((o) => o === h || o.killedAt !== null ||
    h.num * BigInt(o.den) * TD >= T * BigInt(o.num) * h.den);
  for (const { frame, index } of stream) {
    const f = features(encodeFrame(frame));
    const click = clicker.click(f);
    for (const h of hyps) {
      if (h.killedAt !== null || h.acceptedAt !== null) continue;
      const fires = h.test(f);
      if (fires && click) { h.num *= PN * QD; h.den *= PD * QN; }        // (p)/(q) = 3/2
      else if (fires && !click) { h.num *= (PD - PN) * QD; h.den *= PD * (QD - QN); } // 1/2
      else if (!fires && click) { h.num = 0n; h.den = 1n; h.killedAt = index; }
      else { h.num *= QD; h.den *= QN; }                                  // 2
    }
    path.push(hyps.map((h) => `${h.num}/${h.den}`));
    for (const h of hyps) {
      if (h.killedAt === null && h.acceptedAt === null && h.num * TD >= T * h.den && beatsAll(h)) {
        h.acceptedAt = index;
        accepted.push(ledger.seal("HYPOTH-E", `rule/${h.name}@eproc`, {
          name: h.name, e_num: h.num.toString(), e_den: h.den.toString(),
          alpha_num: 1, alpha_den: 32, at_frame: index, generative: "p=3/4 vs null q=1/2",
        }));
      }
    }
  }
  return { ledger, accepted, hyps: hyps.map(({ name, num, den, killedAt, acceptedAt }) => ({ name, e: `${num}/${den}`, killedAt, acceptedAt })), path };
}
