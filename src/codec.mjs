// src/codec.mjs — the lattice codec: pixel frame → three aligned text layers.
//
// NOVELTY (beyond Syzygy's kernel): Syzygy emits braille, edge-glyph, and tone
// as SEPARATE channels. Here the tone character is selected from an
// ORIENTATION-KEYED ramp: the local Sobel class (flat / h / v / diag+ / diag-)
// chooses WHICH density ramp a cell's brightness draws from, so one printed
// glyph encodes TWO dimensions — brightness band AND edge orientation. A human
// reads shading; a machine reads orientation; both read the same character.
//
// The set of five ramps, taken together, is the "lattice": a 5×10 structure
// where neighbors differ in one dimension at a time (structure ↔ tone ↔
// orientation), the same discipline as a crystal lattice: nearest-neighbor
// steps change exactly one coordinate.
//
// All selection is argmax/comparison over integers — no floats cross a cell
// boundary, no probabilities anywhere (Syzygy's 0007 discipline).
import { fnv1a64 } from "./synth.mjs";

export const CELL_W = 2, CELL_H = 4; // 8 pixels = one Braille byte (U+2800+mask)
// dot bit layout matches Unicode braille: left column 1,2,3,7; right 4,5,6,8
const DOT_BITS = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80];

// five ramps × 10 bands. Chosen so ramps differ pairwise in ≥3 positions and
// every ramp spans low→high ink. FLAT uses the classic density ladder.
export const RAMPS = {
  flat:  " .:-=+*#%@", // shading, no edge
  h:     " ▁▂▃▄▅▆▇█▰", // horizontal-edge scenes get horizontal GROWTH glyphs (10 bands)
  v:     " ▏▎▍▌▋▊▉█▮", // vertical scenes get vertical growth glyphs
  dp:    " ·:;/Xx%#@", // diag / (rising) — ten bands, same as every ramp
  dn:    " ·:\\Xx%#&*", // diag \ (falling)
};
const BANDS = 10;

export function sobelClass(y, w, h, px, py) {
  // 3×3 Sobel magnitude & dominant direction, integer.
  const at = (x, yy) => (x < 0 || yy < 0 || x >= w || yy >= h) ? 0 : y[yy * w + x];
  const gx = (-at(px-1,py-1) -2*at(px-1,py) -at(px-1,py+1)
              +at(px+1,py-1) +2*at(px+1,py) +at(px+1,py+1)) >> 3;
  const gy = (-at(px-1,py-1) -2*at(px,py-1) -at(px+1,py-1)
              +at(px-1,py+1) +2*at(px,py+1) +at(px+1,py+1)) >> 3;
  const ax = Math.abs(gx), ay = Math.abs(gy);
  const mag2 = ax * ax + ay * ay;
  if (mag2 < 64) return { cls: "flat", mag2 };
  const diag = Math.abs(ax - ay) * 2 < (ax + ay); // within ~27° of 45°
  // class = LINE orientation, not gradient direction: a horizontal line has
  // a vertical gradient (ay > ax). The ramp then draws glyphs that GROW the
  // way the line runs.
  if (!diag) return { cls: ax > ay ? "v" : "h", mag2 };
  // sign of gx*gy picks which diagonal the gradient points along
  return { cls: (gx > 0) === (gy > 0) ? "dn" : "dp", mag2 };
}

export function encodeFrame({ y, w, h }) {
  const cols = Math.floor(w / CELL_W), rows = Math.floor(h / CELL_H);
  const braille = new Uint8Array(cols * rows); // dot masks
  const glyphs = new Uint8Array(cols * rows);  // ramp band indices (0..9)
  const classes = [];                          // per-cell orientation class
  const bytes = new Uint8Array(cols * rows * 2);
  const THRESH = 100; // same default as Syzygy's fused pass
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let mask = 0;
      const cellLuma = [];
      for (let dy = 0; dy < CELL_H; dy++) {
        for (let dx = 0; dx < CELL_W; dx++) {
          const px = c * CELL_W + dx, py = r * CELL_H + dy;
          const v = y[py * w + px];
          cellLuma.push(v);
          if (v > THRESH) mask |= DOT_BITS[dy * CELL_W + dx];
        }
      }
      // cell class from the center pixel; band from mean luma
      const { cls } = sobelClass(y, w, h, c * CELL_W + 1, r * CELL_H + 2);
      const mean = (cellLuma[0] + cellLuma[1] + cellLuma[2] + cellLuma[3]
                  + cellLuma[4] + cellLuma[5] + cellLuma[6] + cellLuma[7]) >> 3;
      const band = Math.min(BANDS - 1, (mean * BANDS) >> 8);
      const i = r * cols + c;
      braille[i] = mask; glyphs[i] = band; classes.push(cls);
      bytes[i * 2] = mask; bytes[i * 2 + 1] = band | (classCode(cls) << 4);
    }
  }
  return { cols, rows, braille, glyphs, classes, bytes, hash: fnv1a64(bytes) };
}

export function classCode(cls) { return { flat: 0, h: 1, v: 2, dp: 3, dn: 4 }[cls]; }
export function codeClass(code) { return ["flat", "h", "v", "dp", "dn"][code]; }

// render the glyph layer as text (the human-readable lattice)
export function renderGlyphs(enc) {
  const out = [];
  for (let r = 0; r < enc.rows; r++) {
    let line = "";
    for (let c = 0; c < enc.cols; c++) {
      const i = r * enc.cols + c;
      line += RAMPS[enc.classes[i]][enc.glyphs[i]];
    }
    out.push(line);
  }
  return out.join("\n");
}

export function renderBraille(enc) {
  const out = [];
  for (let r = 0; r < enc.rows; r++) {
    let line = "";
    for (let c = 0; c < enc.cols; c++) line += String.fromCodePoint(0x2800 + enc.braille[r * enc.cols + c]);
    out.push(line);
  }
  return out.join("\n");
}
