// src/synth.mjs — deterministic test-frame synthesis.
// Seeded, integer-only. No camera, no files: the lab must run anywhere a
// stranger can run node, and produce the same bytes (Syzygy's promise,
// kept in the layer above the kernel).
export function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const W = 32, H = 16; // frame size in pixels (luma plane)
export const LUMA_MAX = 255;

// Scenes are signed distance fields sampled at pixel centers; intensity is a
// linear ramp of -distance. Integer output.
export function synthFrame(scene, seed = 1) {
  const rand = mulberry32(seed);
  const jitter = (rand() * 16 - 8) | 0; // per-frame placement jitter, deterministic
  const y = new Uint8Array(W * H);
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const d = scene(px + jitter, py, W, H);
      let v = 128 + d * 10; // inside the shape (d>0) is BRIGHT, the field dark
      v = v < 0 ? 0 : v > 255 ? 255 : v | 0;
      y[py * W + px] = v;
    }
  }
  return { y, w: W, h: H, seed, jitter };
}

export const scenes = {
  // filled diamond centered at (20,8), radius 6
  diamond: (px, py) => {
    const dx = Math.abs(px - 20), dy = Math.abs(py - 8);
    return 6 - (dx + dy);
  },
  // horizontal bar across the middle → a pure-horizontal edge pair
  hbar: (px, py) => {
    const dy = Math.abs(py - 8);
    return dy < 2 ? 5 : -(dy - 2) * 4 - 1;
  },
  // vertical bar → pure-vertical edges
  vbar: (px, py) => {
    const dx = Math.abs(px - 16);
    return dx < 2 ? 5 : -(dx - 2) * 4 - 1;
  },
  // diagonal ridge (top-left to bottom-right)
  diag: (px, py) => 4 - Math.abs(px - py - 4),
  // two disks: one bright, one dim — a two-region segmentation task
  twodisk: (px, py) => {
    const d1 = Math.hypot(px - 9, py - 5) - 4;
    const d2 = Math.hypot(px - 23, py - 11) - 4;
    return Math.min(d1, d2 * 1.8);
  },
};

export function fnv1a64(bytes) {
  let h = 0xcbf29ce484222325n;
  for (const b of bytes) {
    h ^= BigInt(b & 0xff);
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h.toString(16).padStart(16, "0");
}
