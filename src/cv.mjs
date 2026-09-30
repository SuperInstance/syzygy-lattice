// src/cv.mjs — computer vision that runs ON THE ENCODED LATTICE.
// Never touches pixels. The image is text now; the algorithms read bytes.
// Outputs are typed detections with coordinates in CELLS, ready to be sealed
// as receipts (src/receipts.mjs) or addressed like canvas cells.
import { codeClass } from "./codec.mjs";

// Connected components over the braille dot grid, expanded to pixel dots
// (each cell = 2×4 possible dots). 8-connectivity, union-find, deterministic
// (labels in scan order, merged upward).
export function components(enc, minDots = 3) {
  const dotW = enc.cols * 2, dotH = enc.rows * 4;
  const has = (x, yy) => {
    if (x < 0 || yy < 0 || x >= dotW || yy >= dotH) return false;
    const c = (x >> 1), r = (yy >> 2); // cell = 2 wide × 4 tall — row shift is ÷4
    const dx = x & 1, dy = yy & 3;
    const DOT = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80]; // row-major 2×4
    return (enc.braille[r * enc.cols + c] & DOT[dy * 2 + dx]) !== 0;
  };
  const label = new Int32Array(dotW * dotH).fill(-1);
  const parent = [];
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const boxes = new Map(); // root -> {x0,y0,x1,y1,mass}
  let next = 0;
  for (let yy = 0; yy < dotH; yy++) {
    for (let x = 0; x < dotW; x++) {
      if (!has(x, yy)) continue;
      const neigh = [];
      for (const [nx, ny] of [[x-1,yy],[x,yy-1],[x-1,yy-1],[x+1,yy-1]]) {
        if (nx >= 0 && ny >= 0 && label[ny * dotW + nx] >= 0) neigh.push(find(label[ny * dotW + nx]));
      }
      const idx = yy * dotW + x;
      if (neigh.length === 0) {
        label[idx] = next; parent.push(next); next++;
        boxes.set(label[idx], { x0: x, y0: yy, x1: x, y1: yy, mass: 1 });
      } else {
        let root = Math.min(...neigh);
        label[idx] = root;
        const b = boxes.get(root);
        b.x0 = Math.min(b.x0, x); b.y0 = Math.min(b.y0, yy); // joins must GROW the box both ways
        b.x1 = Math.max(b.x1, x); b.y1 = Math.max(b.y1, yy);
        b.mass++;
        for (const other of neigh) {
          if (other === root) continue;
          const ro = find(other);
          if (ro === root) continue;
          parent[ro] = root;
          const bo = boxes.get(ro);
          b.x0 = Math.min(b.x0, bo.x0); b.y0 = Math.min(b.y0, bo.y0);
          b.x1 = Math.max(b.x1, bo.x1); b.y1 = Math.max(b.y1, bo.y1);
          b.mass += bo.mass;
          boxes.delete(ro);
        }
      }
    }
  }
  const out = [];
  for (const [root, b] of boxes) {
    if (b.mass < minDots) continue;
    out.push({
      kind: "component",
      mass: b.mass,
      bbox_dot: [b.x0, b.y0, b.x1, b.y1],
      bbox_cell: [b.x0 >> 1, b.y0 >> 2, b.x1 >> 1, b.y1 >> 2],
    });
  }
  out.sort((a, b) => b.mass - a.mass || a.bbox_dot[0] - b.bbox_dot[0]);
  return out;
}

// Orientation histogram from the GLYPH layer: each cell votes its class,
// weighted by band (brighter cells = more confident structure).
// Returns the majority class and per-class vote totals.
export function orientationField(enc) {
  const votes = { flat: 0, h: 0, v: 0, dp: 0, dn: 0 };
  for (let i = 0; i < enc.classes.length; i++) votes[enc.classes[i]] += enc.glyphs[i] + 1;
  const total = Object.values(votes).reduce((a, b) => a + b, 0);
  const majority = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  return { votes, majority_class: majority[0], majority_share: majority[1] / total };
}

// Two-region segmentation by band threshold on the glyph layer: a binary
// split of cells into {low, high} regions; reports each region's cell count
// and bounding box. Reads ONLY the band bytes — pure text operation.
export function bandSegment(enc, cut = 5) {
  const regions = { low: { cells: 0, bbox: null }, high: { cells: 0, bbox: null } };
  for (let r = 0; r < enc.rows; r++) {
    for (let c = 0; c < enc.cols; c++) {
      const i = r * enc.cols + c;
      const side = enc.glyphs[i] >= cut ? "high" : "low";
      regions[side].cells++;
      const b = regions[side].bbox;
      if (!b) regions[side].bbox = [c, r, c, r];
      else { b[0] = Math.min(b[0], c); b[1] = Math.min(b[1], r); b[2] = Math.max(b[2], c); b[3] = Math.max(b[3], r); }
    }
  }
  return regions;
}
