// src/receipts.mjs — the vision ledger.
// Every detection is sealed as a receipt in the quilt/cell-receipt@v1 shape
// (the same schema quilt-canvas-tui's fabric seals) — op DETECT, addr =
// cell-space address, result = the typed detection. A vision pipeline whose
// outputs are receipts: a forged "I saw X at cell (a,b)" fails chain
// verification, exactly like a forged memory claim in PoEM's threat model.
// Chain rule is public; see test pins for what that means and doesn't.
import { createHash } from "node:crypto";

const stable = (o) => {
  if (o === null || typeof o !== "object") return JSON.stringify(o);
  if (Array.isArray(o)) return "[" + o.map(stable).join(",") + "]";
  const keys = Object.keys(o).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stable(o[k])).join(",") + "}";
};

export class VisionLedger {
  constructor() { this.receipts = []; }
  seal(op, addr, result) {
    const parent = this.receipts.length ? this.receipts[this.receipts.length - 1].receipt_id : null;
    const rid = createHash("sha256").update(stable({ op, addr, result, parent }), "utf8").digest("hex").slice(0, 16);
    const rec = { schema: "quilt/cell-receipt@v1", receipt_id: rid, parent, op, addr, result };
    this.receipts.push(rec);
    return rec;
  }
  sealDetections(detections) {
    const out = [];
    for (const d of detections) {
      const addr = d.kind === "component" ? `cells(${d.bbox_cell.join(",")})` : d.kind;
      out.push(this.seal("DETECT", addr, d));
    }
    return out;
  }
  verifyChain() {
    let parent = null;
    for (let i = 0; i < this.receipts.length; i++) {
      const r = this.receipts[i];
      if (r.parent !== parent) return { ok: false, at: i, why: `parent ${r.parent} != expected ${parent}` };
      const want = createHash("sha256").update(stable({ op: r.op, addr: r.addr, result: r.result, parent }), "utf8").digest("hex").slice(0, 16);
      if (r.receipt_id !== want) return { ok: false, at: i, why: "receipt_id mismatch (content or order edited)" };
      parent = r.receipt_id;
    }
    return { ok: true, len: this.receipts.length, tip: parent };
  }
}

// decode: reconstruct an approximate luma frame from the braille layer alone.
// An independent path back to pixels — "re-executable by a stranger": given
// only the text, anyone can rebuild the picture within the threshold's band.
export function decodeToLuma(enc, w = 32, h = 16) {
  const y = new Uint8Array(w * h);
  const DOT = [0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80];
  for (let r = 0; r < enc.rows; r++) {
    for (let c = 0; c < enc.cols; c++) {
      const mask = enc.braille[r * enc.cols + c];
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const on = (mask & DOT[dy * 2 + dx]) !== 0;
          // invert the threshold: dot on ≈ mid-of-upper-half luma, off ≈ mid-of-lower
          const v = on ? 177 : 77;
          const px = c * 2 + dx, py = r * 4 + dy;
          if (px < w && py < h) y[py * w + px] = v;
        }
      }
    }
  }
  return { y, w, h };
}
