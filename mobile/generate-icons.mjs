// Generates deterministic installable 192px/512px PNG icons without extra deps.
// The source icon.svg remains the scalable website icon.
import { writeFileSync, mkdirSync } from "node:fs";
import { deflateSync } from "node:zlib";

function crc32(buffer) {
  let n = 0xffffffff;
  for (const b of buffer) {
    n ^= b;
    for (let i = 0; i < 8; i++) n = (n >>> 1) ^ ((n & 1) ? 0xedb88320 : 0);
  }
  return (n ^ 0xffffffff) >>> 0;
}
function chunk(name, data) {
  const type = Buffer.from(name, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([type, data])));
  return Buffer.concat([length, type, data, checksum]);
}
function inTriangle(x, y, a, b, c) {
  const sign = (p, q, r) => (p[0] - r[0]) * (q[1] - r[1]) -
    (q[0] - r[0]) * (p[1] - r[1]);
  const p = [x, y];
  const d1 = sign(p, a, b), d2 = sign(p, b, c), d3 = sign(p, c, a);
  return !(d1 < 0 || d2 < 0 || d3 < 0) || !(d1 > 0 || d2 > 0 || d3 > 0);
}
function createIcon(size) {
  // RGBA scanlines with filter type 0; compatible with PNG decoders.
  const raw = Buffer.alloc(size * (1 + size * 4));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size, v = y / size;
      const r = Math.hypot(u - .5, v - .5);
      // Maskable-safe composition: all content inside central 72% region.
      const onA = inTriangle(u, v, [.31, .68], [.44, .28], [.5, .28]) ||
        inTriangle(u, v, [.5, .28], [.56, .28], [.69, .68]) ||
        (u >= .38 && u <= .62 && v >= .53 && v <= .59);
      const notch = inTriangle(u, v, [.5, .41], [.465, .52], [.535, .52]);
      const foreground = onA && !notch;
      const outline = r < .38 && r > .365;
      const color = foreground ? [248, 255, 252] :
        outline ? [84, 220, 199] : [12, 25, 37];
      const i = y * (1 + size * 4) + 1 + x * 4;
      raw[i] = color[0]; raw[i + 1] = color[1];
      raw[i + 2] = color[2]; raw[i + 3] = 255;
    }
  }
  const signature = Buffer.from("89504e470d0a1a0a", "hex");
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 6; // RGBA, 8-bit
  const png = Buffer.concat([
    signature, chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
  const target = "icons/icon-" + size + ".png";
  writeFileSync(target, png);
  console.log("Created " + target + " (" + png.length + " bytes)");
}
mkdirSync("icons", { recursive: true });
for (const size of [192, 512]) createIcon(size);
