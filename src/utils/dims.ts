// dims.ts — intrinsic pixel size of a published image, read at build time.
//
// Why this exists: the photographs row is height-driven (`height:100%;width:auto`),
// so a frame whose bytes have not arrived yet is 0px wide. With 13 of 16 frames
// lazy, the reel measured 3531px on load and 8543px once everything had landed —
// every scroll-snap point sliding sideways underneath the reader as they went.
// Scrolling *back* through frames they had already passed was the worst of it,
// because the geometry was no longer the geometry they scrolled through.
//
// Handing the browser `width`/`height` fixes it at the source: the intrinsic
// ratio is known before the image loads, so each frame reserves its final width
// immediately and `loading="lazy"` still defers the bytes.
//
// This module touches the filesystem — it is for `.astro` frontmatter only,
// never a client `<script>`. That is why it is not part of img.ts.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pub = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../public");

export interface Dim {
  width: number;
  height: number;
}

/**
 * Width and height from the JPEG's SOF segment.
 *
 * Every file under public/img/g/ is a baseline JPEG written by ImageMagick
 * (see scripts/add-photos.mjs), so parsing the header beats pulling in an image
 * library. Anything else returns null and the caller simply omits the attributes.
 */
function jpegSize(buf: Buffer): Dim | null {
  if (buf.length < 4 || buf.readUInt16BE(0) !== 0xffd8) return null;
  let i = 2;
  while (i + 9 < buf.length) {
    if (buf[i] !== 0xff) {
      i++; // resync past padding
      continue;
    }
    const marker = buf[i + 1];
    // Fill bytes, and the standalone markers that carry no length field.
    if (marker === 0xff) {
      i++;
      continue;
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      i += 2;
      continue;
    }
    if (marker === 0xda) break; // start of scan — the header is behind us
    // SOF0–SOF15, minus DHT (c4), JPG (c8) and DAC (cc), which are not frames.
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7) };
    }
    i += 2 + buf.readUInt16BE(i + 2);
  }
  return null;
}

const cache = new Map<string, Dim | null>();

/**
 * Intrinsic size of a site-absolute image path (`/img/g/DSCF6560.jpg`), or null
 * if the file is missing or is not a JPEG. Cached — the reel asks sixteen times
 * per page and every page asks again in dev.
 */
export function dims(src: string): Dim | null {
  if (cache.has(src)) return cache.get(src)!;
  let out: Dim | null = null;
  try {
    // Read the header only; these are 2800px files and the SOF sits near the top.
    const fd = fs.openSync(path.join(pub, src.replace(/^\//, "")), "r");
    try {
      const buf = Buffer.alloc(65536);
      const read = fs.readSync(fd, buf, 0, buf.length, 0);
      out = jpegSize(buf.subarray(0, read));
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    out = null; // not published yet, or not a file we can measure
  }
  cache.set(src, out);
  return out;
}
