#!/usr/bin/env node
/**
 * add-photos — ingest camera originals into /photographs.
 *
 *   npm run photos
 *
 * Reads every image in ./photos-inbox (gitignored, never published), writes a
 * web-sized, EXIF-stripped copy to public/img/g/, and appends a caption stub to
 * src/data/frames.json. Then you edit the captions and delete the originals.
 *
 * EXIF is stripped on purpose: phone and camera files routinely carry GPS
 * coordinates, and publishing those publishes where you live and travel.
 *
 * Requires ImageMagick (`brew install imagemagick`). HEIC works if your
 * ImageMagick was built with libheif — `magick -list format | grep HEIC`.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const inbox = path.join(root, "photos-inbox");
const outDir = path.join(root, "public", "img", "g");
const thumbDir = path.join(outDir, "thumb");
const dataFile = path.join(root, "src", "data", "frames.json");

// Two tiers, wired up by src/utils/img.ts as a srcset:
//   full  — 2800px, the size the design itself shipped; lightbox and reel
//   thumb — 1200px, for the home 4-up, the writing plate and small viewports
const LONG_EDGE = 2800;
const QUALITY = 82;
const THUMB_EDGE = 1200;
const THUMB_QUALITY = 80;
const EXTS = new Set([".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"]);

function die(message) {
  console.error(`\n  ✗ ${message}\n`);
  process.exit(1);
}

try {
  execFileSync("magick", ["-version"], { stdio: "ignore" });
} catch {
  die("ImageMagick not found. Install it with:  brew install imagemagick");
}

if (!fs.existsSync(inbox)) {
  fs.mkdirSync(inbox, { recursive: true });
  console.log(`\n  Created ${path.relative(root, inbox)}/ — drop photos in there and run again.\n`);
  process.exit(0);
}

const files = fs
  .readdirSync(inbox)
  .filter((f) => EXTS.has(path.extname(f).toLowerCase()))
  .sort();

if (!files.length) {
  console.log(`\n  Nothing to do — ${path.relative(root, inbox)}/ is empty.\n`);
  process.exit(0);
}

fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(thumbDir, { recursive: true });
const frames = JSON.parse(fs.readFileSync(dataFile, "utf8"));
const known = new Set(frames.map((f) => f.src));
const added = [];

for (const file of files) {
  const base = path.basename(file, path.extname(file));
  const out = path.join(outDir, `${base}.jpg`);
  const src = `/img/g/${base}.jpg`;

  execFileSync("magick", [
    path.join(inbox, file),
    "-auto-orient",
    "-resize", `${LONG_EDGE}x${LONG_EDGE}>`,
    "-strip", // drops EXIF, including GPS
    "-quality", String(QUALITY),
    out,
  ]);
  execFileSync("magick", [
    path.join(inbox, file),
    "-auto-orient",
    "-resize", `${THUMB_EDGE}x${THUMB_EDGE}>`,
    "-strip",
    "-quality", String(THUMB_QUALITY),
    path.join(thumbDir, `${base}.jpg`),
  ]);

  const size = (fs.statSync(out).size / 1024).toFixed(0);
  if (known.has(src)) {
    console.log(`  ↻ ${base}.jpg  (${size} KB) — re-exported, already in frames.json`);
    continue;
  }

  // Recorded after -auto-orient, so these are the dimensions as displayed.
  // The pages need them to reserve space before the file lands, and the home
  // four-up needs them to tell a landscape frame from a portrait one.
  const [width, height] = execFileSync("magick", ["identify", "-format", "%w %h", out])
    .toString()
    .trim()
    .split(/\s+/)
    .map(Number);

  // A trailing run of digits is the frame number on most camera filenames.
  const frame = (base.match(/(\d{3,})\s*$/) || [, base])[1];
  // A photograph carries only what is true of it: the camera's own number, its
  // size, and where it was made. The place is written twice — the site is
  // bilingual and a photograph with no Chinese place reads English in Chinese
  // mode. Leaving `placeZh` empty is allowed: it falls back to the English and
  // the fill-in checklist reminds you (src/config/blanks.ts).
  added.push({
    src,
    frame,
    width,
    height,
    place: "TODO place",
    placeZh: "",
  });
  console.log(`  + ${base}.jpg  (${size} KB)`);
}

if (added.length) {
  // Newest first, matching how the reel reads.
  fs.writeFileSync(dataFile, JSON.stringify([...added, ...frames], null, 2) + "\n");
}

console.log(`
  ${files.length} file(s) processed → public/img/g/ (2800px) + /thumb (1200px)
  ${added.length} new entr${added.length === 1 ? "y" : "ies"} prepended to src/data/frames.json

  Next:
    1. Edit src/data/frames.json — replace every TODO place, and fill in
       placeZh so it reads Chinese in Chinese mode. Array order is display
       order.
    2. Delete the originals from photos-inbox/ once you are happy.
    3. npm run dev  →  http://localhost:4321/photographs/
       Anything still blank is listed in the terminal and in the panel in the
       bottom-right corner of the page.
`);
