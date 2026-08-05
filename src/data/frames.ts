// frames.ts — the photographs.
//
// One continuous row: every photograph keeps its own aspect ratio, the centre
// one is the one being looked at, its neighbours only half in view.
//
// A photograph carries only what is true of it: the number the camera gave it,
// its pixel size, and where it was made. It carries no caption, no roll number
// and no film stock — these are digital frames, and inventing darkroom
// provenance for them was a lie the design told on the reader's behalf.
//
// The photographs themselves live in `frames.json` so `npm run photos` can
// append to them safely. Edit places there; edit the shape and the derived
// values here. See AUTHORING.md (not published) for the full workflow.

import data from "./frames.json";
import { ui, t, type L } from "../config/i18n";

/** A row of frames.json, exactly as it is written on disk. */
export interface FrameSource {
  /** Absolute site path, e.g. /img/g/DSCF6560.jpg */
  src: string;
  /** The number the camera gave it — the `6560` of `DSCF6560.jpg`. */
  frame: string;
  /** Pixel size of the 2800px export, after -auto-orient. Written by
   *  `npm run photos`; used to reserve space and to tell landscape from
   *  portrait without loading the file. */
  width: number;
  height: number;
  /** Where it was made. */
  place: string;
  /** The same place in Chinese. Falls back to `place` while it is missing. */
  placeZh?: string;
}

/** A photograph as the pages consume it: the place already in both languages. */
export interface Frame {
  src: string;
  frame: string;
  width: number;
  height: number;
  place: L;
}

const rows = data as FrameSource[];

const filled = (s: string | undefined): string => (s ?? "").trim();

export const reel: Frame[] = rows.map((f) => ({
  src: f.src,
  frame: f.frame,
  width: f.width,
  height: f.height,
  place: t(f.place, filled(f.placeZh) || f.place),
}));

/**
 * The home "Photographs" band: four frames side by side in equal columns.
 * Landscape only — the canvas picks four landscape frames, and mixing in a
 * portrait makes one column three times the height of its neighbours. Falls
 * back to the head of the reel if there are ever fewer than four.
 */
export const preview: Frame[] = (() => {
  const wide = reel.filter((f) => f.width > f.height);
  return (wide.length >= 4 ? wide : reel).slice(0, 4);
})();

/** "16 PHOTOGRAPHS" / "16 张" */
export const gcount = ui.photographs.count(reel.length);

/**
 * Photographs still showing English in Chinese mode. Surfaced by the fill-in
 * checklist (src/config/blanks.ts) rather than left to be spotted on the page.
 */
export const untranslatedFrames = rows.filter((f) => !filled(f.placeZh)).map((f) => f.frame);

// NOTE: /writing's plate pane is no longer keyed off a hand-written map. It
// derives from each note's OWN first figure — see `firstFigure()` in
// src/utils/post.ts. A note with no figure simply gets no ◼ marker.
