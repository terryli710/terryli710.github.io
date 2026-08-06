// img.ts — responsive sources for the photographs.
//
// Every frame under public/img/g/ exists at two widths: the full 2800px file
// (what the design itself shipped) and a 1200px copy in public/img/g/thumb/.
// `npm run photos` writes both. A four-up grid should not pull 2800px files,
// and the lightbox should not be soft on a retina screen — srcset settles it.

/** "…/thumb/X.jpg 1200w, …/X.jpg 2800w" for a /img/g/X.jpg path. */
export function srcset(src: string): string | undefined {
  if (!src.startsWith("/img/g/") || src.includes("/thumb/")) return undefined;
  return `${src.replace("/img/g/", "/img/g/thumb/")} 1200w, ${src} 2800w`;
}

/** The 1200px copy, for places that never need more (grids, the plate pane). */
export function thumb(src: string): string {
  return src.startsWith("/img/g/") ? src.replace("/img/g/", "/img/g/thumb/") : src;
}

// `sizes` per context — how wide the image actually renders, so the browser can
// pick before layout. Deliberately approximate; over-picking costs bandwidth,
// under-picking costs sharpness.
export const SIZES = {
  /** home "Photographs" — four equal columns, two below 640px */
  preview: "(max-width:640px) 45vw, 22vw",
  /** photographs row — height-driven, so this is a working estimate */
  reel: "(max-width:820px) 90vw, 60vw",
  /** lightbox */
  full: "94vw",
  /** writing hover plate */
  plate: "(max-width:1000px) 0px, 23rem",
  /** profile "More frames" column */
  frames: "(max-width:820px) 100vw, 33vw",
};
