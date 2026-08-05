// lightbox.ts — one full-screen frame, shared by /photographs and the home page.
//
// The behaviour used to live inside photographs.astro. It moved here so the home
// page can open a frame directly, at exactly the frame that was clicked, with
// the same flight: a copy of the clicked image flies from its place on the page
// to the middle of the screen (FLIP), and the source is hidden for exactly as
// long as the copy is in the air.
//
// The whole reel is embedded once as JSON by Lightbox.astro, so stepping ← →
// works past the frames a given page happens to show — the home strip and the
// photographs row step through the same 16.
//
// Both callers import this module, and Vite gives them one shared instance, so
// there is exactly one lightbox and one open index per page.

import { bi } from "./bilingual";

export interface Frame {
  src: string;
  srcset?: string;
  frame: string;
  /** Where it was made — rendered as `.le`/`.lz` spans, so the language toggle
   *  works on an open lightbox without re-opening it. */
  place: { en: string; zh: string };
}

let frames: Frame[] = [];
let root: HTMLElement | null = null;
let img: HTMLImageElement | null = null;
let back: HTMLElement | null = null;
let meta: HTMLElement | null = null;

/** Open index, or null when closed. */
let open: number | null = null;
/** The index currently painted — lags `open` while a swap animates. */
let shown: number | null = null;
let closing = false;
let srcRect: DOMRect | null = null;

const listeners = new Set<(i: number) => void>();

/** Called with the new index whenever the open lightbox moves. */
export function onFrameChange(fn: (i: number) => void) {
  listeners.add(fn);
}

export function isOpen(): boolean {
  return open !== null;
}

export function count(): number {
  return frames.length;
}

function num(i: number): string {
  return `${String(i + 1).padStart(2, "0")} / ${frames.length}`;
}

/**
 * The on-page element for a frame — the FLIP takeoff point. A page may show a
 * frame more than once (/photographs keeps both views in the DOM) or not at all
 * (stepping past the end of the home strip), so take the first one that is
 * actually laid out.
 */
function frameEl(i: number): HTMLElement | null {
  const all = document.querySelectorAll<HTMLElement>(`[data-frame][data-index="${i}"]`);
  for (const el of all) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) return el;
  }
  return null;
}

function hideSource(i: number) {
  showSources();
  const el = frameEl(i);
  if (el) el.style.visibility = "hidden";
}

function showSources() {
  document.querySelectorAll<HTMLElement>("[data-frame]").forEach((el) => {
    el.style.visibility = "";
  });
}

function paint(i: number) {
  const f = frames[i];
  if (!f || !img || !root) return;
  img.src = f.src;
  if (f.srcset) img.srcset = f.srcset;
  else img.removeAttribute("srcset");

  const set = (key: string, html: string) => {
    const el = root!.querySelector<HTMLElement>(`[data-lb="${key}"]`);
    if (el) el.innerHTML = html;
  };
  set("frame", f.frame);
  set("place", bi(f.place.en, f.place.zh));
  set("n", num(i));
}

function flyIn() {
  if (!img) return;
  const chrome = [back, meta].filter((el): el is HTMLElement => !!el);
  chrome.forEach((el) => {
    el.style.transition = "none";
    el.style.opacity = "0";
  });

  if (srcRect) {
    const to = img.getBoundingClientRect();
    if (to.width && to.height) {
      const sx = srcRect.width / to.width;
      const sy = srcRect.height / to.height;
      const dx = srcRect.left + srcRect.width / 2 - (to.left + to.width / 2);
      const dy = srcRect.top + srcRect.height / 2 - (to.top + to.height / 2);
      img.style.transition = "none";
      img.style.transform = `translate(${dx.toFixed(1)}px,${dy.toFixed(1)}px) scale(${sx.toFixed(4)},${sy.toFixed(4)})`;
      img.style.opacity = "1";
    }
  } else {
    img.style.transition = "none";
    img.style.transform = "scale(.94)";
    img.style.opacity = "0";
  }

  void img.offsetWidth;
  requestAnimationFrame(() => {
    if (!img) return;
    img.style.transition = "transform .48s cubic-bezier(.19,.86,.21,1),opacity .3s ease";
    img.style.transform = "none";
    img.style.opacity = "1";
    chrome.forEach((el) => {
      el.style.transition = "opacity .4s ease";
      el.style.opacity = "1";
    });
  });
}

function fadeSwap() {
  if (!img) return;
  img.style.transition = "none";
  img.style.opacity = "0";
  img.style.transform = "scale(.985)";
  void img.offsetWidth;
  requestAnimationFrame(() => {
    if (!img) return;
    img.style.transition = "opacity .26s ease,transform .34s cubic-bezier(.2,.8,.2,1)";
    img.style.opacity = "1";
    img.style.transform = "none";
  });
}

function sync() {
  if (open === null) {
    shown = null;
    return;
  }
  if (closing) return;
  if (shown === null) {
    shown = open;
    hideSource(open);
    flyIn();
  } else if (shown !== open) {
    shown = open;
    hideSource(open);
    fadeSwap();
  }
}

/** Open on frame `i`, flying from `sourceEl` if it is on screen. */
export function openLightbox(i: number, sourceEl?: HTMLElement | null) {
  if (!root || !frames.length) return;
  const el = sourceEl ?? frameEl(i);
  srcRect = el ? el.getBoundingClientRect() : null;
  open = i;
  paint(i);
  root.hidden = false;
  document.body.style.overflow = "hidden";
  listeners.forEach((fn) => fn(i));
  sync();
}

export function closeLightbox() {
  if (open === null || closing || !img) return;
  closing = true;

  const el = frameEl(open);
  const to = img.getBoundingClientRect();
  const src = el ? el.getBoundingClientRect() : srcRect;
  const chrome = [back, meta].filter((c): c is HTMLElement => !!c);

  chrome.forEach((c) => {
    c.style.transition = "opacity .3s ease";
    c.style.opacity = "0";
  });
  img.style.transition = "transform .44s cubic-bezier(.3,.72,.18,1)";
  if (src && to.width && to.height) {
    const sx = src.width / to.width;
    const sy = src.height / to.height;
    const dx = src.left + src.width / 2 - (to.left + to.width / 2);
    const dy = src.top + src.height / 2 - (to.top + to.height / 2);
    img.style.transform = `translate(${dx.toFixed(1)}px,${dy.toFixed(1)}px) scale(${sx.toFixed(4)},${sy.toFixed(4)})`;
  } else {
    img.style.transform = "scale(.94)";
    img.style.opacity = "0";
  }

  setTimeout(() => {
    closing = false;
    shown = null;
    open = null;
    showSources();
    if (root) root.hidden = true;
    document.body.style.overflow = "";
  }, 400);
}

/** Move the open lightbox by ±1, wrapping. No-op when closed. */
export function stepLightbox(d: number) {
  if (open === null || !frames.length) return;
  const n = frames.length;
  const i = (open + d + n) % n;
  open = i;
  paint(i);
  listeners.forEach((fn) => fn(i));
  sync();
}

/**
 * Wire the lightbox up. Called once by Lightbox.astro; safe to call again.
 * Frames come from the JSON the component embedded, so the reel is complete
 * even on a page showing four of it.
 */
export function initLightbox() {
  root = document.querySelector<HTMLElement>("[data-lb]");
  if (!root) return;

  const json = document.getElementById("ink-frames")?.textContent;
  frames = json ? JSON.parse(json) : [];

  img = root.querySelector<HTMLImageElement>('[data-lb-el="img"]');
  back = root.querySelector<HTMLElement>('[data-lb-el="back"]');
  meta = root.querySelector<HTMLElement>('[data-lb-el="meta"]');

  root.addEventListener("click", (e) => {
    // Clicks on the chrome (arrows, close) handle themselves; a click on the
    // backdrop or the photograph closes.
    if ((e.target as HTMLElement).closest("button")) return;
    closeLightbox();
  });
  root.querySelector("[data-lb-close]")?.addEventListener("click", (e) => {
    e.preventDefault();
    closeLightbox();
  });
  root.querySelectorAll<HTMLButtonElement>("[data-lb-step]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      stepLightbox(Number(b.dataset.lbStep));
    });
  });

  window.addEventListener("keydown", (e) => {
    if (open === null) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closeLightbox();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      stepLightbox(1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepLightbox(-1);
    }
  });
}
