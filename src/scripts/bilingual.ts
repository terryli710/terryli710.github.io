// bilingual.ts — the one thing client scripts need from the language system.
//
// Most labels are rendered twice by T.astro and chosen by CSS. A label that a
// script writes at runtime — a status bar, a caption pane, a lightbox — has to
// build the same pair by hand, or it would freeze in whatever language was
// showing when it was written.

/** `<span class="le">en</span><span class="lz">zh</span>`, safely escaped. */
export function bi(en = "", zh = ""): string {
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return `<span class="le">${esc(en)}</span><span class="lz">${esc(zh)}</span>`;
}
