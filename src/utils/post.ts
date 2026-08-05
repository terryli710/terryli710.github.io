// post.ts — small helpers shared by the index, writing and post pages.

/** "2020·04·24" — the stamp format used throughout the design canvas. */
export function stamp(date: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${date.getUTCFullYear()}·${p(date.getUTCMonth() + 1)}·${p(date.getUTCDate())}`;
}

/** Whole minutes at 200 wpm, floored to 1. CJK is counted per character. */
export function readingTime(body: string): number {
  const cjk = (body.match(/[一-鿿぀-ヿ]/g) || []).length;
  const latin = body
    .replace(/[一-鿿぀-ヿ]/g, " ")
    .split(/\s+/)
    .filter(Boolean).length;
  return Math.max(1, Math.round((latin + cjk / 2.5) / 200));
}

/**
 * The first figure in a note — what /writing develops in the plate pane when
 * you hover its row. Taken from the post itself; never a stand-in photograph.
 *
 * Matches Markdown `![alt](src)` and raw `<img src="…" alt="…">`, whichever
 * comes first, and ignores images inside fenced code blocks. Returns null when
 * a note carries no figure, which is what suppresses its ◼ marker.
 */
export function firstFigure(body: string): { src: string; alt: string } | null {
  // Drop fenced code so a ![…](…) inside an example is not mistaken for a figure.
  const prose = body.replace(/^```[\s\S]*?^```/gm, "");

  const md = /!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/.exec(prose);
  const html = /<img\b[^>]*?\bsrc=["']([^"']+)["'][^>]*>/i.exec(prose);

  const first =
    md && html ? (md.index < html.index ? "md" : "html") : md ? "md" : html ? "html" : null;
  if (!first) return null;

  const src = first === "md" ? md![2] : html![1];
  if (!/^\/|^https?:/.test(src)) return null; // page-bundle relative leftovers

  let alt = first === "md" ? md![1] : (/alt=["']([^"']*)["']/i.exec(html![0])?.[1] ?? "");
  alt = alt.trim();

  return { src, alt };
}

/** Front-matter `categories` is a string in some posts and a list in others. */
export function categoryOf(categories?: string | string[]): string {
  if (!categories) return "NOTE";
  return (Array.isArray(categories) ? categories[0] : categories).toUpperCase();
}
