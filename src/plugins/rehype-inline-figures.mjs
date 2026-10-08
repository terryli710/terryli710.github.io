// rehype-inline-figures — put a post's own drawn figures into the page.
//
// `![alt](/img/fig/<post>/<name>.svg)` is built by figures/build.py with every
// colour as a CSS variable (--ink, --fa ...). Through <img> those variables
// never resolve — an image document cannot see the page's CSS — so the figure
// would lose its colours, and with them the DAY/NIGHT switch. Inlined, it is
// part of the page and follows the theme like the text around it.
//
// Only /img/fig/ is inlined. Anything else, including the older hand-made
// SVGs elsewhere under /img/, stays an ordinary <img>.
import { readFileSync } from "node:fs";
import { fromHtml } from "hast-util-from-html";

const ROOT = new URL("../../public", import.meta.url);

function walk(node, visit) {
  if (!node || !Array.isArray(node.children)) return;
  node.children.forEach((child, i) => {
    visit(child, i, node);
    walk(child, visit);
  });
}

export default function rehypeInlineFigures() {
  return (tree) => {
    walk(tree, (node, index, parent) => {
      if (node.type !== "element" || node.tagName !== "img") return;
      const src = String(node.properties?.src ?? "");
      if (!src.startsWith("/img/fig/") || !src.endsWith(".svg")) return;

      const file = new URL(`.${src}`, ROOT + "/");
      const svg = fromHtml(readFileSync(file, "utf8"), { fragment: true, space: "svg" }).children.find(
        (n) => n.type === "element" && n.tagName === "svg",
      );
      if (!svg) throw new Error(`rehype-inline-figures: no <svg> in ${src}`);

      // The alt text is the figure's accessible name, as it was on the <img>.
      const alt = String(node.properties?.alt ?? "");
      svg.properties = { ...svg.properties, role: "img", ariaLabel: alt || undefined };
      parent.children[index] = svg;
    });
  };
}
