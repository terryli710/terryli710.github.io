/**
 * rehype-mathjax-tex — hand math off to MathJax at runtime, the way the design
 * canvas does.
 *
 * `remark-math` leaves the TeX in a code element:
 *   inline   <code class="language-math math-inline">…</code>
 *   display  <pre><code class="language-math math-display">…</code></pre>
 *
 * This rewrites them into the contract the canvas established:
 *   inline   <span class="mr" data-tex="…">
 *   display  <div class="po-eq"><span class="mr" data-tex="…" data-tex-display>
 *              <span class="n">(1)</span></div>
 *
 * `.po-eq` is the rule-topped, centred, horizontally scrollable band the canvas
 * uses for display equations, numbered in the right margin. The raw TeX stays
 * as the element's text so the page is still readable if the MathJax CDN never
 * answers; `setMath()` in PostLayout replaces it with typeset CHTML and marks
 * the node `data-tex-set`.
 */
import { visit } from "unist-util-visit";

function classesOf(node) {
  const value = node.properties?.className;
  if (!value) return [];
  return Array.isArray(value) ? value : String(value).split(/\s+/);
}

function texOf(node) {
  let out = "";
  visit(node, "text", (t) => {
    out += t.value;
  });
  return out.trim();
}

function mathSpan(tex, display) {
  return {
    type: "element",
    tagName: "span",
    properties: { className: ["mr"], dataTex: tex, ...(display ? { dataTexDisplay: "" } : {}) },
    children: [{ type: "text", value: tex }],
  };
}

/** The <code> carrying display math, if this node is its <pre> wrapper. */
function displayCodeOf(node) {
  if (node.tagName !== "pre") return null;
  const kids = (node.children || []).filter((c) => c.type === "element");
  if (kids.length !== 1 || kids[0].tagName !== "code") return null;
  return classesOf(kids[0]).includes("math-display") ? kids[0] : null;
}

export default function rehypeMathjaxTex() {
  return (tree) => {
    let n = 0;

    visit(tree, "element", (node, index, parent) => {
      if (!parent || index === null) return;

      const display = displayCodeOf(node);
      if (display) {
        n += 1;
        parent.children[index] = {
          type: "element",
          tagName: "div",
          properties: { className: ["po-eq"] },
          children: [
            mathSpan(texOf(display), true),
            {
              type: "element",
              tagName: "span",
              properties: { className: ["n"] },
              children: [{ type: "text", value: `(${n})` }],
            },
          ],
        };
        return ["skip"];
      }

      if (node.tagName === "code" && classesOf(node).includes("math-inline")) {
        parent.children[index] = mathSpan(texOf(node), false);
        return ["skip"];
      }
    });
  };
}
