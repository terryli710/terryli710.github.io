function walk(node, visit, parent = null) {
  if (!node || typeof node !== "object") return;
  visit(node, parent);
  if (Array.isArray(node.children)) {
    node.children.forEach((child) => walk(child, visit, node));
  }
}

function hasDataProperty(node, name) {
  const properties = node.properties ?? {};
  return Object.prototype.hasOwnProperty.call(properties, name)
    || Object.prototype.hasOwnProperty.call(properties, name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`));
}

function isFootnoteBackref(node) {
  return node.type === "element" && hasDataProperty(node, "dataFootnoteBackref");
}

function findAnnotation(node) {
  if (!node || typeof node !== "object") return null;
  if (node.type === "element" && node.tagName === "annotation") return node;
  if (!Array.isArray(node.children)) return null;
  for (const child of node.children) {
    const annotation = findAnnotation(child);
    if (annotation) return annotation;
  }
  return null;
}

function textContent(node) {
  if (!node || typeof node !== "object" || isFootnoteBackref(node)) return "";
  if (node.type === "text") return node.value ?? "";
  // Math is left for MathJax as `data-tex` (see rehype-mathjax-tex); a popover
  // is plain text, so use the TeX source rather than recursing into the
  // fallback markup. `annotation` is the older KaTeX MathML shape.
  if (node.type === "element") {
    const tex = node.properties?.["data-tex"] ?? node.properties?.dataTex;
    if (typeof tex === "string") return tex;
    const classes = node.properties?.className;
    if (Array.isArray(classes) && classes.includes("katex")) {
      const annotation = findAnnotation(node);
      return annotation ? textContent(annotation) : "";
    }
  }
  if (!Array.isArray(node.children)) return "";
  return node.children.map(textContent).join(" ");
}

function addClass(node, className) {
  const current = node.properties?.className;
  const classes = Array.isArray(current) ? current : current ? [current] : [];
  if (!classes.includes(className)) classes.push(className);
  node.properties = { ...node.properties, className: classes };
}

export default function rehypeFootnotePopovers() {
  return (tree) => {
    const definitions = new Map();

    walk(tree, (node) => {
      if (node.type !== "element") return;
      const id = node.properties?.id;
      if (typeof id === "string" && id.startsWith("user-content-fn-")) {
        definitions.set(id, node);
      }
    });

    walk(tree, (node, parent) => {
      if (node.type !== "element" || node.tagName !== "a" || !hasDataProperty(node, "dataFootnoteRef")) return;

      const href = node.properties?.href;
      if (typeof href !== "string" || !href.startsWith("#")) return;
      const definition = definitions.get(decodeURIComponent(href.slice(1)));
      if (!definition) return;

      const host = parent?.type === "element" && parent.tagName === "sup" ? parent : node;
      addClass(host, "fn");

      const popoverText = textContent(definition).replace(/\s+/g, " ").trim();
      if (!popoverText) return;
      host.children.push({
        type: "element",
        tagName: "span",
        properties: { className: ["pop"], ariaHidden: "true" },
        children: [{ type: "text", value: popoverText }],
      });
    });
  };
}
