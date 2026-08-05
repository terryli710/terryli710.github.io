#!/usr/bin/env node
/**
 * new-post — scaffold a note.
 *
 *   npm run post -- "Survival Analysis, Revisited"
 *   npm run post -- "A Note on Kernels" --math --tags "cs229, kernels"
 *
 * Writes src/content/posts/<slug>.md with front matter the collection schema
 * accepts, and prints where it landed. See AUTHORING.md (not published).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);

const flag = (name) => argv.includes(`--${name}`);
const value = (name) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? null : argv[i + 1];
};

const title = argv.filter((a) => !a.startsWith("--") && argv[argv.indexOf(a) - 1] !== "--tags")[0];

if (!title) {
  console.error(`
  Usage:  npm run post -- "Your Title" [--math] [--draft] [--tags "a, b"]
`);
  process.exit(1);
}

const slug = title
  .toLowerCase()
  .replace(/['’]/g, "")
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-|-$/g, "");

const file = path.join(root, "src", "content", "posts", `${slug}.md`);
if (fs.existsSync(file)) {
  console.error(`\n  ✗ ${path.relative(root, file)} already exists.\n`);
  process.exit(1);
}

const tags = (value("tags") || "")
  .split(",")
  .map((t) => t.trim())
  .filter(Boolean);

// Local time with offset — the same shape the migrated posts use.
const now = new Date();
const pad = (n) => String(n).padStart(2, "0");
const offset = -now.getTimezoneOffset();
const date =
  `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` +
  `T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}` +
  `${offset >= 0 ? "+" : "-"}${pad(Math.floor(Math.abs(offset) / 60))}:${pad(Math.abs(offset) % 60)}`;

const front = [
  "---",
  `title: ${/[:#\-]/.test(title) ? JSON.stringify(title) : title}`,
  `date: ${date}`,
  `tags: [${tags.join(", ")}]`,
  "categories: NOTE",
  // Quoted, not bare: a bare `description:` is YAML null, which the collection
  // schema rejects. An empty string is valid and simply renders no lead line.
  'description: ""',
  ...(flag("math") ? ["math: true"] : []),
  ...(flag("draft") ? ["draft: true"] : []),
  "---",
  "",
  "",
].join("\n");

fs.mkdirSync(path.dirname(file), { recursive: true });
fs.writeFileSync(file, front);

console.log(`
  ✓ ${path.relative(root, file)}

  `+`description: fills the large grey line under the title — write one sentence.
  categories:  NOTE · CASE · ARCHIVE · MATERIAL (shown as the kicker)
  math: true   only if the post uses $…$ or $$…$$

  npm run dev  →  http://localhost:4321/posts/${slug}/
`);
