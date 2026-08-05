# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Personal blog (`terryli710.github.io`) built with **Hugo** using the **PaperMod** theme. Pure static site — no application code, no package.json. Content is Markdown; the site is deployed to GitHub Pages.

## Commands

```bash
hugo server -D        # local dev server with drafts (draft: true posts), http://localhost:1313
hugo server           # local dev server, drafts hidden (matches production)
hugo --gc --minify    # production build into ./public (gitignored)
```

There are no tests or linters. CI (`.github/workflows/hugo.yaml`) builds with **Hugo extended v0.115.1** + Dart Sass on every push to `main` and deploys via GitHub Pages. Match that Hugo version locally to avoid build drift; the extended build is required (SCSS).

## Critical setup

The PaperMod theme is a **git submodule** (`themes/PaperMod`). After cloning, run `git submodule update --init --recursive` or the build fails. Do not edit files under `themes/PaperMod/` — override theme behavior by mirroring the file path under `layouts/` or `assets/` instead (Hugo's lookup order prefers the project root over the theme).

## Architecture

- **`config.yml`** — the single source of site config: menus, social icons, PaperMod params, KaTeX math (`math: true`, `math_renderer: katex`), and `markup.goldmark.renderer.unsafe: true` (raw HTML in Markdown is allowed). `configTaxo.yml` is a secondary taxonomy/privacy config, not loaded by default.
- **`content/`** — all posts and pages.
  - `content/posts/*.md` — blog posts. Front matter convention: `title`, `date`, `tags: [...]`, `categories` (e.g. `NOTE`, `ARCHIVE`), `description`, optional `draft: true`. Post-specific assets (images, PDFs) live in a sibling folder named after the post (e.g. `content/posts/r-survival/*.png` for `r-survival.md`) — this is a Hugo page bundle pattern; reference assets by bare filename.
  - `content/about/`, `content/resume/`, `content/search.md`, `content/archives.md` — special pages. `search.md` and `archives.md` set `layout:` to PaperMod's built-in `search`/`archives` layouts.
- **`layouts/`** — project-level overrides on top of the theme:
  - `layouts/shortcodes/pdfReader.html` — embeds a PDF inline: `{{<pdfReader "FILE.pdf">}}` where the PDF sits in the post's page-bundle folder. Used heavily in `content/posts/projects.md` and course-note posts.
  - `layouts/partials/extend_head.html` — injects KaTeX (or MathJax if `math_renderer: mathjax`) into pages where `math` is truthy. This is what makes `$...$` / `$$...$$` render.
- **`assets/css/extended/dracula.css`** — custom CSS layered over PaperMod (PaperMod auto-loads anything in `assets/css/extended/`).

## Math rendering

Math only renders when the post front matter (or site config) has `math: true`. KaTeX is the default renderer; delimiters are `$`/`$$` and `\(`/`\[` (configured in `extend_head.html`).
