import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeMathjaxTex from "./src/plugins/rehype-mathjax-tex.mjs";
import rehypeFootnotePopovers from "./src/plugins/rehype-footnote-popovers.mjs";
import rehypeInlineFigures from "./src/plugins/rehype-inline-figures.mjs";

// https://astro.build/config
export default defineConfig({
  site: "https://terryli710.github.io",
  base: "/",
  integrations: [mdx(), sitemap()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathjaxTex, rehypeFootnotePopovers, rehypeInlineFigures],
    // The design sets code in the body ink on the page ground, with only
    // keywords and comments carrying colour — so a single theme, restyled by
    // `.po-code` in global.css, rather than a dual light/dark pair.
    shikiConfig: {
      theme: "css-variables",
      langs: ["r", "python", "bash", "yaml", "ts", "js"],
      langAlias: { R: "r" },
    },
  },
  image: {
    // No image transforms are used anywhere in this site (plain <img> tags
    // referencing /public assets) — avoid a hard dependency on the `sharp`
    // native binary, which isn't installed and isn't guaranteed to build in
    // every environment.
    service: { entrypoint: "astro/assets/services/noop" },
  },
});
