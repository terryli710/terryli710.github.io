import { defineCollection, z } from "astro:content";

// Permissive schema — mirrors the existing Hugo front matter conventions in
// content/posts/*.md (see CLAUDE.md) so migrated posts validate as-is:
//   title, date, tags: [...], categories (NOTE/ARCHIVE/CASE/MATERIAL/...),
//   description, optional draft, optional math (KaTeX toggle), optional cover.
const posts = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).optional().default([]),
    categories: z.union([z.string(), z.array(z.string())]).optional(),
    // `.nullable()` so a bare `description:` (YAML null, easy to leave behind
    // when drafting) degrades to "no lead line" instead of failing the build.
    description: z.string().nullable().optional(),
    draft: z.boolean().optional().default(false),
    math: z.boolean().optional().default(false),
    cover: z.string().optional(),
  }),
});

export const collections = { posts };
