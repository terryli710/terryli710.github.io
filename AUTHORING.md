# Authoring — terryli710.github.io

Everything you need to run the site day to day. **This file is never published.**
Astro only builds what is under `src/pages/` and `public/`; Markdown at the repo
root is not a route, so this stays between you and the repo.

---

## Setup, once

```bash
fnm use 20 && npm install
```

Node 20+ is required (Astro 4). `.node-version` pins it, so `fnm use` in the repo
root picks it up on its own.

Everyday loop:

```bash
npm run dev
```

→ http://localhost:4321

---

## 1. Adding photos

Photographs live in two places: the **image** in `public/img/g/`, and its **row**
in `src/data/frames.json`. The script handles the first and stubs the second.

A photograph carries only what is true of it — the number the camera gave it,
its pixel size, and where it was made. There are no captions, no roll numbers
and no film stock: these are digital files, and the site used to invent darkroom
provenance for them.

```bash
npm run photos
```

**Step by step**

1. Drop camera originals — any size, JPG/PNG/HEIC/TIFF — into `photos-inbox/`.
   That folder is gitignored: originals never get committed or published.
2. Run `npm run photos`. Each file is rotated upright, **stripped of EXIF**, and
   written at two sizes — `public/img/g/<name>.jpg` (2800px, the size the design
   itself shipped; used by the lightbox and the row) and
   `public/img/g/thumb/<name>.jpg` (1200px, used by the home 4-up and small
   screens). The two are wired together as a `srcset` by `src/utils/img.ts`, so
   nothing downloads a 2800px file to fill a thumbnail. A stub row is
   prepended to `src/data/frames.json`.
3. Open `src/data/frames.json` and replace the `TODO`s:

   ```json
   {
     "src": "/img/g/DSCF1234.jpg",
     "frame": "1234",          // the camera's own number, from the filename
     "width": 2800,
     "height": 1576,           // both written by the script
     "place": "Fuji Five Lakes",
     "placeZh": "富士五湖"
   }
   ```

   `placeZh` falls back to the English while it is empty, and anything still
   falling back is listed by the fill-in checklist (§3).

   **Array order is display order** — the first entry is the one the row opens
   on.
4. Delete the originals from `photos-inbox/` once you are happy.

> **EXIF is stripped deliberately.** Several of the current originals carried GPS
> coordinates. Publishing those publishes where you live and travel. The script
> removes them; don't add images to `public/img/g/` by hand and skip that.

**Where photos show up**

| Place | Source |
| --- | --- |
| `/photographs/` row + column + lightbox | every entry in `frames.json` |
| Home "Photographs" strip | every entry in `frames.json` — the whole set, dragged sideways; a click opens that one full screen |
| `/writing/` hover plate | **the note's own first figure** — extracted by `firstFigure()` in `src/utils/post.ts`; nothing to configure |
| `/profile/` "More photographs" | three picked **by camera number** at the top of `src/pages/profile.astro` (`6862`, `7125`, `3921`) — captions come from `frames.json` |
| Home hero, profile About band, 404 plate | `public/img/hero.jpg` |
| `/profile/` portrait | `public/img/portrait-a.jpg` |

To swap the hero or portrait, overwrite that file — but resize and strip it
first:

```bash
magick <original> -auto-orient -resize 2800x2800\> -strip -quality 82 public/img/hero.jpg
```

---

## 2. Adding a post

```bash
npm run post -- "Survival Analysis, Revisited"
npm run post -- "A Note on Kernels" --math --tags "cs229, kernels"
```

Writes `src/content/posts/<slug>.md`. Flags: `--math` (the post uses `$…$` or
`$$…$$`), `--draft` (kept out of the build), `--tags "a, b"`.

**Front matter**

| Key | Notes |
| --- | --- |
| `title` | The `<h1>`. Quote it if it contains `:` or `--`. |
| `date` | Drives ordering and the `2020·04·24` stamp. |
| `description` | **The large grey lead line under the title.** One sentence. Skipping it leaves a visible gap — worth writing. |
| `tags` | A list. Feeds the `/writing/` filter bar and the `#TAG` strip. |
| `categories` | `NOTE` · `CASE` · `ARCHIVE` · `MATERIAL`. Shown in the kicker. |
| `math` | `true` only if the post has TeX. |
| `draft` | `true` hides it everywhere. |

**Things you get for free**

- Reading time and prev/next are computed — don't set them.
- Code blocks get the bordered frame and COPY button automatically.
- `$…$` / `$$…$$` are typeset by MathJax in the Pagella math face. Display
  equations are auto-numbered `(1)`, `(2)` in the right margin.
- Footnotes (`[^1]`) get a hover plate over the marker.
- `##` / `###` headings become the sticky Contents rail.

**Images inside a post**

Put them in `public/img/<post-slug>/` and reference them absolutely:

```markdown
![Decision boundary](/img/generative-models/gda_visual.png)
```

**PDFs inside a post** — rename the file to `.mdx`, then:

```mdx
import PdfReader from "../../components/PdfReader.astro";

<PdfReader src="/pdf/cs229/ps1.pdf" />
```

**The writing index plate** — a note's row shows a `◼` in the seal accent and develops a figure
in the side pane when the note contains one. It uses **the note's own first
figure**, found automatically; there is nothing to wire up. Add a figure to a
note and the marker appears. 18 of the 20 published notes currently have one —
the two that do not are the `.mdx` PDF posts.

Hovering a note with **no** figure empties the plate to a bare frame captioned
`NO FIGURE`, rather than leaving the previous note's figure up under the new
note's title. Nothing to configure; it follows from the note having no figure.

11 of those 18 plates are still hotlinked from other sites (imgur, wikipedia,
blog posts). They work, but they are someone else's bandwidth and can vanish. If
you want them local, save each into `public/img/<post-slug>/` and change the URL
in the note — the plate follows automatically.

The other 7 are local. Six of them are generated: see
[`scripts/figures.py`](scripts/figures.py), which draws the diagrams for the
notes that had none, in the day-mode palette so they sit flush on the page in
DAY and read as prints in NIGHT. Edit that script and re-run it rather than
hand-patching the SVGs, which carry computed curve points.

---

## 3. Still needs you — the fill-in checklist

**Everything blank is in one file: [`src/config/me.ts`](src/config/me.ts).** Open
it, fill in what you know, leave the rest.

You do not have to remember what is missing. Anything still empty is reported
two ways, and both disappear the moment you fill it in:

1. **The terminal.** `npm run dev` and `npm run build` each print a numbered
   checklist — what is missing, where on the site it shows, and the exact line
   to type:

   ```
   Yiheng Li — 4 things still to fill in

   01. LinkedIn profile URL
       shows: home → Elsewhere · profile → LINKEDIN button
       fix:   src/config/me.ts → linkedin: "https://www.linkedin.com/in/…"
   ```

2. **The page.** While `npm run dev` is running, a small seal-coloured **"4 to
   fill in"** tab sits in the bottom-right corner of every page. Click it for the
   same list. It is rendered only in development — it is not in the built site
   at all, so it can never be published by accident.

**A blank link is left out, not published dead.** An unfilled entry is dropped
from the built page entirely — the Elsewhere list and the profile buttons simply
show one fewer item, and the keypoint project shows its venue without a link.
Nothing on the live site ever points nowhere.

| Field in `me.ts` | What it wants | State |
| --- | --- | --- |
| `email` | The address on the Elsewhere list and the EMAIL button. | ⚠️ still `hello@terryli.me`, which came from the design mockup — **confirm or replace** |
| `linkedin` | Full profile URL. | ✅ |
| `scholar` | Google Scholar citations page. Bare `user=` id only — `authuser=` is your own browser's session and sends readers to the wrong account. | ✅ |
| `stanford` | Your Stanford people page. | ✅ |
| `orcid` | ORCID record — the persistent id publishers and indexes key off. | ✅ |
| `github` | GitHub profile. | ✅ |
| `resume` | The RÉSUMÉ button. Easiest: drop the PDF at `public/resume.pdf`, then put `/resume.pdf` here. | ⬜ |
| `keypointWriteup` | "Selected work" → the keypoint-detection entry. Leave blank if nothing is public. | ⬜ optional |

**Those links are also stated for machines.** `person` in `src/config/site.ts`
collects them into a schema.org `Person` block (`sameAs`), emitted as JSON-LD on
every page, and the profile buttons carry `rel="me"`. That is how a search engine
learns this site, that ORCID record and that Scholar profile are one person — so
add a new profile in `me.ts` and it joins the identity graph automatically.

The checklist also covers **photographs with no Chinese place** (`placeZh` in
`frames.json`), since those read English in Chinese mode.

Two more, outside all of that:

- **`nlp_insights.md` has no `description`** — it is a draft, so nothing renders
  it yet.
- **The hero photograph** is `DSCF6560` ("One bird, unplanned"), taken from the
  design itself. It appears in three places: home hero, profile About band, 404
  plate. To change it, overwrite `public/img/hero.jpg`.

---

## 3a. The two languages

The toggle in the top bar switches the whole site between English and Chinese.
There is **one build**: every label ships twice and CSS shows one of the pair, so
switching is instant and the URL never changes.

The rule the site is held to:

- **EN — English only.** The only Chinese left is the two 印章 (a chop is a mark,
  not a word) and the `中` on the language button itself.
- **ZH — Chinese only.** The only Latin left is proper nouns with no settled
  Chinese form: GitHub, PyTorch, MONAI, ANTs, ISMRM, RSNA, FUJIFILM. Names that
  *do* have one are translated — 斯坦福大学, 领英, 谷歌学术.
- **Post bodies stay English in both modes.** They are content, not chrome. The
  title, kicker, reading time, contents rail and prev/next around them translate.

**Where the words live**

| File | Holds |
| --- | --- |
| `src/config/i18n.ts` | every chrome/UI string, as a two-column `t("English", "中文")` table |
| `src/config/site.ts` | the CV, publications, links, news — same two-column shape |
| `src/data/frames.json` | photo places, via `placeZh` |

**Adding a translated string.** Put the pair in `i18n.ts`, then render it with
the `T` component — never write user-visible text straight into a page:

```astro
<p class="pf-sec"><T t={ui.profile.experience} /></p>
```

For text that lands in an *attribute* rather than a text node, write
`data-t-<attr>-en` / `-zh` and BaseLayout repaints it on toggle:

```astro
<input data-t-placeholder-en="search titles & tags" data-t-placeholder-zh="搜索标题与标签" />
```

For text a *script* writes at runtime, build the pair with `bi()` from
`src/scripts/bilingual.ts` — plain text would freeze in whichever language was
showing when it was written.

## 4. Deploying

**`main` is the live site. Nothing else is.**
`.github/workflows/astro-deploy.yml` triggers on `push:` to `main` and on manual
`workflow_dispatch` — those are the only two things that can change what is
served at https://terryli710.github.io.

So: work on a branch, push it as often as you like, and the published site does
not move.

```bash
git switch -c some-change     # anything but main
```

`npm run dev` and `npm run build` are local only — neither one can reach the
live site, no matter what they produce. Merging to `main` is the single moment
the site changes, and it is deliberate.

Worth doing once, if you want the guarantee enforced rather than remembered:
turn on branch protection for `main` in the repo settings (Settings → Branches),
so a stray `git push` cannot go straight to production.

`.github/workflows/hugo.yaml` is **retired** — it used to deploy the old Hugo
site on every push to `main` and raced the Astro workflow for the same Pages
environment. Its `push` trigger is removed; it is manual-only, kept as a rollback
path. Once you are confident, delete it along with the Hugo tree: `config.yml`,
`configTaxo.yml`, `content/`, `layouts/`, `assets/`, `themes/`, `.hugo_build.lock`.

Those Hugo files are dead weight but harmless — Astro never reads them and they
are not published.

---

## 5. Where things are

```
src/
  config/me.ts        ← THE ONLY FILE WITH BLANKS IN IT
  config/i18n.ts      every UI string, English and Chinese side by side
  config/blanks.ts    turns whatever is still empty into the fill-in checklist
  config/site.ts      nav, CV, publications, links — most editable copy
  data/frames.json    photographs: number, size, place (script-writable)
  data/frames.ts      photograph shape + derived counts
  content/posts/      the 22 notes
  pages/              index · writing · photographs · profile · 404 · posts/[...slug]
  layouts/            BaseLayout (chrome, mode/lang, progress) · PostLayout
  components/         Chrome · Footer · Seal · T · Lightbox · FillIn · PdfReader
  scripts/lightbox.ts the full-screen view — shared by /photographs and home
  scripts/bilingual.ts the `.le`/`.lz` pair, for captions written by script
  styles/global.css   the whole design system, one file
  utils/post.ts       reading time · date stamp · first-figure extraction
  utils/img.ts        srcset for the photographs
  plugins/            math → MathJax handoff · footnote popovers
scripts/              add-photos · new-post
public/img/g/         photographs (2800px) + thumb/ (1200px)
```

**The top bar** is one component, `Chrome.astro`, on every page including home.
On home it is passed `home` and becomes fixed rather than sticky: it stays on
screen the whole way down, transparent and light-on-dark over the hero
photograph, then takes the page's own colours once the photograph has scrolled
past (`data-solid`, set from the shared `ink:scroll` tick).

**Design tokens** live at the top of `global.css`: `--bg --paper --ink --soft
--faint --line --line2 --seal --hi --onseal --imgf`, one block per mode. Change
a colour there and it moves everywhere.

The palette is **azur** — the harbour-blue room. The design canvas carries six
(sépia, nymphéas, azur, marine, brume, étang) behind a `[data-pal]` switch; the
site ships one, inlined onto `[data-mode]`. `--seal` is the accent as it sits on
the page and darkens in DAY to hold contrast; `--hi` is that accent at its NIGHT
brightness in both modes, for surfaces that stay dark either way (the lightbox).

Four places are deliberately *not* tokenised, because they sit over a
photograph and must not follow the mode: the home chrome and scroll cue
(`#eef6f9`), the lightbox chrome (`rgba(228,238,242,…)`), the profile band
caption (`#e6eef1`), and the 404 plate (`rgba(236,245,248,…)`). Their grounds
are `rgba(4,9,12,…)`. If you change palette, change these by hand too.

Type: **Spectral** (body), **IBM Plex Mono** (all chrome/labels), **Noto Serif
SC** (Chinese + seals), **STIX Two Text** (math fallback). Loaded from Google
Fonts in `BaseLayout.astro`.

The seals — 虛室生白 (footer), 大象無形 (end of post) — use **traditional**
glyphs on purpose. Body and UI copy stay simplified. Don't "fix" them.
