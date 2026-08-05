// i18n.ts — every word the chrome and the page furniture says, in both
// languages.
//
// One build serves both: each string ships twice and `.scene[data-lang]` picks
// which copy is displayed (`.le` / `.lz` in global.css). The toggle stays
// instant, the URL stays stable, and nothing needs a server.
//
// ── The rule the two modes are held to ──────────────────────────────────────
//   EN  — English only. The 印章 stay, because a chop is a mark, not a word.
//   ZH  — Chinese only. One exception: a proper noun with no settled Chinese
//         form (GitHub, PyTorch, MONAI, ANTs, ISMRM, RSNA, FUJIFILM). Names
//         that DO have one are translated — 斯坦福大学, 领英, 谷歌学术.
//   Post bodies are authored English prose and stay English in both modes;
//   they are content, not chrome. Everything around them translates.
//
// Anything user-visible belongs here or in site.ts — never inline in a page.

/** A string in both languages. */
export interface L {
  en: string;
  zh: string;
}

/** Terse constructor so the tables below read as two columns. */
export const t = (en: string, zh: string): L => ({ en, zh });

/** Same word in both languages — a proper noun, an acronym, a number. */
export const both = (s: string): L => ({ en: s, zh: s });

export const ui = {
  nav: {
    home: t("Home", "首页"),
    writing: t("Writing", "文章"),
    photographs: t("Photographs", "摄影"),
    profile: t("Profile", "履历"),
  },

  // The masthead. The name carries the site; the line under it says plainly
  // what the site is, rather than naming it a second time.
  brand: {
    name: t("Yiheng Li", "李易恒"),
    sub: t("Personal website", "个人网站"),
  },

  home: {
    cue: t("Scroll", "向下滚动"),
    links: t("Links & contact", "链接与联系"),
    notes: t("Notes", "笔记"),
    photographs: t("Photographs", "摄影"),
    news: t("Research & news", "研究与近况"),
    toPhotographs: t("All photographs →", "全部摄影 →"),
    toWork: t("All work →", "全部工作 →"),
    /** "All 21 →" / "全部 21 篇 →" */
    allNotes: (n: number): L => t(`All ${n} →`, `全部 ${n} 篇 →`),
    /** "7 min" / "7 分钟" */
    min: (n: number): L => t(`${n} min`, `${n} 分钟`),
  },

  writing: {
    search: t("search titles & tags", "搜索标题与标签"),
    all: t("All", "全部"),
    /** "12 MIN" / "12 分钟" */
    min: (n: number): L => t(`${n} MIN`, `${n} 分钟`),
    fig: t("FIG. 1", "图 1"),
    /** Caption for the plate when the hovered note carries no figure. */
    noFig: t("NO FIGURE", "无图"),
    /** The note HAS a figure, but it is still coming down the wire. */
    developing: t("DEVELOPING…", "显影中…"),
    /**
     * The note has a figure and it did not load — several older notes hotlink
     * figures to hosts that have since 404'd or started blocking hotlinks.
     * Said out loud, because the alternative is leaving the previous note's
     * figure up under this note's title.
     */
    figGone: t("FIGURE UNAVAILABLE", "图片无法加载"),
  },

  photographs: {
    kicker: t("Photographs", "摄影"),
    camera: both("FUJIFILM X-T · 2022—2023"),
    /** The two arrangements: side by side, or one under the next. */
    viewRow: t("ROW", "横排"),
    viewColumn: t("COLUMN", "竖排"),
    close: t("CLOSE ✕", "关闭 ✕"),
    prevFrame: t("Previous photograph", "上一张"),
    nextFrame: t("Next photograph", "下一张"),
    /** "16 PHOTOGRAPHS" / "16 张" */
    count: (n: number): L => t(`${n} PHOTOGRAPH${n === 1 ? "" : "S"}`, `${n} 张`),
  },

  post: {
    contents: t("Contents", "目录"),
    prev: t("← PREVIOUS", "← 上一篇"),
    next: t("NEXT →", "下一篇 →"),
    copy: t("COPY", "复制"),
    copied: t("COPIED", "已复制"),
    copyLabel: t("Copy code", "复制代码"),
    /** "12 MIN" / "12 分钟" */
    min: (n: number): L => t(`${n} MIN`, `${n} 分钟`),
    /** Front-matter `categories`, as the kicker shows them. */
    category: (c: string): L => {
      const key = c.toUpperCase();
      const zh: Record<string, string> = {
        NOTE: "笔记",
        ARCHIVE: "存档",
        PROJECT: "项目",
        PROJECTS: "项目",
      };
      return t(key, zh[key] ?? key);
    },
  },

  profile: {
    kicker: t("Profile", "履历"),
    portraitPlace: t("LAKE TAHOE", "太浩湖"),
    metaRole: t(
      "Research Staff · Stanford University School of Medicine",
      "医学深度学习研究员 · 斯坦福大学医学院",
    ),
    metaPlace: t(
      "Division of Computational Medicine · Palo Alto, CA",
      "计算医学部 · 加州帕洛阿尔托",
    ),
    lead: t(
      "I build self-supervised foundation models for cancer imaging — pretrained on unlabeled 3D scans, fused with pathology and clinical records, and held to the questions oncology actually asks: a tumor's genotype, how far it has spread, how it responds to treatment over time.",
      "我做的是面向肿瘤影像的自监督基础模型——在无标注的三维影像上预训练，再与病理和临床数据融合，用来回答肿瘤学真正关心的问题：肿瘤的基因型、扩散到了哪一步、以及随时间如何响应治疗。",
    ),
    focus: t("Focus", "方向"),
    experience: t("Experience", "经历"),
    selectedWork: t("Selected work", "研究"),
    abstracts: t("Accepted abstracts", "会议摘要"),
    education: t("Education", "教育"),
    skills: t("Skills", "技能"),
    otherProjects: t("Other projects", "其他项目"),
    about: t("About", "关于"),
    moreFrames: t("More photographs", "更多摄影"),
    toPhotographs: t("PHOTOGRAPHS →", "摄影 →"),
    made: t("How this site is made", "这个站点是怎么做的"),
    // Both paragraphs state their facts and stop. No scene-setting.
    about1: t(
      "I'm Yiheng Li — Terry to most people. Outside work: guitar, singing, travel, badminton.",
      "我是李易恒，多数人叫我 Terry。工作之外：吉他、唱歌、旅行、羽毛球。",
    ),
    made1: t(
      "Astro, hosted on GitHub Pages. Hugo + PaperMod before that, Hexo + NexT before that. Type is Spectral, IBM Plex Mono, and Noto Serif SC.",
      "Astro 搭建，托管在 GitHub Pages。此前是 Hugo + PaperMod，更早是 Hexo + NexT。字体为 Spectral、IBM Plex Mono 与思源宋体。",
    ),
  },

  notFound: {
    plate: t("PAGE NOT FOUND", "页面未找到"),
    h1: t("This page doesn't exist.", "这个页面不存在。"),
    body: t(
      "404 — whatever was at this address has moved or been removed.",
      "404 —— 这个地址上原有的内容已经移走或删除。",
    ),
    reload: t("HOME", "首页"),
    photographs: t("PHOTOGRAPHS", "摄影"),
  },

  /** Labels for the mode toggle. It names the destination, not the state. */
  toggle: {
    day: t("DAY", "日间"),
    night: t("NIGHT", "夜间"),
  },
};
