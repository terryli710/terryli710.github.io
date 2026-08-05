// ═══════════════════════════════════════════════════════════════════════════
//  FILL THIS IN — this is the only file with blanks in it.
//
//  Leave a value as "" and the site keeps working. Nothing breaks and nothing
//  renders as a dead link: an unfilled entry is simply left out of the built
//  site, and instead it is reported to you two ways —
//
//    1. the terminal — `npm run dev` and `npm run build` each print a
//       checklist saying what is missing, where it shows, and what to type;
//    2. the page — a small "TO FILL IN" panel in the bottom-right corner
//       while `npm run dev` is running. Development only; never published.
//
//  Fill one in, save, and it disappears from both.
// ═══════════════════════════════════════════════════════════════════════════

export const me = {
  /** Shown on the home "Elsewhere" list and the profile EMAIL button. */
  email: "li.terry710@gmail.com",

  /** Full profile URL, e.g. "https://www.linkedin.com/in/yiheng-li-xxxx" */
  linkedin: "https://www.linkedin.com/in/yiheng-li/",

  /** Google Scholar citations page, e.g. "https://scholar.google.com/citations?user=XXXX"
   *  Keep it to the bare `user=` id: `hl=` and especially `authuser=` are your
   *  own browser's session, and `authuser=2` sends other readers to whichever
   *  Google account happens to be second in *their* browser. */
  scholar: "https://scholar.google.com/citations?user=jivWdwkAAAAJ",

  /** Your Stanford people page. */
  stanford: "https://profiles.stanford.edu/yiheng-li",

  /** ORCID — the persistent id publishers and indexes key off. Full URL. */
  orcid: "https://orcid.org/0000-0003-0135-8824",

  /** The RÉSUMÉ button.
   *  Easiest: drop your PDF at public/resume.pdf, then put "/resume.pdf" here. */
  resume: "/resume.pdf",

  /** GitHub — already correct, here so every outbound link lives in one place. */
  github: "https://github.com/terryli710",
};

export type MeKey = keyof typeof me;

/**
 * What each blank is for, in the words the checklist uses. Keeping this beside
 * the values means the reminder can say *where on the site* the gap shows,
 * rather than just naming a variable.
 *
 * `optional: true` — the site is complete without it; it is listed as a
 * suggestion rather than as something missing.
 */
export const FIELDS: Record<MeKey, { what: string; where: string; example: string; optional?: boolean }> = {
  email: {
    what: "Your email address",
    where: "home → Elsewhere · profile → EMAIL button",
    example: "hello@terryli.me",
  },
  linkedin: {
    what: "LinkedIn profile URL",
    where: "home → Elsewhere · profile → LINKEDIN button",
    example: "https://www.linkedin.com/in/yiheng-li-1a2b3c",
  },
  scholar: {
    what: "Google Scholar citations page",
    where: "home → Elsewhere · profile → SCHOLAR button",
    example: "https://scholar.google.com/citations?user=XXXXXXXX",
  },
  stanford: {
    what: "Stanford people page",
    where: "home → Elsewhere · profile → STANFORD button",
    example: "https://profiles.stanford.edu/yiheng-li",
  },
  orcid: {
    what: "ORCID record",
    where: "home → Elsewhere · profile → ORCID button · the page's identity data",
    example: "https://orcid.org/0000-0003-0135-8824",
  },
  resume: {
    what: "Résumé PDF",
    where: "profile → RÉSUMÉ button (the seal-red one)",
    example: "/resume.pdf  — drop the file at public/resume.pdf",
  },
  github: {
    what: "GitHub profile URL",
    where: "home → Elsewhere · profile → GITHUB button",
    example: "https://github.com/terryli710",
  },
};

/** True when the field has a real value. */
export const has = (key: MeKey): boolean => me[key].trim().length > 0;
