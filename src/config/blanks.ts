// blanks.ts — what is still waiting on you.
//
// Everything the design specified but cannot know is collected in me.ts (links)
// and frames.json (Chinese captions). This module turns whatever is still empty
// into one checklist, and that checklist is shown three ways:
//
//   · the terminal, once per `npm run dev` / `npm run build` / `npm run blanks`
//   · a small panel in the corner of every page — DEV ONLY, never published
//   · nothing else: an unfilled link is left out of the site rather than
//     published as a dead one
//
// Nothing here runs in the browser except the panel's own markup.

import { me, FIELDS, type MeKey } from "./me";
import { untranslatedFrames } from "../data/frames";

export interface Blank {
  /** Stable id, e.g. "me.linkedin". */
  id: string;
  /** What is missing, in one line. */
  what: string;
  /** Where the gap shows on the site. */
  where: string;
  /** The edit that closes it. */
  fix: string;
  /** The site is complete without it — a suggestion, not a hole. */
  optional: boolean;
}

const missingLinks: Blank[] = (Object.keys(FIELDS) as MeKey[])
  .filter((key) => !me[key].trim())
  .map((key) => ({
    id: `me.${key}`,
    what: FIELDS[key].what,
    where: FIELDS[key].where,
    fix: `src/config/me.ts → ${key}: "${FIELDS[key].example}"`,
    optional: FIELDS[key].optional ?? false,
  }));

const missingCaptions: Blank[] = untranslatedFrames.length
  ? [
      {
        id: "frames.zh",
        what: `Chinese place for ${untranslatedFrames.length} photograph${untranslatedFrames.length === 1 ? "" : "s"}`,
        where: "photographs · home — these read English in Chinese mode",
        fix: `src/data/frames.json → add "placeZh" to ${untranslatedFrames.slice(0, 6).join(", ")}${untranslatedFrames.length > 6 ? " …" : ""}`,
        optional: false,
      },
    ]
  : [];

export const blanks: Blank[] = [...missingLinks, ...missingCaptions];

/** Blanks that actually leave a gap, as opposed to suggestions. */
export const required = blanks.filter((b) => !b.optional);

/** The checklist as plain text — what the terminal and `npm run blanks` print. */
export function report(): string {
  if (!blanks.length) {
    return "\n  Yiheng Li — nothing left to fill in.\n";
  }

  const lines = blanks.map((b, i) => {
    const n = String(i + 1).padStart(2, "0");
    const tag = b.optional ? " (optional)" : "";
    return `  ${n}. ${b.what}${tag}\n      shows: ${b.where}\n      fix:   ${b.fix}`;
  });

  const head = required.length
    ? `${required.length} thing${required.length === 1 ? "" : "s"} still to fill in`
    : "nothing required — one optional suggestion";

  return `\n  Yiheng Li — ${head}\n\n${lines.join("\n\n")}\n\n  Unfilled links are left out of the built site, not published dead.\n`;
}

// Print once per process. Astro evaluates this module a single time per dev
// server / build, so this is the "tell me in the terminal" half of the answer.
declare global {
  // eslint-disable-next-line no-var
  var __inkBlanksReported: boolean | undefined;
}
if (!globalThis.__inkBlanksReported) {
  globalThis.__inkBlanksReported = true;
  if (blanks.length) console.log(report());
}
