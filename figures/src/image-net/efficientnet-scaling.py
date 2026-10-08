# Prints the coordinate block that is pasted into the .tex of the same name.
# generates the block stacks for efficientnet-scaling.tex
W = [8, 14, 25, 34, 42]          # baseline widths (#channels), stage 1..5
H = [16, 12, 8, 4.5, 2.8]        # baseline heights (resolution)
G2, G1 = 12, 7                   # gap between stages, between repeats
def stack(name, cx, ws, hs, reps):
    out, y, k = [], 0.0, 0
    for s,(w,h) in enumerate(zip(ws,hs)):
        for r in range(reps):
            if out: y += G1 if r else G2
            out.append(f"  \\blk{{{name}{k}}}{{{cx}}}{{{y:.2f}}}{{{w:g}}}{{{h:g}}}")
            y += h; k += 1
    out.append(f"  \\chain{{{name}}}{{{k-1}}}")
    return out, y
panels = [
  ("a", 0,   W, H, 1),
  ("b", 92, [2*w for w in W], H, 1),
  ("c", 162, W, H, 2),
  ("d", 236, W, [h*1.7 for h in H], 1),
  ("e", 336, [round(w*1.3,1) for w in W], [round(h*1.25,2) for h in H], 2),
]
for p in panels:
    lines, top = stack(*p)
    print(f"  % ({p[0]}) top at {top:.2f}")
    print("\n".join(lines))
