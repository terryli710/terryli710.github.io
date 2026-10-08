# Writes the .tex next to this script; rebuild the figure with figures/build.py.
import numpy as np
rng = np.random.default_rng(7)
def ring(n, r, s):
    t = rng.uniform(0, 2*np.pi, n); rr = r + rng.normal(0, s, n)
    return rr*np.cos(t), rr*np.sin(t)
ox, oy = ring(90, 1.0, .06); ix, iy = ring(40, .4, .05)
def tab(x, y):
    return "\n".join(f"  {a:.3f} {b:.3f} {a*a+b*b:.3f}\\\\" for a, b in zip(x, y))
tex = r"""% Kernel-method post, figure 2: two classes on concentric rings are not
% linearly separable in R^2 (left); the feature map z = x^2 + y^2 lifts them
% to R^3, where the plane z = 0.55 separates them (right). The post calls the
% transform "somewhat like (x - mu)^2"; with mu at the origin this is it.
% Toy data, deterministic: numpy default_rng(7); outer ring 90 points at
% radius N(1, .06), inner ring 40 points at radius N(.4, .05), angles uniform
% (inner = fa triangles, outer = fd circles, as in the original's markers).
\documentclass[tikz,border=3pt]{standalone}
\input{FIGKIT/preamble.tex}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\begin{document}
\pgfplotstableread[row sep=\\]{
  x y z\\
OUTER
}\outer
\pgfplotstableread[row sep=\\]{
  x y z\\
INNER
}\inner
\pgfplotsset{
  base/.style={
    axis line style={draw=faint, line width=.5pt},
    every tick/.style={draw=faint, line width=.5pt},
    major tick length=1.2mm,
    tick label style={font=\ttfamily\scriptsize, text=soft},
    label style={font=\small, text=ink},
    title style={font=\rmfamily\small, text=ink},
    grid style={draw=rule, line width=.3pt},
    xlabel style={sloped=false}, ylabel style={sloped=false, rotate=0},
  },
  outer/.style={only marks, mark=*, mark size=1.5pt,
    mark options={draw=fd, fill=fd, fill opacity=.45, line width=.4pt}},
  inner/.style={only marks, mark=triangle*, mark size=1.9pt,
    mark options={draw=fa, fill=fa, fill opacity=.45, line width=.4pt}},
}
\begin{tikzpicture}[fig]
  \begin{axis}[base, name=flat, width=58mm, height=58mm, scale only axis=false,
      axis equal image, xmin=-1.4, xmax=1.4, ymin=-1.4, ymax=1.4,
      xtick={-1,0,1}, ytick={-1,0,1}, axis lines=box,
      xlabel={$x$}, ylabel={$y$},
      ylabel style={rotate=-90},
      title={in $\mathbb{R}^2$: not linearly separable}]
    \addplot[outer] table[x=x, y=y] {\outer};
    \addplot[inner] table[x=x, y=y] {\inner};
  \end{axis}
  \begin{axis}[base, name=lift, at={($(flat.east)+(26mm,0)$)}, anchor=west, clip=false,
      width=62mm, height=58mm, view={35}{18},
      xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, zmin=0, zmax=1.4,
      xtick={-1,0,1}, ytick={-1,0,1}, ztick={0,.5,1},
      grid=major, axis lines=box,
      xlabel={}, ylabel={}, zlabel={},
      title={in $\mathbb{R}^3$: a plane separates them}]
    \addplot3[inner] table {\inner};
    \addplot3[surf, shader=flat, draw=fe, line width=.5pt, fill=fe, fill opacity=.18,
      samples=2, domain=-1.2:1.2, y domain=-1.2:1.2] {0.55};
    \addplot3[outer] table {\outer};
    \node[font=\small, yshift=-8mm] at (axis cs:0,-1.2,0) {$x$};
    \node[font=\small] at (axis cs:2.0,.1,0) {$y$};
    \node[font=\small, anchor=south] at (axis cs:-1.2,-1.2,1.47) {$z$};
  \end{axis}
  % the feature map, between the panels
  \draw[arr=fe] ($(flat.east)+(3mm,0)$) -- ++(14mm,0)
    node[midway, above=1mm, font=\footnotesize, text=ink] {$\phi$}
    node[midway, below=1mm, font=\footnotesize, text=soft] {$z=x^2+y^2$};
\end{tikzpicture}
\end{document}
"""
tex = tex.replace("OUTER", tab(ox, oy)).replace("INNER", tab(ix, iy))
open(__import__('pathlib').Path(__file__).with_suffix('.tex'), "w").write(tex)
