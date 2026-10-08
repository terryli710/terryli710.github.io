# Writes the .tex next to this script; rebuild the figure with figures/build.py.
import numpy as np
from sklearn.mixture import GaussianMixture
rng=np.random.default_rng(1)
n=400
X=np.vstack([rng.multivariate_normal([1,2],[[2,0],[0,.5]],n),rng.multivariate_normal([-3,-5],np.eye(2),n)])
gm=GaussianMixture(2,covariance_type='full',random_state=0,n_init=3).fit(X)
lab=gm.predict(X); order=np.argsort(-gm.means_[:,1])
tables=[];ells=[]
for rank,k in enumerate(order):
    P=X[lab==k]
    rows='\n'.join(f'  {a:.3f} {b:.3f}\\\\' for a,b in P)
    tables.append(f"\\pgfplotstableread[row sep=\\\\]{{\n  x y\\\\\n{rows}\n}}\\comp{'AB'[rank]}\n")
    w,V=np.linalg.eigh(gm.covariances_[k]); a,b=np.sqrt(w[1]),np.sqrt(w[0])
    th=np.arctan2(V[1,1],V[0,1]); mx,my=gm.means_[k]
    c,s=np.cos(th),np.sin(th)
    ells.append(dict(mx=mx,my=my,a=a,b=b,c=c,s=s,w=gm.weights_[k],cov=gm.covariances_[k]))
acc=['fa','fd']
plots=[]
for rank,e in enumerate(ells):
    for r,lw,op in [(1,.8,1),(2,.7,.75),(3,.6,.5)]:
        plots.append(f"    \\addplot[draw={acc[rank]}, line width={lw}pt, draw opacity={op}, domain=0:360, samples=120, smooth]\n"
          f"      ({{{e['mx']:.4f} + {r}*({e['a']*e['c']:.4f}*cos(x) - {e['b']*e['s']:.4f}*sin(x))}},\n"
          f"       {{{e['my']:.4f} + {r}*({e['a']*e['s']:.4f}*cos(x) + {e['b']*e['c']:.4f}*sin(x))}});")
hdr=f"""% EM post, figure 1: 2-D data and the fitted two-component Gaussian mixture.
% Illustrative redraw of MathWorks' "Cluster Data Using a Gaussian Mixture
% Model" figure 1 (scatter plot and fitted GMM contours; recovered via the
% Wayback Machine). Generating model matched to the original plot: 400 points (the
% original drew 1000) from each of N([1 2], diag(2, .5)) and
% N([-3 -5], I), here drawn with numpy default_rng(1). The mixture was fitted
% by EM (sklearn GaussianMixture, full covariances, random_state=0); points are
% coloured by the fitted component that claims them, and each component is
% drawn from its fitted mean and covariance as its 1, 2 and 3 sigma
% (Mahalanobis) ellipses. Fitted parameters:
%   A: w={ells[0]['w']:.3f} mu=({ells[0]['mx']:.3f}, {ells[0]['my']:.3f}) Sigma=[{ells[0]['cov'][0,0]:.3f} {ells[0]['cov'][0,1]:.3f}; {ells[0]['cov'][1,0]:.3f} {ells[0]['cov'][1,1]:.3f}]
%   B: w={ells[1]['w']:.3f} mu=({ells[1]['mx']:.3f}, {ells[1]['my']:.3f}) Sigma=[{ells[1]['cov'][0,0]:.3f} {ells[1]['cov'][0,1]:.3f}; {ells[1]['cov'][1,0]:.3f} {ells[1]['cov'][1,1]:.3f}]
% Regenerate with figures/src/em-algorithm/gmm-clusters.py.
\\documentclass[tikz,border=3pt]{{standalone}}
\\input{{FIGKIT/preamble.tex}}
\\usepackage{{pgfplots}}\\pgfplotsset{{compat=1.18}}
\\begin{{document}}
"""
body=f"""\\pgfplotsset{{
  base/.style={{
    axis line style={{draw=faint, line width=.5pt}},
    every tick/.style={{draw=faint, line width=.5pt}},
    major tick length=1.2mm,
    tick label style={{font=\\ttfamily\\scriptsize, text=soft}},
    label style={{font=\\small, text=ink}},
  }},
  pts/.style={{only marks, mark=*, mark size=.75pt,
    mark options={{draw=#1, fill=#1, draw opacity=0, fill opacity=.38}}}},
}}
\\begin{{tikzpicture}}[fig]
  \\begin{{axis}}[base, width=92mm, scale only axis, axis equal image, clip=false,
      xmin=-7, xmax=7, ymin=-8.5, ymax=5,
      xtick={{-6,-4,...,6}}, ytick={{-8,-6,...,4}},
      axis x line=bottom, axis y line=left,
      x axis line style={{-}}, y axis line style={{-}},
      xlabel={{$x_1$}}, ylabel={{$x_2$}}, ylabel style={{rotate=-90}}]
    \\addplot[pts=fa] table {{\\compA}};
    \\addplot[pts=fd] table {{\\compB}};
{chr(10).join(plots)}
    \\node[font=\\small, text=fa, anchor=west] at (axis cs:{ells[0]['mx']+3*ells[0]['a']+.3:.2f},{ells[0]['my']+1.4:.2f}) {{component 1}};
    \\node[font=\\small, text=fd, anchor=west] at (axis cs:{ells[1]['mx']+3*ells[1]['a']+.3:.2f},{ells[1]['my']+1.4:.2f}) {{component 2}};
    \\node[lab, anchor=north west, align=left] at (axis cs:2.2,-5.2) {{CONTOURS: 1, 2, 3 SIGMA\\\\OF EACH FITTED GAUSSIAN}};
  \\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
open(__import__('pathlib').Path(__file__).with_suffix('.tex'),'w').write(hdr+''.join(tables)+body)
