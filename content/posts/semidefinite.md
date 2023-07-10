---
title: Positive Semidefinite Matrix in Machine Learning
date: 2020-04-12 08:57:24
tags: [machine learning, linear algebra]
categories: NOTE
description: Positive Semidefinite Matrix is closely associated with some of machine learning algorithms.
---

## What is Positive Semidefinite (PSD) Matrix 

#### Definition

Matrix $A \in \mathbb{R}^{n\times n}$ is *positive semi-definite* (PSD), denoted $A \succeq 0$, is defined as:

- $A = A^{T}$ ($A$ is symmetric)
- $x^{T}Ax \geq 0$ for all $x \in \mathbb{R}$  

So from the definition, we can infer some properties of PSD matrix.

#### Properties

1. If $A \succeq 0 $ then $A$ is invertible and $A^{-1} \succeq 0$.

2. If $A \succeq 0 $ , then $\forall Q \in \mathbb{R}^{n\times n}$, we have $Q^{T}AQ \succeq 0$.

3.  $A \succeq 0 \iff \text{Eigenvalues of } A \geq 0$  ($A$ is PSD if and only if all the eigenvalues of $A$ are non-negative)

4. For a semidefinite matrix $A$, we can find two matrices $P$ and $D \in \mathbb{R^{n\times n}}$, such that 
   $$
   M =  P^{-1}DP
   $$
   ​	where $P$  is an orthogonal matrix, $P = [p^{(1)}，p^{(2)}，\dots,p^{(n)}]$ , $p^{(i)}$ are eigenvectors of $A$; and $D$ is a diagonal matrix whose main diagonal contains the corresponding eigenvalues. This is called an [**eigen decomposition**](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) of M.



## Why Is It Important in Machine Learning

There are some connections that I found (though I believe they are not the whole picture) between PSD matrix and machine learning algorithms. Here are these connections.

#### Covariance Matrix of a Multi-variate Random Variable is PSD

For example, $\overrightarrow X = [X_1, X_2, \dots, X_n] \in N(\mathbf{\mu}, \mathbf{\Sigma})$, is a multivariate Gaussian distribution where **each $X_i \in N(\mu, \sigma_i^2)$ is a Gaussian distribution** and **every linear combination of $X_i$s has a univariate normal distribution**. The [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) $\mathbf{\Sigma}$ is a positive semidefinite matrix, where $(i,j)$ entry is the covariance (or variance):
$$
K_{X_i,X_j} = cov(X_i, X_j) = E(X_iX_j^{T}) - \mu_i\mu_j
$$


#### Quadratic Form of a PSD is Convex

If $A$ is a PSD matrix, then the quadratic form $f(\mathbf{x}) = x^TAx$ is a convex function. (In fact $\nabla f(\mathbf{x}) = Ax$ and $\nabla^2 f(\mathbf{x}) = A$). Idea of proof: $f(\mathbf{x}) = \sum_{i=1}^n\lambda_i z_i^2$ , where $\lambda_i$ are eigenvalues of $A$ , and $z_i = Px_i$ is a changing-based $x_i$. This proof used the fact that PSE can be diagonalized.

[Another property](https://wiki.math.ntnu.no/_media/tma4180/2016v/note2.pdf) , that is similar, is listed here without explanations: 

> A twice differentiable function $f:\mathbb{R}^n \to \mathbb{R}$ is **convex**, if and only if the Hessian $\nabla^2 f(x)$ is positive semi-definite for all $x \in \mathbb{R}^n$.



#### Spectral Theorem and Eigenvalues

> If $A\in \mathbb{R}^{n\times n}$ is symmetric, then  $A$ has $n$ orthogonal eigenvectors with real eigenvalues.

\- Spectral Theorem

Along with the property that $A$‘s eigenvalues are $\geq 0$, we can conclude that:  
$$
A \succeq 0 \iff \text{Eigenvalues of } A \geq 0
$$

#### Programming 

[Semidefinite programming](https://en.wikipedia.org/wiki/Semidefinite_programming) (SDP) is a subfield of convex optimization concerned with the optimization of a linear objective function (a user-specified function that the user wants to minimize or maximize) over the intersection of the cone of positive semidefinite matrices with an affine space.



[Quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming) could be simplified if $A$ is positive definite. The problem is a special case of the more general field of convex optimization.