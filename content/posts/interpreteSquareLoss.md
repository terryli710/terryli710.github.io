---
title: Probabilistic Interpretation of Sum of Square Loss Function
date: 2020-04-13 21:11:55
tags: [machine learning, loss function, probability]
categories: NOTE
description: One way to interpret the form of "square" in square loss.

---

## Square Loss Function (in Linear Regression)

For linear regression, the way that we used to find the optimal parameters $\overrightarrow \theta$ is called gradient descent, which we seek for $\overrightarrow \theta$ that minimize the loss function:
$$
\mathcal{J}(\theta) = \frac{1}{2} \sum_{i=1}^{n}(y^{(i)} - \theta^T x^{(i)})^2
$$
That is:
$$
\hat \theta = \underset{\theta}{\mathrm{argmin}}[\frac{1}{2} \sum_{i=1}^{n}(y^{(i)} - \theta^T x^{(i)})^2]
$$

## Interpret the Loss Function as MLE

In linear regression, we assume the model to be:
$$
\overrightarrow y = \theta^T x^{(i)} + \epsilon^{(i)}
$$
where $\epsilon$ is called the error term which conposes of unmodelled factors and random noise. And under general assumption, $\epsilon^{(i)}$s are gaussian random variables that are independent from each other
$$
\epsilon \in \text{iid }N(0,\sigma^2) 
$$
Under this assumption, the distribution of $y$ can be expressed as
$$
P(y^{(i)}|x^{(i)};\theta) = \frac{1}{\sqrt{2\pi}\sigma} \text{exp}(\frac{-(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2})
$$
That indicates
$$
y^{(i)} \sim \mathcal{N}(\theta^Tx^{(i)}, \sigma^2)
$$
The likelihood function
$$
\begin{align}
\mathcal{L}(\theta) &= P(\overrightarrow y|X; \theta) \\
&= \prod_{i=1}^{n} P(y^{(i)}|X^{(i)}; \theta) \\
&= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} \text{exp}(\frac{-(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}
$$
The log likelihood function
$$
\mathcal{l}(\theta) = n\log(\frac{1}{\sqrt{2\pi}\sigma}) + \sum_{i=1}^{n} \frac{-(y^{(i)} - \theta^Tx^{(i)})^2}{2\sigma^2}
$$
We can see here
$$
l(\theta) \propto  - \frac{1}{2} \sum_{i=1}^{n}(y^{(i)} - \theta^T x^{(i)})^2 = - \mathcal{J}(\theta)
$$
So we can see minimizing loss function $\mathcal{J}$ is actually equivalent to find the maximum likelihood estimation of $\overrightarrow y$.



## References

1. [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
2. [Square loss](http://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/lossfunctions/SquaredLoss.pdf)