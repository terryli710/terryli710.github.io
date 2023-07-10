---
title: Kernel Method Note
date: 2020-05-12 11:48:35
tags: [machine learning, algorithm, kernel]
categories: NOTE
description: Note down some facts about kernel method.
---

## Motivation of Kernel Method

In classifications, it is often the case that we want to obtain a non-linear decision boundary. 

<img title="figure 1: non-linear boundary" src="https://miro.medium.com/max/1144/1*C1tN-IxPjg6fwAHKkJthEw.png" alt="non-linear boundary" style="zoom:50%;" />

For example, for this problem (figure 2), we want a desicion boundary that is somewhat like a circle, however, our model only yields linear boundaries. In order to let our model to have more flexibility without changing the basic algorithms, we can apply a transformation onto the feature space $X$. like the figure on the right. In higher dimensionality, we can see that a linear seperation hyperplane serves as a non-linear decision boundary in original feature space. In this case, we performed a transfor somewhat like $(x-\mu)^2$ to make the center $X$ to be smaller while the further away from the center $\mu$, the larger the values are.

<img title="figure 2: Kernel example" src="https://miro.medium.com/max/1400/0*ngkO1BblQXnOTcmr.png" alt="kernel visualize" style="zoom:80%;" />

This idea is called feature map.

#### Feature Map

Consider a function $\phi(X): \mathbb{R}^d \to \mathbb{R}^p$ that transforms (or, projects) the features $X \in \mathbb{R}^d$ to a new feature space $X’ \in \mathbb{R}^p$. We call 

$$
\begin{align}
 \text{the function } \phi &\text{: feature map} \\
 X' &\text{: (new) features} \\
 X &\text{: attributes}
\end{align}
$$

e.g. for instance, the attributes contain 2 variables, $X = \{x_1, x_2\}$, the feature map wants to create a new feature space that includes **any combination of attributes within degree of 3**. Thus

$$
\phi(X) = [1, x_1, x_2, x_1^2, x_2^2, x_1x_2, x_1^3, x_2^3, x_1^2x_2, x_1x_2^2]
$$

Now $\phi(X)$ has 10 terms. However, this number could go up very rapidly as the **number of attributes** and the number of polynomials that we want to express goes up, which makes feature map a problem in computing. That is, **<u>after feature map, the feature space could have a tremendously large (or infinite) dimensionality, which causes problem for computing</u>**. That is where we introduce the kernel method.

## Kernel Method

#### Definition

Given a feature map function $\phi$, we define the kernel that corresponds to this feature mapping is as follows

$$
K(x, z) = <\phi(x), \phi(z)>
$$

where $<a, b> = a^Tb$, denotes dot product of two vectors. And $K: \mathbb{R}^{p} \times \mathbb{R}^{p} \to \mathbb{R}$

The trick is: 

1. We can design a kernel function instead of a feature transformation function, since feature map functions maybe hard to image in higher dimensionalities, knowing that a valid kernel maps to a feature map;
2. Using kernel, we can avoid computational problem with feature map;

Why these are true? Let’s take a closer look at each of them.

#### 1. What is a Valid Kernel

For functions $K$ from $\mathbb{R}^{p} \times \mathbb{R}^{p}$ to $\mathbb{R}$, valid kernel functions will be able to be “decomposed” into feature map’s dot product like above. However, it is not always convenient to check the feature map function since it could be very complex. The save way that we can design a kernel function knowing it is valid or not is by looking at the kernel matrix.

###### Kernel Matrix

Suppose there are $n$ samples $X = x^{(1)},\dots, x^{(n)} $, the **kernel matrix** of a kernel function $K(x, z)$ on this data set $X$ is defined as $K \in \mathbb{R}^{n \times n}$, where 

$$
K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T\phi(x^{(j)})
$$

the element in position $(i,j)$ is the kernel function’s result of $x^{(i)}$ and $x^{(j)}$. 

​    Observations:

1. The kernel matrix is symmetric, 
   
   $$
   K_{ij}= \phi(x^{(i)})^T\phi(x^{(j)}) = \phi(x^{(j)})^T\phi(x^{(i)}) = K_{ji}
   $$
   
   2. For any vector $z$, we have $z^TKz = \sum_k (\sum_iz_i \phi(x^{(i)}))^2 \geq 0$.

These lead to our theorem to judge whether a kernel is valid.

**Mercer’s Theorem**: Let $K:\mathbb{R}^{p} \times \mathbb{R}^{p} \to \mathbb{R}$ be given, Then for $k$ to be a valid (Mercer) kernel, iff for any $\\{x^{(1)}, \dots, x^{(n)}\\}, (m < \infty)$, the corresponding kernel matrix is **symmetric positiev semi-definite**.

#### 2. How to Avoid $\phi$ in Calculation

That is the topic of implementing kernel method in a algorithm.

## Implementation of Kernel Method

#### Version 1: With the Example of Least Mean Squares

For example whe implementing linear regression, where our model is 

$$
y = \theta^Tx + \epsilon
$$

Now we want to tranform our feauture $x$ to let the model has the ability to capture more trends. Now new features become $\phi(x)$. The model’s prediction is now

$$
\hat y = h_\theta(\phi(x)) = \theta^T\phi(x)
$$

###### Update Rule

If we are using gradient descent to update this model, the update rule is

$$
\theta := \theta + \alpha \frac{1}{n}\sum_{i=1}^n(y^{(i)} - \theta^T\phi(x^{(i)}))\phi(x^{(i)})
$$

**Claim**: (proof is omitted here) we can find a set of parameters $\beta_i$ where $i = 1,2,\dots,n$, such that

$$
\theta = \sum_{i=1}^n \beta_i\phi(x^{(i)})
$$

then the update rule can be expressed as

$$
\begin{align}
 \theta &:= \sum_{i=1}^n \beta_i \phi(x^{(i)}) + \alpha \frac{1}{n}\sum_{i=1}^n(y^{(i)} - \theta^T\phi(x^{(i)}))\phi(x^{(i)}) \\\\
 &= \sum_{i=1}^n ( \beta_i + \alpha \frac{1}{n} \sum_{i=1}^n(y^{(i)} - \theta^T\phi(x^{(i)})))\phi(x^{(i)}) \\\\
 &= \sum_{i=1}^n \beta_{i(new)}\phi (x^{(i)})
\end{align}
$$

If we let every $\beta_i$ to be updated **just like** $\theta$ in the following way: 

$$
\begin{align}
 \beta_i &:= \beta_i + \alpha \frac{1}{n} (y^{(i)} - \theta^T\phi(x^{(i)})) \\\\
 &= \beta_i + \alpha \frac{1}{n} (y^{(i)} - \sum_{j=1}^n \beta_j\phi(x^{(j)})^T\phi(x^{(i)})) \\\\
 &= \beta_i + \alpha \frac{1}{n} (y^{(i)} - \sum_{j=1}^n \beta_jK(x^{(j)}, x^{(i)}))
\end{align}
$$

where $\theta = \sum_{j=1}^n \beta_j\phi(x^{(j)})$ by our assumption; $K(x, z)$ is the kernel function for $\phi$; $i = 1, 2, \dots, n$. we can see that $\theta$ is completly replaced by calculation of kernel function $K$ in this case. 

**NOTE**: The updates of $\theta$ and $\beta$ are “synchronized”. Each iteration of $\theta$ updated is equivalent to $\beta$ updated once. This is more clear if we write the update rule of $\beta$ in vecotrized way.

Given $\beta \in \mathbb{R}^n$, $K^{(i)} = [K(x^{(i)}, x^{(1)}), K(x^{(i)}, x^{(2)}), \dots, K(x^{(i)}, x^{(n)})]^T$, so that $K = [K^{(1)}, K^{(1)}, \dots, K^{(n)}]$, we have

$$
\beta := \alpha \frac{1}{n}(y - \beta^TK)
$$

is equivalent to 

$$
\theta := \theta + \alpha \frac{1}{n}\sum_{i=1}^n(y^{(i)} - \theta^T\phi(x^{(i)}))\phi(x^{(i)})
$$

where $\theta = \sum_{j=1}^n \beta_j\phi(x^{(j)})$.

###### Prediction

Using $\beta$  and $K$ instead of $\alpha$ and $\phi$ to perform prediction. Assume here is a new $x$:

$$
\begin{align}
 \hat y &= h_\theta(\phi(x)) = \theta^T\phi(x) \\\\
 &= \sum_{j=1}^n \beta_j\phi(x^{(j)})\phi(x) \\\\
 &= \sum_{j=1}^n \beta_jK(x^{(i)}, x) \\\\
 &= h_{\beta}(K, x)
\end{align}
$$

#### Version 2: More General

For example in linear regression, or logistic regression, or support vector machine, we defined different loss function to minimize (respectively, mean square error, logit loss, hinge loss). For these cases, we can express the loss function as the following form:

$$
L(y, \theta^Tx)
$$

where $\theta^Tx$ yields the prediction of the model. In all of above cases, $\theta^Tx$ can be replaced by a function $f(x)$ which denotes the prediction of model for input $x$. 

The regularized cost function can be written as:

$$
J_\lambda(\theta) = \frac{1}{n} \sum_{i=1}^{n}L(y^{(i)}, f(x^{(i)})) + \frac{\lambda}{2}||f||_2^2
$$

where $f(x) = g(\theta^Tx)$ for the examples above.

###### The Representer Theorem

Consider the optimization problem

$$
\min_{f} D(f(x_1), \dots, f(x_n)) + P(||f||^2_2)
$$

where $P$ is nondecreasing function and $D$ depends on $f$ only though $f(x_i)$. It has a minimizer of the form 

$$
f(z) = \sum_{i=1}^{n} \beta_i K(z, x_i)
$$

where $\beta \in \mathbb{R}$ and $K$ is a kernel function. And if $P$ is strictly increasing, then every solution of this optimization problem will have this form.

Proof of this theorem can be found [here](http://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/13_kernel_methods.pdf) or [here](http://cs229.stanford.edu/extra-notes/representer-function.pdf).

###### Prediction

From the last equation, we can see that this theorem enables us to **make predictions** using $\beta$ and $K$ other than $\theta$ or $f$.

###### Update Rule

We need a new update rule to update $\beta$ directly, instead of calculating $\theta$. Let $f(x_i) = K^{(i)}\beta$ where $K^{(i)}$ is the same definition as in Version 1. The consise form of the cost function is

$$
J_P(\beta) = D(K^{(1)}\beta, \dots, K^{(n)}\beta) + P(\beta^TK\beta)
$$

To get the update rule for $\beta$, we compute the gradient for $\beta$. For all of our 3 examples above, $D$ is an averaging function of the loss functions and $P$ is usually take the form of l2 regularization, $P(r) = \frac{\lambda}{2}r$. Then this gradient is

$$
\begin{align}
 \nabla_\beta J_P(\beta) &= \nabla_\beta(\frac{1}{n} \sum_{i=1}^{n} L(K^{(i)^T} \beta, y) + \frac{\lambda}{2}\beta^TK\beta) \\\\
 &= \frac{1}{n} \sum_{i=1}^{n} L'(K^{(i)^T} \beta, y)K^{(i)} + \lambda K\beta
\end{align}

$$

This is how we update $\beta$.

## References

1. [Harish Kandan: kernel method](https://towardsdatascience.com/understanding-the-kernel-trick-e0bc6112ef78)
2. [Tejumade Afonja: kernel method](https://towardsdatascience.com/kernel-function-6f1d2be6091)
3. [CS229 Supplemental Notes: representer theorem](http://cs229.stanford.edu/extra-notes/representer-function.pdf)
4. [UMICH EECS 598: kernel methods](http://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/13_kernel_methods.pdf)
5. [Berkeley CS281B: Representer theorem and kernel examples](https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/8.pdf)
