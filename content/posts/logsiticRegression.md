---
title: Logistic Regression Updated with Newton's Method
date: 2020-04-20 09:44:06
tags: [machine learning, logistic regression, algorithm, Newton's method, loss function]
categories: NOTE
description: Insight of logistic regression and Newton's method
---

Logistic regression is a very important binary classification algorithm, in this article, some essential details inside the algorithm will be discussed. Plain language will be used to discuss the most detail aspects so that beginners of machine learning can easily get the idea.

## Assumptions of Logistic Regression

Logistic regression does not require as many assumptions as linear regression. There are a few that are interested and we will shortly discussed about.

1. The error terms are independent to each other.
   - In the experiment design, each sample should be “equivalent” to each other, no paired/match samples or before/after experiment samples.
2. There is no high correlations between the features (multicollinearity).
   - The model might not converge in this case by MLE estimate
3. The log odds has linear relationship with the independent variables.
   - While we underlyingly assuming this, it would not hold true in most cases. There is no extra step to test this assumption. And if it fails, logistic regression just won’t work. So in practice, we try logistic regression to “test” this assumption.
4. The dependent variable follows a Bernoulli distribution.
   - By this we assume that $Y \sim \text{Bernoulli}(\phi)$, and automatically independent to each other.

## Insight of Logistic Regression

#### Constructing the Model

In practice, assuming we have our features $X \in \mathbb{R}^{n \times d}$, and we want to predict a binary response $Y_i \in \{0,1\}$ for each sample of features. What we do is creating model that outputs $p(y=1) = \phi = h_\theta(x)$. According to the assumption, the **log odds** 
$$
\text{logit}(\phi) = \log(\frac{\phi}{1-\phi})
$$
has a linear relationship with the features. We use a vector $\theta \in \mathbb{R}^{d}$ to denote this linear relationship:
$$
\log(\frac{\phi}{1-\phi}) = \theta^Tx
$$
So, we get the model’s predictions will be:
$$
h_\theta(x)= g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}
$$
where $g(z)$ is called **[sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)**, which is an important function in machine learning.
$$
g(z) = \frac{1}{1+e^{z}}
$$
![sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg "Shape of Sigmoid Function")



The model’s goal is trying to learn the parameters $\theta$.



## Loss function

#### Define Loss

To find the “best” $\theta$, we should first define what is “better”. The **loss function** of a model denotes how well the model fits the data (the better the model, the fewer the loss). Thus the process of finding the best $\theta$ is concretized to minimizing the loss function of a model. we define the loss function of logistic regression as follows:
$$
L(y, \hat y) = -(y^{(i)}\log(\hat y^{(i)}) + (1-y^{(i)})\log(1-\hat y^{(i)}))
$$
This function is binary form of [**cross entropy loss**](https://en.wikipedia.org/wiki/Cross_entropy), which is widely used in classification models.

#### Intuition of Loss function

Here provides an intuition of why this loss function is chosen. As we mentioned in assumptions we made, $y \sim \text{Bernoulli}(\phi)$. So the pdf of $y$ can be written as
$$
p(y|x) = \Phi^{y}(1 - \Phi)^{(1-y)}
$$
In our model, $\phi$ is estimated by $h_\theta(x)$. To make this probability model to resemble $y$ the most, it is a natural thing to maximize it’s likelihood. The following two step resembles the process of calculating MLE of $\phi$. We have the likelihood is:
$$
L(y^{(i)}|x^{(i)};\theta) = h_\theta(x)^{y}(1-h_\theta(x))^{(1-y)}
$$
The log-likelihood is:
$$
l(y^{(i)}|x^{(i)};\theta) = y^{(i)}\log(h_\theta(x)) + (1 - y^{(i)})\log(1-h_\theta(x))
$$
Noticing that log-likelihood is negative of loss function of the model. <u>Minimizing the loss function is equivalent to maximizing the likelihood of the parameter in Bernoulli distribution of $y$.</u> 

## Fit the Model

With the assumptions about the distribution of $y$ and the relationship between the response and the features, we constructed the model and defined the parameters that requires optimizing. We defined that the better the parameters, the larger the likelihood of $y$ comes from the distribution that we assumed. Now, in order to optimize $\theta$, instead of trying out every single combination of possible values of $\theta$, we need a parameter searching technique. 

#### Newton’s Method

Apart from gradient descent, we have a another technique that sometimes (this “sometimes” includes logistic regression) run much faster, called Newton’s method. Which is a method used for recursively looking for root of a function. Here is the idea of Newton’s method:

1. Start with an initial guess which is reasonably close to the true root $x_0$;
2. Then to approximate the function using the first derivative of this point $f'(x_0)$, and draw a tangent line;
3. And finally to compute the x-intercept of this tangent line. This x-intercept will typically be a better approximation to the original function's root than the first guess;
4. Iterated.

<img src="https://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif" alt="newton’s method gif" title="GIF Illustration of Newton's Method" style="zoom:80%;" />



<img src="https://openstax.org/resources/03a495b2b2b3d4dfa2b027fccdae44d1aba527a1" alt="newtons method png" title="Newton's Method: How to Get Next Guesses of Root of a Function" style="zoom: 80%;" />



The update rule for Newton’s method is as follows (not hard to get):
$$
x_{n+1} = x_{n} - \frac{f(x_n)}{f'(x_n)}
$$
For our problem, we want to maximize the likelihood function, if this function is convex (in fact it is), then our mission is equivalent to **find the root of the first derivative of log-likelihood**. Using Newton’s method to do this, our update rule is:
$$
\theta := \theta - \frac{l'(\theta)}{l''(\theta)}
$$
where, as $\theta$ is a vector instead of a real value in most practical cases. Using the linear algebra form of first derivatives looks like this:
$$
\nabla_\theta l = 
\begin{bmatrix}
    \frac{\partial}{\partial\theta_1}l(\theta) \\\\
    \frac{\partial}{\partial\theta_2}l(\theta) \\\\
    \vdots \\\\
    \frac{\partial}{\partial\theta_d}l(\theta) \\\\
\end{bmatrix}
$$
And second derivative is called Hessian matrix $\textbf{H} = \nabla^2 l(\theta) \in \mathbb{R}^{d \times d}$.
$$
\textbf{H}_{ij} = \frac{\partial^2l(\theta)}{\partial\theta_i\partial\theta_j}
$$


#### Trade-off between Newton’s Method and Gradient Descent

Though Newton’s method typically takes much shorter iterations of update to converge compared to gradient descent. It requires the calculation of first and the secondary derivative of the loss function, while gradient descent only requires the first derivative. So in each iteration, the calculation cost for Newton’s method is higher, especially when we cannot trace back the function and explicitly calculate the second derivative. Thus though we can implement Newton’s method here and achieve a faster convergence. This case might not be true for most of other algorithms, typically the more complicated ones. That is why gradient descent is still the most handy option for many other machine learning applications.

## References

1. [WIKIPEDIA](https://en.wikipedia.org/wiki/Newton%27s_method)
2. [OPENSTAX](https://openstax.org/books/calculus-volume-1/pages/4-9-newtons-method)

3. [STANFORD CS229](http://cs229.stanford.edu/notes/cs229-notes1.pdf)