---
title: Further Discussion of Relative Importance
date: 2020-08-10 17:20:30
tags: [stats, algorithm, linear regression]
categories: NOTE
description: Discussed two more methods of estimating relative importance of linear regression features. Commonality Analysis and Dominance analysis.

---

<!-- more -->

In this article, two more methods will be discussed that takes not only linear correlation of a single predictor variable with the dependent variable, but also considers the intertwined effects.

These two methods are Commonality Analysis (CA) and Dominance Analysis (DA). They share some similarities yet differs in focuses of analyzing relative importance.

### Commonality Analysis

#### Idea

[Commonality Analysis](https://en.wikipedia.org/wiki/Commonality_analysis#:~:text=Commonality%20analysis%20is%20a%20statistical,regression%20model%20into%20commonality%20coefficients.) is a statistical technique within [multiple linear regression](https://en.wikipedia.org/wiki/Linear_regression) that decomposes a model's [*R*2 statistic](https://en.wikipedia.org/wiki/Coefficient_of_determination) (i.e., explained variance) by all [independent variables](https://en.wikipedia.org/wiki/Independent_variable) on a [dependent variable](https://en.wikipedia.org/wiki/Dependent_variable) in a multiple linear regression model into commonality coefficients. 

To illustrate this idea, we can image a model with three independent variables $$X_1, X_2, X_3$$. The idea of commonality analysis assumes that **each variable** (eg. $$X_i$$), **as well as each combinations of variables** (eg. $$X_i X_j$$), including combinations of all variables adding together, **have some effect on the $$R^2$$ of the full model**.
$$
 Y \sim X_1 + X_2 + X_3
$$
So commonality analysis is trying to decompose the $R^2$ of the full model to each combinations of variables by computing linear models for **all possible subsets regression**.

#### How to Interpret CA

Suppose we used commonality analysis on the `iris` dataset with the model
$$
\text{Sepal.Length} \sim \text{Sepal.Width} + \text{Petal.Length} + \text{Petal.Width}
$$
Here is the typical table that we will get as result.

| Variable Names                       | Coefficient | $$\%$$ Total |
| ------------------------------------ | ----------- | ------------ |
| Sepal.width                          | 0.09235042  | 0.10755784   |
| Petal.Length                         | 0.15137464  | 0.17630163   |
| Petal.Width                          | 0.01843388  | 0.02146941   |
| Sepal.Width,Petal.Length             | -0.05414103 | -0.06305648  |
| Sepal.Width,Petal.Width              | -0.01212723 | -0.01412423  |
| Petal.Length,Petal.Width             | 0.67498054  | 0.78613012   |
| Sepal.Width,Petal.Length,Petal.Width | -0.01225950 | -0.01427829  |

Explanation of each column:

- `Variable Names`: names of all combinations of variables, all of them will have a row in the table;
- `Coefficient`: calculated $R^2$ associated with the specific combinations of variables by CA algorithm, this serves as the 'relative importance';
- `% Total`: contribution to linear model as percent.

The rows with only one variable denote effects that are "**unique**", while other rows with multiple variables together are "**common**" effects. Coefficients are calculated $$R^2$$ increases of associated variable or variable combinations. Common effects and unique effects can be compared directly and more positive the coefficient, the more important they are. *Note that common effects are not simply the $$R^2$$ with more variables, so it would make sense to compare rows with different amount of variables in them.* And unique effects of each variable also denote the importance as well as the **irreplaceability** of individual variables.

For example in our case, `Petal.Length,Petal.Width` is the most significant contributor to the model, which contributed 78.6% of the $R^2$ of the final model. And among unique effects, `Petal.Length` is the most important one, while its effect will be greatly deteriorated without `Petal.Width`. 

*Side note: With `Petal.Length,Petal.Width` being the most important feature, one can easily think of that `Sepal.Length` is somehow related to the **area of Petal**. But actually, the area of Petal cannot be computed in linear model using `Petal.Length,Petal.Width`. Linear models only computes the linear combinations of these two variables, while area should be something like `variable` $$=$$ `Petal.Length` $$\times$$ `Petal.Width`.*

#### How to Perform CA

##### Theoretically

###### STEP 1: All Possible Subsets Regression

The first steps of performing CA involves performing linear regressions for all possible subsets of independent variables. $$R^2$$s are recorded.

For example for `iris` dataset that we just mentioned. All Possible Subsets Regression is performed.

| Variable Names                       | K   | $$R^2$$    |
| ------------------------------------ | --- | ---------- |
| Sepal.width                          | 1   | 0.01382265 |
| Petal.Length                         | 1   | 0.75995465 |
| Petal.Width                          | 1   | 0.66902769 |
| Sepal.Width,Petal.Length             | 2   | 0.84017784 |
| Sepal.Width,Petal.Width              | 2   | 0.70723708 |
| Petal.Length,Petal.Width             | 2   | 0.76626130 |
| Sepal.Width,Petal.Length,Petal.Width | 3   | 0.85861172 |

Each row denotes a regression model with the features in this model recorded in the first column. `K` is the number of features in the models. The $$R^2$$ for each regression model is recorded in the third column.

###### STEP 2: Commonality Weights

To compute the commonality effect, we need commonality weights, which is a transform matrix from APS $$R^2$$ to commonality effects. Before explain the nature of commonality weight matrix, here we show two examples of feature number $$n=2$$ and $$n = 3$$ respectively.

| Contributors | $$R^2_{y\cdot i}$$ | $$R^2_{y \cdot j}$$ | $$R^2_{y \cdot ij}$$ |
| ------------ | ------------------ | ------------------- | -------------------- |
| $$U(i)$$     | 0                  | -1                  | 1                    |
| $$U(j)$$     | -1                 | 0                   | 1                    |
| $$C(i, j)$$  | 1                  | 1                   | -1                   |

| Contributors  | $$R^2_{y\cdot i}$$ | $$R^2_{y\cdot j}$$ | $$R^2_{y\cdot k}$$ | $$R^2_{y\cdot ij}$$ | $$R^2_{y\cdot ik}$$ | $$R^2_{y\cdot jk}$$ | $$R^2_{y\cdot ijk}$$ |
| ------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------- | ------------------- | -------------------- |
| $$U(i)$$      | 0                  | 0                  | 0                  | 0                   | 0                   | -1                  | 1                    |
| $$U(j)$$      | 0                  | 0                  | 0                  | 0                   | -1                  | 0                   | 1                    |
| $$U(k)$$      | 0                  | 0                  | 0                  | -1                  | 0                   | 0                   | 1                    |
| $$C(i,j)$$    | 0                  | 0                  | -1                 | 0                   | 1                   | 1                   | -1                   |
| $$C(i,k)$$    | 0                  | -1                 | 0                  | 1                   | 0                   | 1                   | -1                   |
| $$C(j,k)$$    | -1                 | 0                  | 0                  | 1                   | 1                   | 0                   | -1                   |
| $$C(i,j, k)$$ | 1                  | 1                  | 1                  | -1                  | -1                  | -1                  | 1                    |

Each row is a commonality effect to compute while each column is a regression $$R^2$$ that may be used to compute the effect. For example in the first row of $$n=2$$ table, the formula to compute unique effect of variable $$i$$ is :
$$
U(i) = (-1) \times R^2_{y \cdot j} + 1 \times R^2_{y \cdot ij}
$$

###### Nature of Commonality Weights

Thinking of the computation of common and unique effects as a computation in overlapping sets illustrated by the Venn diagram below (variable 1 $$= i$$, variable 2 $$= j$$, variable 3 $$= k$$). The $$R^2$$s of regression models (columns of the table) are complete circular area or combination of **circular area**, think of them as **unions**. For example:
$$
\begin{align}
    R^2_{y \cdot i} &= \text{Unique to var i} + \text{Common to var i&j} + \text{Common to var i&j} + \text{Common to var i&j&k} \\\\
    R^2_{y \cdot ij} &= \text{Unique to var i} + \text{Common to var i&j} + \text{Common to var i&j} + \text{Common to var i&j&k} + \text{Common to var j&k}
\end{align}
$$
![Illustrating Commonality Analysis. | Download Scientific Diagram](https://www.researchgate.net/profile/Robert_Capraro/publication/238605329/figure/fig1/AS:669410532548621@1536611316685/Illustrating-Commonality-Analysis.png)

And unique effects or common effects are the **small pieces in the diagram** as labeled, think of them as **complements** or **intersections**. 

The commonality weights serves as a transfer matrix for these two groups of values that have meanings in set theory.

##### In Practice

`R` provides a package for Commonality Analysis: [`yhat`](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.9485&rep=rep1&type=pdf) , documentation of the package by CRAN is [here](https://cran.r-project.org/web/packages/yhat/yhat.pdf) and by RDocumentation is [here](https://www.rdocumentation.org/packages/yhat/versions/2.0-2). (But there is no package in Python that provides these functions)

Core functions for commonality analysis are [`aps`](https://www.rdocumentation.org/packages/yhat/versions/2.0-2/topics/aps), which performs all possible subset regression; and [`commonality`](https://www.rdocumentation.org/packages/yhat/versions/2.0-2/topics/commonality), which uses commonality weights to do the transformation. 

[`commonalityCoefficients`](https://www.rdocumentation.org/packages/yhat/versions/2.0-2/topics/commonalityCoefficients) function is like an end-to-end function that outputs the final results of CA from input variables. Here is an very straight forward example of performing CA using `R` with `yhat` package with `iris` dataset.

```R
> library(yhat)
> library(datasets)
> data(iris)
> # all possible subset regression and results
> all_subset_regression <- aps(iris, "Sepal.Length", list("Sepal.Width", "Petal.Length", "Petal.Width"))
> all_subset_regression
$ivID
             [,1]
Sepal.Width     1
Petal.Length    2
Petal.Width     4

$PredBitMap
             [,1] [,2] [,3] [,4] [,5] [,6] [,7]
Sepal.Width     1    0    1    0    1    0    1
Petal.Length    0    1    1    0    0    1    1
Petal.Width     0    0    0    1    1    1    1

$apsBitMap
                                     [,1]
Sepal.Width                             1
Petal.Length                            2
Petal.Width                             4
Sepal.Width,Petal.Length                3
Sepal.Width,Petal.Width                 5
Petal.Length,Petal.Width                6
Sepal.Width,Petal.Length,Petal.Width    7

$APSMatrix
                                     k         R2
Sepal.Width                          1 0.01382265
Petal.Length                         1 0.75995465
Petal.Width                          1 0.66902769
Sepal.Width,Petal.Length             2 0.84017784
Sepal.Width,Petal.Width              2 0.70723708
Petal.Length,Petal.Width             2 0.76626130
Sepal.Width,Petal.Length,Petal.Width 3 0.85861172

> # commonality calculation
> commonality(all_subset_regression)
                                     Coefficient     % Total
Sepal.Width                           0.09235042  0.10755784
Petal.Length                          0.15137464  0.17630163
Petal.Width                           0.01843388  0.02146941
Sepal.Width,Petal.Length             -0.05414103 -0.06305648
Sepal.Width,Petal.Width              -0.01212723 -0.01412423
Petal.Length,Petal.Width              0.67498054  0.78613012
Sepal.Width,Petal.Length,Petal.Width -0.01225950 -0.01427829

> # or, just using commonalityCoefficients
> commonalityCoefficients(iris, "Sepal.Length", list("Sepal.Width", "Petal.Length", "Petal.Width"))
$CC
                                                     Coefficient     % Total
Unique to Sepal.Width                                     0.0924       10.76
Unique to Petal.Length                                    0.1514       17.63
Unique to Petal.Width                                     0.0184        2.15
Common to Sepal.Width, and Petal.Length                  -0.0541       -6.31
Common to Sepal.Width, and Petal.Width                   -0.0121       -1.41
Common to Petal.Length, and Petal.Width                   0.6750       78.61
Common to Sepal.Width, Petal.Length, and Petal.Width     -0.0123       -1.43
Total                                                     0.8586      100.00

$CCTotalbyVar
             Unique  Common  Total
Sepal.Width  0.0924 -0.0786 0.0138
Petal.Length 0.1514  0.6086 0.7600
Petal.Width  0.0184  0.6506 0.6690
```

###### A Mathematical Algorithm

Computation of commonality weights, or the process of constructing the commonality weight matrix, is not a very straight forward process. Though there are many approaches to do so, here, I will provide my thinking of doing it. This approach is different from `yhat` package, and was the way that I came up with when implementing CA in Python.

To simplify the symbol representations, here we let 

- $$U(X)$$ denotes the unions of effects, thus, are calculated from all possible subset regression(APS), and is considered known in computing commonality effects. eg. $$U(i, j) = R^2_{y \cdot ij}$$, is the r squared of regression containing variable $$i$$ and $$j$$.
- $$I(X)$$ denotes the intersection or complement of sets. When $$X$$ contains only one set, it's complement of that set. eg. $$I(i) = i - \cup j, k, l ,\dots$$. When $$X$$ contains more than one set, it means intersection. eg. $$I(i, j, k) = i \cap j \cap k$$. These values are intermediate values that neither do we know nor do we want as results (except for the complements).
- $$C(X)$$ denotes the exclusive complements of sets, which fits the definition of commonality effects, thus are the goals of the algorithm. eg. $$C(i) = i - \cup j,k,l, \dots = I(i)$$ can be represented as "unique effect of $$i$$" in the Venn diagram; $$C(i, j) = i \cap j - \cup k , l, \dots$$ is represented as "unique effect of $$i$$ and $$j$$".

**ALGORITHM 1: Calculating Commonality Effect**

1. Calculate $$I(X)$$ from $$U(X)$$, $$X$$ is any combinations of input variables.
   1. For order in 1 to num_variables: 
      - Calculate $$I(X)$$ for X from combination of order, using inclusion-exclusion principle
      - eg. $$I(i, j) = i \cap j =  U(i) + U(j) - U(i, j)$$
2. Calculate $$C(X)$$ from $$I(X)$$ and $$U(X)$$.
   1. For order in 1 to num_variables:
      - Calculate $$C(X)$$ for X from combination of order, using inclusion-exclusion principle
      - eg. $$C(i, j) = I(i, j) - I(i, j, k) - I(i,j,l) - \dots + \dots$$

### Dominance Analysis

#### Idea

The idea behind dominance analysis is reasonably well stated in the quote below. Budescu (1993) set out three conditions for determining the relative importance
of predictors in a regression equation:

> (a) Importance should be defined in terms of a variable’s “reduction of error” in predicting the criterion, y; 
> 
> (b) The method should allow for direct comparison of relative importance instead of relying on inferred measures; 
> 
> (c) Importance should reflect a variable’s direct effect (i.e. when considered by itself), total effect (i.e., conditional on all other predictors), and partial effect (i.e. conditional on subsets of predictors).

> Dominance analysis compares pairs of predictors and how they behave in $$h = 2^{(p-2)}$$ models. These models involve all subsets of predictors. Variable $$a$$ dominates variable $$b$$ in model $$h$$ if adding variable a to model $$h$$ results in a greater $$R^2$$ than adding variable $$b$$ to model $$h$$. By performing a dominance analysis for all pairs of the $$p$$ predictors, dominance analysis determines an order of importance for the predictors, if that order exists.

#### How to interpret DA

Dominance analysis provides the "$$R^2$$ increased by adding a variable". Average of the effect of adding the variable in each possible order and in any number of variables of models. 

DA does not meant to rank the variables, but provides comparisons between pairs of variables. There are kinds of relationship between variables, as shown in the figure below. Descriptions can be found [here](https://github.com/dominance-analysis/dominance-analysis).

![img](https://github.com/dominance-analysis/dominance-analysis/raw/master/images/Dom%20Stat.jpg)

Some notes about these relationships

1. Individual Dominance: like method `first`.
2. Interactional Dominance: like method `last`.
3. Total Dominance: Average of all the conditional values.

#### How to Perform DA

This question is much simpler than CA, since they share the first steps and the second step of DA is much more straight forward.

##### Theoretically

STEP 1: All Possible Subset Regression

STEP 2: Form the DA table.

##### In Practice

There is an python package to perform DA, `dominance-analysis`, with the source code [here](https://github.com/dominance-analysis/dominance-analysis).

## References

1. [Multiple Regression in Industrial Organizational Psychology: Relative Importance and Model Sensitivity](https://conservancy.umn.edu/handle/11299/213096)
2. [Commonality analysis -- Wikipedia](https://en.wikipedia.org/wiki/Commonality_analysis#:~:text=Commonality%20analysis%20is%20a%20statistical,regression%20model%20into%20commonality%20coefficients.)
3. [Which types of income matter most for well‐being in China: Absolute, relative or income aspirations?](https://onlinelibrary.wiley.com/doi/full/10.1002/ijop.12284)