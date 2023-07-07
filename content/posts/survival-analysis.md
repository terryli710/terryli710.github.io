---
title: Survival Analysis -- the Basics
date: 2020-06-14 00:50:17
tags: [stats, algorithm, survival analysis]
categories: NOTE
description: The basics of survival analysis, why do we need it and how does it work

---

## What is Survival Analysis and When to Use It?

Survival analysis can be generalized as <u>time-to-event analysis</u>, which is **analyzing the time to certain event (e.g. disease, death, or some positive events, etc.)**. Survival analysis gives us the estimation of, for example:

1. the time it takes before some events occur;
2. the probability of having experienced certain event at certain time point;
3. or which factors have what effect on the time to certain event.

*Case 1*: Here is a group of patients with lung cancer, the study[^1] recorded **whether the patient’s death was observed** within the range of the study, **the time to event**, and other information about this group. 

```R
> library(survival)
> data(lung)
> head(lung)
  inst time status age sex ph.ecog ph.karno pat.karno meal.cal wt.loss
1    3  306      2  74   1       1       90       100     1175      NA
2    3  455      2  68   1       0       90        90     1225      15
3    3 1010      1  56   1       0       90        90       NA      15
4    5  210      2  57   1       1       90        60     1150      11
5    1  883      2  60   1       0      100        90       NA       0
6   12 1022      1  74   1       1       50        80      513       0
```

Questions may include:

1. What is the probability of survival for this group after time $t$?
2. Given a new patient with same information known, can we predict the probability that this new patient will survival at time $t$?
3. Is there statistical differences of survival between male patients and female patients?
4. Within those information that we know about the patients, which ones have influence on the probabilities of survival?

All of those questions can be answered by survival analysis, using the following methods that will be covered in this article:

1. Kaplan-Meier Estimator
2. Log-Rank Test
3. Cox Proportional Hazards Model

These methods are by far the most commonly used techniques for answering these questions in medical literature[^2].

## Survival Data and Censoring

### What is New about Survival Data?

Look back to the data example we have above, if we want to estimate the time, which are numeric values for each patient, can we just use linear regression?

The answer is **NO** and that is because **<u>there are some patients who didn’t died even until the end of the study</u>** (or we simply don’t have that information recorded). Thus, some patient don’t have the “time to event”. How to solve this question? Here are some tries:

1. Manually set or impute that data. 
   1. *"For those who don’t have time-to-death, we simply impute them as the mean or median of the data we know."* In fact, for those patients who “outlived” the study, they typically won’t be represented by the notion of “average”. They were much better than “average” (in term of survival).
   2. *“Since no event occurred , we set the time to event to be $\infty$.”* That makes more sense than $a$, but that causes a big problem for linear regression.
2. Exclude those patients in our study. That won’t work either. That is because not only that part of patients contain information that is needed by the model to make better predictions, but also when a new patient with mild symptoms coming in, we want to have the ability to predict that is patient won’t die, rather than another “time to event”.

### Censoring

**The subjects whose events were not observed**, these observations are called censored. There are many types of censoring. Let’s take a look this figure.

<img src="https://jtd.amegroups.com/article/viewFile/22276/html/178542" alt="censoring[^3]" style="zoom:67%;" />*Fig 1. Examples of Censoring[^3]*

1. Right-censoring: subject 3 and 4 are right-censoring. It means that either follow-up information was lack to know the time for outcome (subject 4), or the outcome happened after study ended (subject 3).
2. Left-censoring: subject 6 is left-censoring. Exact time for entry or exposure happened before study began, which is unknown. What is known is the time that study began.

## Kaplan-Meier Estimator

It is also called “product-limit method”, is a non-parametric method used to estimate the probability of survival past given time points. Specifically, the function $S(t) = \text{probability of survival at time } t$. 

### Assumptions

There are several assumptions for this analysis to be valid (pretty much all the methods that we will talked about rely on these assumptions more or less). Here we will mention two most important assumptions[^4]:

1. Left-censoring should be minimized or avoided. Left-censoring makes the survival time not accurate and does not reflect observed survival time.
2. Independence of censoring and the event. This assumption is saying that when subjects are marked “censored”, it is not because they are at greater risk of the event. Legitimate censoring includes natural dropout or withdrawal (e.g. inpatient discharged) or event not occurred during study interval. Kaplan-Meier method assumes that **censored data behaves in the same way as uncensored data (after the censoring)**.

### The Model

$S(t)$ estimates the survival rate at time $t$: $P_s(t)$, and is called survivor function.
$$
P_s(t) = \frac{\text{number of subjects surviving}}{\text{number of subjects in the study}}
$$
And
$$
S(t) = \prod_{t_i < t}(1 - \frac{d_i}{n_i})
$$
where $t_j$ is a small time window that we study, and $t_j < t$, this window is within the following time of the study. $d_j$ is the subjects that have the outcome at $t_j$. And $n_j$ is the subjects that had no outcome before $t_j$, which also includes $d_j$. 

With K-M estimator, we can plot a curve with x-axis = time and y-axis = survival probability. This curve is called Kaplan-Meier curve or survival curve.

### Kaplan-Meier Curve

Here is an example[^5] of Kaplan-Meier curve. Patients are divided into two groups, and survival curve for two groups are plotted on the same figure. Each point on the line represent a value pair of (time after entry, survival rate of the group).

<img src="https://www.students4bestevidence.net/app/uploads/2016/04/Figure-2-Loai-JPEG.jpg" alt="km_curve" style="zoom:67%;" />

As soon as we can understand this figure, question arises as “Can we statistically compare these two group and say 'treatment' group is better than ‘placebo’ ?”. That when we move to the next common method, log-rank test. 

## Log-rank Test

Log-rank test is one of the most widely used [Pearson’s chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test). But there are some differences. Let’s take a look. Note that for simplicity, we are discussing two groups, while more than two groups cases are also valid using this test.
$$
H_0 : h_1(t) = h_2(t) \\
H_1 : h_1(t) \neq h_2(t)
$$

*Null hypothesis: two groups share an identical hazard function.*

The test statistics is defined as:
$$
Z = \frac{\sum_{t=1}^{T}(O_{i,t} - E_{i, t})}{\sqrt{\sum_{t=1}^{T}V_{i,t}}} \to^d \mathcal{N}(0,1)
$$
where
$$
\begin{align}
    N_{i,t} &= \text{observed subjects at time point }t \\\\
    O_{i,t} &= \text{observed outcomes at time point }t \\\\
    E_{i,t} &= \text{expected number of }O_{i,t}\text{ at }t \\\\
    V_{i,t} &= \text{varaince of }O_{i,t}\text{ at }t
\end{align}
$$
Assuming for each group $i = 1,2$, $O_{i,t}$ follows a [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution) with mean and variance (under null hypothesis, they have the same risk function, which is estimated by averaging two groups):
$$
\begin{align}
    E_{i,t} &= N_{i,t}\frac{O_t}{N_t} \\\\
    V_{i,t} &= E_{i, t}(\frac{N_t - O_t}{N_t})(\frac{N_t - N_{i,t}}{N_t - 1})
\end{align}
$$
Using the test statistics $Z$, we can have the *p-value* using quantiles of Gaussian distribution. Here are some further notes about this test.

*NOTES*[^6]: 

- Symmetric in two groups
- Only rank matters
- At each time point, we treat as independent
- If at certain time point, for one group, $N_{i,t} = 0$, then that time point won’t contribute to the test. 

Now, what if what we want to know is not the influence from categorical data like “treatment vs. placebo” but rather continuous data like age?

## Cox Proportional Hazards Regression

As said before, when the question comes to test the relationship between survival time and quantitative predictors or several factors at the same time, KM curves is limited and a new method is needed.

### Hazard Function

Let $T$ be the survival time. $f(T)$ is the probability density function of time to survival. And the *survivor function* $S(T)$ is the probability that an individual survives past $T$. We have:
$$
F(T) = \text{Pr}(t < T) = \int_0^Tf(t)dt
$$
And
$$
S(T) = \text{Pr}(T \geq t) = 1 - F(T)
$$
The *hazard function* is the probability that a subject experiences the event of interest at time $T$, given that the individual has survived until the start of time $T$.
$$
h(T) = \frac{f(T)}{S(T)}
$$

### The Model

We can show:
$$
\frac{d}{dT}[- \log S(T)] =\frac{f(T)}{S(T)} = h(T)
$$
Cox regression assumes the relationship between hazard function and variables is:
$$
\log[\frac{h(T|x)}{h_0(T)}] = \beta^T x
$$
where $x$ we called covariates;

​            $h(T|x)$ is the hazard function at time $T$ given covariates $x$;

​            $\beta^T x = \sum_{i = 1}^p x_i \beta_i$ , $\beta$s are the parameters, note that this function is not a function of time $T$ but just covariates;

​            $h_0(T)$ is called ***baseline hazard function***, and this function is a function of time $T$ but not covariates.

### Solve the Model

This part is left empty for now, may or may not be updated.

## The End

In the next article, a case study of survival analysis using R will be shown.

## References and Citations

[^1]: Loprinzi, C. L., Laurie, J. A., Wieand, H. S., Krook, J. E., Novotny, P. J., Kugler, J. W., ... & Klatt, N. E. (1994). Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. *Journal of Clinical Oncology*, *12*(3), 601-607.
[^2]: Schober, P., & Vetter, T. R. (2018). Survival analysis and interpretation of time-to-event data: the tortoise and the hare. *Anesthesia and analgesia*, *127*(3), 792.
[^3]: Brembilla, A., Olland, A., Puyraveau, M., Massard, G., Mauny, F., & Falcoz, P. E. (2018). Use of the Cox regression analysis in thoracic surgical research. *Journal of thoracic disease*, *10*(6), 3891.
[^4]: Statistics, L. (2013). Kaplan-Meier using SPSS statistics.

[^5]: [Tutorial about Hazard Ratios](https://www.students4bestevidence.net/blog/2016/04/05/tutorial-hazard-ratios/) by Loai Albarqouni
[^6]: Survival Analysis: Logrank Test

[^7]: [NCSS Statistical Software](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Cox_Regression.pdf)