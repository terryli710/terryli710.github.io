---
title: Survival Analysis Case Study Using R
date: 2020-06-16 00:06:20
tags: [R, case, survival analysis]
categories: CASE
description: Survival Analysis Case Study Using R
---

From this article, we will use R language to perform survival analysis to a data set, in order to demonstrate some syntax and show the procedural of survival analysis using R.

## The Data

A very classic data set is used in the purpose of demonstration. Here is a glance of the data set.

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

This data set is about “survival in patients with advanced lung cancer from the North Central Cancer Treatment Group. Performance scores rate how well the patient can perform usual daily activities.” using `help()` function to see the description of the data set.

### Description of the Data Set

>inst:	Institution code
>time:	Survival time in days
>status:	censoring status 1=censored, 2=dead
>age:	Age in years
>sex:	Male=1 Female=2
>ph.ecog:	ECOG performance score as rated by the physician. 0=asymptomatic, 1= symptomatic but completely ambulatory, 2= in bed <50% of the day, 3= in bed > 50% of the day but not bedbound, 4 = bedbound
>ph.karno:	Karnofsky performance score (bad=0-good=100) rated by physician
>pat.karno:	Karnofsky performance score as rated by patient
>meal.cal:	Calories consumed at meals
>wt.loss:	Weight loss in last six months

For more information about the data, here is a reference: *Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ. Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al. Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.*

Note that `time` and `status` are two indicators of “time to event”, while others are covariates in this data set. Among them `inst`, `sex`, `ph.ecog` and `ambulatory` are categorical data while `age`, `ph.karno`, `meal.cal` and `wt.loss` are quantitative data.

### Visualization of the Data Set

Before we started to do survival analysis, first thing is to do an overview of the data. Here, two methods are used. The first one is a heatmap from library `ComplexHeatmap`, which provided various tools for heatmaps, documentation is [here](https://jokergoo.github.io/ComplexHeatmap-reference/book/). And the second one is a correlation matrix, which provides linear relationships for all pairs of variables and distributions of variables, R has many packages that provide good visualization of correlation matrix, e.g. `corrplot`, `PerformanceAnalytics ` and what I used here, `GGally`. 

```R
### Heat Map ###
> library(ComplexHeatmap)
> lungm <- t(scale(data.matrix(lung)))
> ht = Heatmap(lungm, 
               name = "feature value", 
               column_title = "patients", 
               row_title = "features",
               show_column_names = FALSE)
> draw(ht)
### Correlation Matrix ###
> library(GGally)
> ggpairs(lung)
```

<img src="heatmap.png" alt="hm" title="Heatmap of the Data" style="zoom:100%;" />

<img src="cormatrix.png" alt="cm" title="Correlation Matrix of the Data" style="zoom:100%;" />

From the heatmap, `NA`s are marked with grey. We can see some missing values in `pat.karno` or `wt.loss` but not much, which is good. From correlation matrix, it is hard to see strong correlations between any two of the variables, and many of them are categorical data. The `time` is clearly not gaussian distributed. the `age` is skewed to older groups.

After getting some sense about the data, survival analysis can start.

## Kaplan-Meier Analysis and Log-rank Test

Suppose we want to first see the overall survival condition (basic KM curve) and then see is there any significant varied patterns of patients who have different `sex` (KM analysis curve & log-rank test).

```R
> library(survminer)
### KM Analysis ###
# basic curve
fit1 <- survfit(Surv(time, status)~1, data = lung)
ggsurvplot(fit1, data = lung, censor=T)
# KM analysis curve
fit2 <- survfit(Surv(time, status)~sex, data = lung)
ggsurvplot(fit2, data = lung, pval = T, pval.method = T, conf.int = T, censor=T)
```

<img src="basickm.png" alt="bm" title="Basic Kaplan-Meier Curve" style="zoom:100%;" />

<img src="km_curve.png" alt="kc" title="Kaplan-Meier Curves for Male and Female Patients" style="zoom:100%;" />

Here, `survminer` package is used to plot KM curves by `ggsurvplot()` function, here are some parameters that I found useful:

>  conf.int: logical value. If TRUE, plots confidence interval.
>
> pval: logical value, a numeric or a string. If logical and TRUE, the p-value is added on the plot. If numeric, than the computet p-value is substituted with the one passed with this parameter. If character, then the customized string appears on the plot. See examples - Example 3.
>
> pval.method: whether to add a text with the test name used for calculating the pvalue, that corresponds to survival curves' comparison - used only when `pval=TRUE`.

In the second plot, log-rank test is automatically performed and p-value is specified on the plot.

Now, we want to see other covariates’ influences to the survival time.

## Cox Regression

Cox proportional hazard model and its analysis is performed using `survival` as well as `survminer`. 

- `coxph()` is used to perform general Cox PH model, its documentation can be found [here](https://www.rdocumentation.org/packages/survival/versions/3.2-3/topics/coxph). 
- `cox.zph()` is used to test the proportional hazards assumption for `coxph()` with documentation [here](https://www.rdocumentation.org/packages/survival/versions/3.2-3/topics/cox.zph).
- `ggforest()`: in order to get a visualization of the influences of covariates, `ggforest()` from `survminer` is used to create a plot, [doc](https://www.rdocumentation.org/packages/survminer/versions/0.4.6/topics/ggforest).

```R
### Cox Regression ###
> library(survminer)
> # Cox PH Model
> cox <- coxph(Surv(time, status) ~ age + sex + ph.karno + pat.karno + meal.cal + wt.loss, data = lung)
> cox
Call:
coxph(formula = Surv(time, status) ~ age + sex + ph.karno + pat.karno + 
    meal.cal + wt.loss, data = lung)

                coef  exp(coef)   se(coef)      z      p
age        0.0090807  1.0091220  0.0117503  0.773 0.4396
sex       -0.4859823  0.6150927  0.1995728 -2.435 0.0149
ph.karno  -0.0023393  0.9976635  0.0079466 -0.294 0.7685
pat.karno -0.0193962  0.9807907  0.0077533 -2.502 0.0124
meal.cal   0.0000126  1.0000126  0.0002460  0.051 0.9592
wt.loss   -0.0080307  0.9920014  0.0073014 -1.100 0.2714

Likelihood ratio test=17.53  on 6 df, p=0.007508
n= 169, number of events= 122 
   (59 observations deleted due to missingness)
> # Test Proportional Hazards Assumption
> czph <- cox.zph(cox)
> czph
            chisq df      p
age        0.5502  1 0.4583
sex        1.4804  1 0.2237
ph.karno   7.9155  1 0.0049
pat.karno  3.8774  1 0.0489
meal.cal   5.1873  1 0.0228
wt.loss    0.0143  1 0.9050
GLOBAL    14.5493  6 0.0241
> # Visualize Results
> ggforest(cox, data = lung)
```

<img src="visual.png" alt="vs" title="Visualization of Parameters in Cox PH Model" style="zoom:100%;" />

