---
title: SMOTE with Python
date: 2020-04-11 20:57:25
tags: [machine learning, python, over sampling]
categories: NOTE
description: An over-sampling method, inspired by KNN, to deal with imbalanced data.
---
## Motivation
Working on classification problem, especially in medical data practice, we are often faced with problem that there are imbalanced cases in each categories. While it might be OK for other machine learning model builders to overlook this, it is essential that we pay attention to imbalanced data problem in **medical applications**, because in most scenarios, prediction and accuracy of the minority categories is far more important than the most common classes. e.g. predict positive for rare disease, detect abnormal in CT scans.

## SMOTE

### How SMOTE solve imbalanced problem
Synthetic Minority Oversampling Technique (SMOTE) works by **over-sampling the minority class**, selecting example that are close in the feature space, drawing a line between the examples in the feature space, new samples are synthesized on the lines. If the feature space is 2 dimension, we can visualize the lines connecting real instances and synthesized instances.
>… SMOTE first selects a minority class instance a at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b.

\- [Imbalanced Learning: Foundations, Algorithms, and Applications](https://www3.nd.edu/~dial/publications/hoens2013imbalanced.pdf) p.47

###  Python Implementation

 - Package: `imbalance-learn`
     - Install: `pip install imbalance-learn`
     - Documentation: [imbalance-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)

Basic usage
```py
# Set up SMOTE
oversample = SMOTE()  
# Transform the data set
XSmote, YSmote = oversample.fit_resample(X,Y)
```



### Suggestions

- First use random under-sampling to trim the number of examples in the majority class, then use SMOTE to over-sample the minority class to balance the class distribution.
>The combination of SMOTE and under-sampling performs better than plain under-sampling.
\- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)

- Watch out for biases towards minority class (false positives in medical setting) in high dimension feature space. Use feature selection to reduce dimensionality before SMOTE.
>... While in most cases SMOTE seems beneficial with low-dimensional data, it does not attenuate the bias towards the classification in the majority class for most classifiers when data are high-dimensional, and it is less effective than random under-sampling.
>... In practice, in the high-dimensional setting only k-NN classifiers based on the Euclidean distance seem to benefit substantially from the use of SMOTE, provided that variable selection is performed before using SMOTE; the benefit is larger if more neighbors are used. SMOTE for k-NN without variable selection should not be used, because it strongly biases the classification towards the minority class.
\-[SMOTE for high-dimensional class-imbalanced data](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106)



## Other References

 1. [ SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
 2. [SMOTE explained for noobs - Synthetic Minority Over-sampling TEchnique line by line](http://rikunert.com/SMOTE_explained)
 3. [Application of Synthetic Minority Over-sampling Technique (SMOTe) for Imbalanced Datasets](https://medium.com/towards-artificial-intelligence/application-of-synthetic-minority-over-sampling-technique-smote-for-imbalanced-data-sets-509ab55cfdaf)

---
