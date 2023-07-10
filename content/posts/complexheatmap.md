---
title: Complex Heatmap, flexible package for heatmap based on R
date: 2020-09-14 11:09:38
tags: [R, package]
category: NOTE
description: Example based instructions for using ComplexHeatmap for heatmaps
---

Heatmap is great tool for **visualizing two dimensional data magnitude using colors**. This package provides almost complete features for building a heatmap under multiple conditions. Parameters that are used to customize a heatmap could be extremely complicated. This article try to archive part of the features that have been useful in my recently work. Note down their usage and syntax.

## Data

The dataset used in this article comes from [project Tycho](https://www.tycho.pitt.edu/), and is about recorded cases of measles in the United States. The 2-D data that we try to visualize is **the count of cases of measles in each year, in each state**. 

This dataset is downloaded by the following way:

1. Go to https://www.tycho.pitt.edu/data/;
2. Search for "measles" in pre-compiled US datasets;
3. Click [download](https://www.tycho.pitt.edu/dataset/US.14189004/#) using "download data and readme" link as .zip file

Before drawing the heatmap, data preprocessing steps are as follows

```R
# loading packages
library(data.table)
library(dplyr)
library(tidyr)
# loading data
setwd("E:/OneDrive - Stanford/Code/R")
measles <- read.csv("measles.csv")
# processing data
columns <- c("ConditionName", "Admin1Name", "PeriodStartDate")
measles <- measles[columns]
measles$year <- as.numeric(substring(measles$PeriodStartDate,1,4))
measles <- measles%>%group_by(year, Admin1Name)%>%summarise(n=n())
measles <- measles%>%spread(key=year, value=n)
# filling NAs
measles[is.na(measles)] <- 0
# raw numeric data
measles_num_raw <- data.matrix(measles[-1])
# add state names
rownames(measles_num_raw) <- c(unlist(measles[1]))
# show sample data
head(measles_num_raw)

               1888 1889 1890 1891 1892 1893 1894 1895 1896 1897 1898 1899 1900
ALABAMA           0    0    0    0    0    0    0    0    1    0    0    1    2
ALASKA            0    0    0    0    0    0    0    0    0    0    0    0    0
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA           0    0    0    0    0    0    0    0    0    0    0    0    0
ARKANSAS          0    0    0    0    0    0    0    0    0    0    0    0    0
CALIFORNIA        0    3    7    6   14    1    6    2    1    5    1    3    6
               1901 1902 1903 1904 1905 1906 1907 1908 1909 1910 1911 1912 1913
ALABAMA           0    0    0    2    0    0    2    4    1   24   28   19    6
ALASKA            0    0    0    0    0    0    0    0    0    0    0    0    0
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA           0    0    0    0    0    0    0    0    0    0    0    0    0
ARKANSAS          0    0    0    3    0    0    1    0    0    2    0    1    4
CALIFORNIA       20   16   21   11    2   24  105  139  178  144  169  175  166
               1914 1915 1916 1917 1918 1919 1920 1921 1922 1923 1924 1925 1926
ALABAMA           3    2   35  124  137   92   61   77   40  165  153  154  155
ALASKA            1    0    0    0    0    0    0    0    0    0    0    0    0
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA           2    6    0    0    0    2    2    5    0    0    6   46   49
ARKANSAS         30   21   18   31   62   21   70   84   19   87  101  104  102
CALIFORNIA      281  246  203  389  452  236  407  333  222  547  146  155  156
               1927 1928 1929 1930 1931 1932 1933 1934 1935 1936 1937 1938 1939
ALABAMA         160  204  199  207  204  196  209  208  200  193  204  202  201
ALASKA            0    0    0    0    0    0    0    0    0    0    0    0    0
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA          16   42   36   96   89   75   56   50   42   47   40   48   47
ARKANSAS        101  150  142  137  150  136  156  125  127  111  144  145  147
CALIFORNIA      160  204  204  201  205  201  210  207  205  206  204  201  197
               1940 1941 1942 1943 1944 1945 1946 1947 1948 1949 1950 1951 1952
ALABAMA         204  197  152  150  154  141  101  107   62   49   49   49  145
ALASKA            0    0    0    0    0    0    0    0    0    0    0    0    0
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA          48   81   50   47   49   47   50   49   41   51   51   50  137
ARKANSAS        147  143   97   98  100   95   85   69   64   52   52   47   66
CALIFORNIA      203  199  202  200  206  202  176  181  116   52   52   51  264
               1953 1954 1955 1956 1957 1958 1959 1960 1961 1962 1963 1964 1965
ALABAMA         117   51   49   53   51   48   45   48   49   49   44   50   98
ALASKA            0   49   45   51   47   45   48   41   46   51   46   45   95
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA         109   52   49   51   51   48   48   46   51   52   48   50  104
ARKANSAS         66   51   48   52   42   41   39   43   39   37   35   33   84
CALIFORNIA      276   52   49   53   51   50   48   52   52   52   47   50  104
               1966 1967 1968 1969 1970 1971 1972 1973 1974 1975 1976 1977 1978
ALABAMA          98  143   60    7   81   95  109   46   53    3    0   51  104
ALASKA           91  115   53   10   62   57   79    3    1    0   24   51   73
AMERICAN SAMOA    0    0    0    0    0    0    0    0    0    0    0    0    0
ARIZONA         100  155   89   48   99   98  144   52   60   25   71   87  106
ARKANSAS         79  114   49    2   59   77  107   58   13    2   15   55  105
CALIFORNIA      102  157  102   50  102  100  149   96  100   51  103   95  151
               1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991
ALABAMA          72    7   25   12   61   13   42    2   37    8  147   42    3
ALASKA           47    3    0   32   26    0    0    0   11   15   99   59   91
AMERICAN SAMOA    0    0    0    0    0    0   29    2   13    1   28    0    1
ARIZONA          72   36   47   27   30   79   92   21  101    3  178   70   33
ARKANSAS         53    6   38    0   66   19   80    5    0   27  140   62    4
CALIFORNIA       96   45   89   93  146  255  166   56  139  152  226  142  139
               1992 1993 1994 1995 1996 1997 1998 1999 2000 2001
ALABAMA           0   31    0    0    0    0   44    0    0    7
ALASKA            0   24   28   88    3    0   49    0   65    0
AMERICAN SAMOA    0   44    0    0    0    0    0    0    0    0
ARIZONA           0   26   52  178   74    3   31   63   29    2
ARKANSAS          0    0   33   89    0    0    0   23   15    0
CALIFORNIA       24   96  110  249  141    6  113  133  234   37
```

## Complex Heatmap

`ComplexHeatmap` package is developed by *Zuguang Gu*, it's documentation (almost all of its information) can be found in its [website](https://jokergoo.github.io/ComplexHeatmap-reference/book/). 

<img src="https://jokergoo.github.io/ComplexHeatmap-reference/book/complexheatmap-cover.jpg" alt="img" style="zoom: 25%;" />

There is hardly any discussion outside. And the installation is a little bit different:

```R
library(devtools)
install_github("jokergoo/ComplexHeatmap")
```

Import the package like this:

```R
library(ComplexHeatmap)
```

## Raw Heatmap

Using the default settings of `ComplexHeatmap`.

```R
# default heatmap
Heatmap(measles_num_raw)
```

![raw_heatmap](raw_heatmap.png)

Features of this heatmap:

- Colors: Used "blue-white-red" continuous color function to map to numeric values of patient count, each color block is the cases number of certain year in certain state; This is OK, since it can denote value changes of numeric value and overall speaking clear.
- Names: Columns are different years, year number are written as x-label, however blocks are small so that the labels are all overlapping and hard to read. But we can see that the years are not in the time order; Similar things happened to the rows, where state names are too large.
- Clustering: That is due to the clustering feature which we can on the top (this is why year order is rearranged to place "similar" columns together) as well as on the left, respectively for columns and rows. This feature will rearrange rows and columns so that similar rows or columns in clustering will be placed together. Sometimes we need this feature to have a better insight of the data and also to get better alignment of the figure as similar color blocks are stacked together (the top right red corner, other strip-like features). However, in other situations we want to keep the original order. For instance the year columns, original order of time would be better since it shows the change of case along with the change of time.
- Others: Fonts, Titles, gaps, annotations and other features to add to the heatmap.

## Label Adjustment

Starting from the easiest part, our objective is to deal with the overlapping year label. The first attempt is to remove the x-labels.

### Remove Names

Use `show_column_names = FALSE` to remove x-labels, similarly, another parameter is `show_row_names`.

```R
# colname removed heatmap
Heatmap(measles_num_raw, show_column_names = FALSE, show_row_names = FALSE)
```

![label_removed_heatmap](label_removed_heatmap.png)

However, we do want to see the year information, just not every year. Maybe once a decade.

### Modify  Column Names

To modify column names so that we only see one year per decade, 9 other names should be set to empty string `''`. This modification can be done on the original matrix.

```R
# colname modified heatmap
cnames <- colnames(measles_num_raw)
for (i in seq(length(cnames))){
  if (i%%10!=1){
    cnames[i]=''
  }
}
cnames
colnames(measles_num_raw) <- cnames
```

## Clustering Adjustment

### Remove Clustering

Column clustering should be removed since we want to show year chronologically instead of grouping them based on similarity. The syntax for this is just adding parameter `cluster_columns = FALSE` or for rows `cluster_rows = FALSE`. 

### Hide Dendrogram

Maybe we want to cluster the states so that the figure looks more organized, but we don't need to show the dendrogram. Hiding dendrogram is controlled by the parameter: `show_row_dend` and `show_column_dend`.

```R
Heatmap(measles_num_raw, cluster_columns = FALSE, show_row_dend = FALSE)
```

![matrix_3](matrix_3.png)

<!-- 
## Color Adjustment

## Titles

## Annotations

## Legend

## Other Features

### Grid

### Fonts

### Export -->



