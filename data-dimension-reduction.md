# Data Dimension Reduction
This notebook is a summary of the medium article [A Complete Guide On Dimensionality Reduction](https://medium.com/analytics-vidhya/a-complete-guide-on-dimensionality-reduction-62d9698013d2). It covers neccessary of dimensionlity reductions, techniques of it.

## Dimensionality Reduction
Dimensionality Reduction is the process of reduction of $n$-dimentions to smaller $k$-dimensions. Given we define $a$, $b$, and $c$ like below,
```
a = [1,2,3]
```
```
b = [[1,2],[3,4]]
``` 
```
c = [
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ],
    [
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27]
    ]
]
```
we can see that visualising larger dimensions array is getting difficult to understand. The dimension reduction is not feasible to analyze every high dimensional data. It does not comnpute in a constant time, it may takes months or even years. Training larger dimensional data derive these following problems. 
1. Large Space reuiqred to store data.
2. Larger dimensions can cause overfitting the model also get increase.
3. We can not visualise high dimensional data.

### Components of Dimension Reduction
1. Feature Selection

2. Feature Projection (Extraction)

Feature Projection also known as Feature Extraction is used to transfrom the high dimensional data to low dimensional space. This transformation can be done in both linear (PCA and LDA) and non-linear (T-SNE). 

## Princpal Component Analysis
