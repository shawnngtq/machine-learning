# Machine Learning
Reference:
- [Cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)
- [List of cheatsheets](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463)

Table of content:
1. [Classification](#classification-predict-category-of-new-observation)
    1. [Decision tree](#decision-tree)
    2. [k-Nearest Neighbors (kNN)](#knn)
    3. [ROC curve](#roc-curve)
2. [Regression](#regression)
    1. [Linear models](#linear-model)
    2. [Techniques for non-parametric regression](#techniques-for-non-parametric-regression)
3. [Clustering](#clustering)
    1. [k-Means](#k-means)
    2. [Scree plot](#scree-plot-choosing-k)
    3. [Hierarchical clustering](#hierarchical-clustering)
4. [Bias and Variance](#bias-and-variance)
5. [Model](#model)



## Classification: Predict category of new observation
### Performance measure: confusion matrix

|||Prediction|Prediction|
| --- | --- | :---: | :---: |
||| P | N |
| **Truth** | P | TP | FN |
| **Truth** | N | FP | TN |

1. Accuracy: `(TP+TN)/(TP+FP+FN+TN)`
2. Precision: `TP/(TP+FP)`
3. Recall: `TP/(TP+FN)`

```
Accuracy = correctly classified instances / total amount of classified instances
Error = 1 - Accuracy
```

### Decision tree
Goal: end up with pure leafs — leafs that contain observations of one particular class. At each node iterate over different feature tests and choose the best. Use **spliting criteria** to decide which test to use. The best test leads to nicely divided classes -> high information gain

**Pruning** tree restricts node size = higher bias = lower overfit chance.

### kNN
Predict using the comparison of **unseen data** and the **training set**

### ROC Curve
Receiver Operator Characteristic Curve: Binary classification which uses decision trees and k-NN to predict class, giving probability as output

```
# TPR = True positive rate; FPR = False positive rate
TPR = recall
FPR = FP / (FP+TN)
```



## Regression
### Performance measure:
1. Root Mean Squared Error (RMSE)
2. R-squared
3. Adjusted R-squared: penalizes more predictors
4. p-values: low p-values = parameter has significant influence

### Linear model:
1. Simple linear: 1 predictor (with approximately linear relationship) to model the response
2. Multi-linear: Higher **predictive power** and **accuracy** = Lower RMSE and higher R-squared
3. Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity.
    - It reduces the standard errors.
    - It adds penalty equivalent to square of the magnitude of coefficients

### Techniques for non-parametric regression
1. KNN
2. Kernel Regression
3. Regression Trees



## Clustering
- Grouping objects in clusters

### Performance measure:
1. Within sum of squares (WSS)
2. Between cluster sum of squares (BSS)
3. Dunn’s index

### k-Means
- Partition data in **k disjoint** subsets

### Scree plot: Choosing k
- WSS keeps decreasing as k increases -> Find k that minimizes WSS

```
TSS = WSS + BSS
WSS / TSS < 0.2
```

### Hierarchical clustering
1. Simple-Linkage: minimal distance between clusters = low BSS
2. Complete-Linkage: maximal distance between clusters = high BSS
3. Average-Linkage: average distance between clusters



## Bias and variance
- Prediction error = reducible + irreducible error
- Reducible error = Bias & Variance

```
Error due to bias = Wrong assumption = diff(prediction, truth) = complexity of model
More model restrictions = high bias = low variance = underfitting
```

```
Error due to variance = error due to the sampling of the training set
Model fits training set closely = high variance = low bias = overfitting
```

## Model
### Cross validation
- Use to validate the stability of the model
