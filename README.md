# Machine Learning

1. [Classification](#classification)
    1. [Decision tree](#decision-tree)
    2. [kNN](#k-nearest-neighbors-knn)
    3. [ROC curve](#receiver-operator-characteristic-curve-roc-curve)
2. [Regression](#regression)
    1. [Linear models]()
    2. [Techniques for non-parametric regression]()
3. [Clustering](#clustering)
4. [Bias and Variance](#bias-and-variance)


## Classification
Predict category of new observation

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

### k-Nearest Neighbors (kNN)
Predict using the comparison of **unseen data** and the **training set**

### Receiver Operator Characteristic Curve (ROC Curve)
Binary classification which uses decision trees and k-NN to predict class, giving probability as output

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

### Techniques for non-parametric regression
1. KNN
2. Kernel Regression
3. Regression Trees


## Clustering
### Performance measure:
1. Within sum of squares (WSS)
2. Between cluster sum of squares (BSS)
3. Dunn’s index

## Bias and Variance
```
Prediction error ~ reducible + irreducible error
Reducible error = Bias & Variance
```
```
Error due to bias = Wrong assumption = diff(prediction, truth) = complexity of model
More model restrictions = high bias = low variance = underfitting
```
```
Error due to variance = error due to the sampling of the training set
Model fits training set closely = high variance = low bias = overfitting
```
