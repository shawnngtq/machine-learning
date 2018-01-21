# Machine Learning

1. [Classification](#classification)
2. [Regression](#regression)
3. [Clustering](#clustering)
4. [Bias and Variance](#bias-and-variance)

## Classification
Predict category of new observation

**Performance measure: confusion matrix**

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

## Regression
**Performance measure: Root Mean Squared Error (RMSE)**

## Clustering
**Performance measure**：
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
