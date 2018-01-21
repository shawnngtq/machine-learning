# Machine Learning

1. [Classification](#classification): Supervised learning
2. [Regression](#regression): Supervised learning
3. [Clustering](#clustering): Unsupervised learning


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

`Accuracy = correctly classified instances / total amount of classified instances`

`Error = 1 - Accuracy`

## Regression
**Performance measure: Root Mean Squared Error (RMSE)**

## Clustering
**Performance measure**：
1. Within sum of squares (WSS)
2. Between cluster sum of squares (BSS)
3. Dunn’s index
