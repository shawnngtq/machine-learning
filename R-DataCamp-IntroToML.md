# Introduction to Machine Learning

Editor: Shawn Ng<br>
Content Author: **Vincent Vankrunkelsven**<br>
Site: https://www.datacamp.com/courses/introduction-to-machine-learning-with-r<br>

1. [What is machine learning](#1-what-is-machine-learning)
	* Algorithms that learn from data
	* Building models for prediction
	* ML problems
		1. [Classification](#classification)
		2. [Regression](#regression)
		3. [Clustering](#clustering)
	* ML tasks
		1. [Supervised learning](#supervised-learning)
		2. [Unsupervised learning](#unsupervised-learning)
		3. [Semi-supervised learning](#semi-supervised-learning)
2. Performance measures
	1. Ratio in confusion matrix
		1. Accuracy: `(TP+TN) / (TP+FP+FN+TN)`
		2. Precision: `TP / (TP+FP)`
		3. Recall: `TP / (TP+FN)`
3. Classification
4. Regression
5. Clustering





## 1. What is machine learning
### Classification
Predict category of new observation

Earlier obs --(estimate)--> classifier

Unseen data --(classifier)--> class

#### Performance measure
accuracy = correctly classified instances / total classified instances

error = 1 - accuracy


### Regression
Predictors --(regression function)--> response

```r
lmModel <- lm(response~predictor, data=DATA)
test <- data.frame(predictor=VALUE-2:VALUE)
predictResult <- predict(lmModel, test)
plot(response~predictor, xlim=c(1,VALUE))
points(VALUE-2:VALUE, predictResult, col=COLOR)
```

#### Performance measure
RMSE: Root Mean Squared Error

Mean distance bet. estimates and regression line


### Clustering
Grouping objects in clusters

```r
type <- data$type
kmeans_data <- kMeans(data, NUMBER OF CLUSTERS)
table(type, kmeans_data$cluster)
plot(response~predictor, data=data, col=kmeans_data$cluster)
```

#### Performance measure
Similarity within each cluster: Within sum of squares (WSS)

Similarity bet. clusters: Between cluster sum of squares (BSS)


### Supervised learning
Function used to assign a class/value to unseen obs.

Given a set of **labeled** obs.

Compare real labels with predicted labels.

Predictions should be similar to real labels.

```r
library(rpart)
decisionTree <- rpart(type ~ pred1 + pred2 + ... + predX, data=data, method='class')
test <- data.frame(pred1=c(x,x), ..., predX=c(x,x))
predict(decisionTree, test, type='class')
```


### Unsupervised learning
Clustering: find groups observation that are similar

Does not require **labeled** observations

No real labels to compare

```r
kmData <- kmeans(data, NUMBER OF CLUSTERS)
plot(data, col=kmData$cluster)
points(kmData$centers, pch = 22, bg = c(1, 2), cex = 2)
```


### Semi-supervised learning
Many unlabeled obs, few labeled obs

Uses clustering classes of labeled obs to assign class to unlabeled obs -> more obs for supervised learning





## 2. Performance measures
### The Confusion Matrix
```r
# A decision tree classification model is built on the data
tree <- rpart(response ~ ., data = data, method = "class")

# Use the predict() method to make predictions, assign to pred
pred <- predict(tree, data, type="class")

# Use the table() method to make the confusion matrix
conf <- table(data$response, pred)

# Assign TP, FN, FP and TN using conf
TP <- conf[1, 1]
FN <- conf[1, 2]
FP <- conf[2, 1]
TN <- conf[2, 2]

# accuracy: acc
acc <- (TP+TN) / (TP+FN+FP+TN)

# precision: prec
prec <- TP / (TP+FP)

# recall: rec
rec <- TP / (TP+FN)
```

### The quality of a regression
```r
str(air)

# Inspect your colleague's code to build the model
fit <- lm(dec ~ freq + angle + ch_length, data = air)

# Use the model to predict for all values: pred
pred <- predict(fit)

# Use air$dec and pred to calculate the RMSE 
rmse <- sqrt((1/nrow(air)) * sum((air$dec - pred)^2))

# Print out rmse
print(rmse)


# Previous model
fit <- lm(dec ~ freq + angle + ch_length, data = air)
pred <- predict(fit)
rmse <- sqrt(sum( (air$dec - pred) ^ 2) / nrow(air))
rmse

# Your colleague's more complex model
fit2 <- lm(dec ~ freq + angle + ch_length + velocity + thickness, data = air)

# Use the model to predict for all values: pred2
pred2 <- predict(fit2)

# Calculate rmse2
rmse2 <- sqrt(sum( (air$dec - pred2) ^ 2) / nrow(air))

# Print out rmse2
rmse2
```

### Clustering
The within sum of squares is far lower than the between sum of squares. Indicating the clusters are well seperated and overall compact
```r
set.seed(1)
str(seeds)

# Group the seeds in three clusters
km_seeds <- kmeans(seeds, 3)

# Color the points in the plot based on the clusters
plot(length ~ compactness, data = seeds, col=km_seeds$cluster)

# Print out the ratio of the WSS to the BSS
# WSS: within sum of squares
# BSS: between cluster sum of squares
km_seeds$tot.withinss / km_seeds$betweenss
```

### Split the sets into training and test data
```r
set.seed(1)

# Shuffle the dataset, call the result shuffled
n <- nrow(titanic)
shuffled <- titanic[sample(n),]

# Split the data in train and test
train <- shuffled[1:round(0.7*n),]
test <- shuffled[(round(0.7 * n) + 1):n,]

# Print the structure of train and test
str(train)
str(test)
```
