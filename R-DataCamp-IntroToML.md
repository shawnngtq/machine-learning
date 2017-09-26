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
tree <- rpart(response ~ ., data = train, method = "class")

# Use the predict() method to make predictions, assign to pred
pred <- predict(tree, test, type="class")

# Use the table() method to make the confusion matrix
conf <- table(test$response, pred)

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

### Using Cross Validation
```r
set.seed(1)

# Initialize the accs vector
accs <- rep(0,6)

for (i in 1:6) {
  # These indices indicate the interval of the test set
  indices <- (((i-1) * round((1/6)*nrow(shuffled))) + 1):((i*round((1/6) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  
  # Include them in the test set
  test <- shuffled[indices,]
  
  # A model is learned using each training set
  tree <- rpart(Survived ~ ., train, method = "class")
  
  # Make a prediction on the test set using tree
  pred <- predict(tree, test, type="class")
  
  # Assign the confusion matrix to conf
  conf <- table(test$Survived, pred)
  
  # Assign the accuracy of this model to the ith index in accs
  accs[i] <- sum(diag(conf))/sum(conf)
}

# Print out the mean of accs
print(mean(accs))
```





## 3. Classification
### Learn a decision tree
```r
set.seed(1)

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Fill in the ___, build a tree model: tree
tree <- rpart(Survived ~ ., train, method="class")

# Draw the decision tree
fancyRpartPlot(tree)

# Predict the values of the test set: pred
pred <- predict(tree, test, type="class")

# Construct the confusion matrix: conf
conf <- table(test$Survived, pred)

# Print out the accuracy
print(sum(diag(conf))/sum(conf))


## Pruning
tree <- rpart(Survived ~ ., train, method = "class", control = rpart.control(cp=0.00001))

# Draw the complex tree
fancyRpartPlot(tree)

# Prune the tree: pruned
pruned <- prune(tree, cp=0.01)

# Draw pruned
fancyRpartPlot(pruned)


## Splitting criterion
# Train and test tree with gini criterion
# The standard splitting criterion of rpart() is the Gini impurity
# Change the first line of code to use information gain as splitting criterion
tree_i <- rpart(spam ~ ., train, method = "class", parms = list(split = "information"))
pred_i <- predict(tree_i, test, type = "class")
conf_i <- table(test$spam, pred_i)
acc_i <- sum(diag(conf_i)) / sum(conf_i)

fancyRpartPlot(tree_i)
```

### k-Nearest Neighbors (k-NN)
```r
## Preprocess the data
# Store the Survived column of train and test in train_labels and test_labels
train_labels <- train$Survived
test_labels <- test$Survived

# Copy train and test to knn_train and knn_test
knn_train <- train
knn_test <- test

# Drop Survived column for knn_train and knn_test
knn_train$Survived <- NULL
knn_test$Survived <- NULL

# Normalize Pclass
min_class <- min(knn_train$Pclass)
max_class <- max(knn_train$Pclass)
knn_train$Pclass <- (knn_train$Pclass - min_class) / (max_class - min_class)
knn_test$Pclass <- (knn_test$Pclass - min_class) / (max_class - min_class)

# Normalize Age
min_age <- min(knn_train$Age)
max_age <- max(knn_train$Age)
knn_train$Age <- (knn_train$Age - min_age) / (max_age - min_age)
knn_test$Age <- (knn_test$Age - min_age) / (max_age - min_age)


## knn()
set.seed(1)
library(class)

# Make predictions using knn: pred
pred <- knn(train = knn_train, test = knn_test, cl = train_labels, k = 5)

# Construct the confusion matrix: conf
conf <- table(test_labels, pred)

# Print out the confusion matrix
print(conf)


## K's choice
library(class)
range <- 1:round(0.2 * nrow(knn_train))
accs <- rep(0, length(range))

for (k in range) {

  # make predictions using knn: pred
  pred <- knn(train=knn_train, test=knn_test, cl=train_labels, k = k)

  # construct the confusion matrix: conf
  conf <- table(test_labels, pred)

  # calculate the accuracy and store it in accs[k]
  accs[k] <- sum(diag(conf)) / sum(conf)
}

# Plot the accuracies. Title of x-axis is "k".
plot(range, accs, xlab = "k")

# Calculate the best k
which.max(accs)
```

### ROC curve
```r
set.seed(1)

# Build a tree on the training set: tree
tree <- rpart(income ~ ., train, method = "class")

# Predict probability values using the model: all_probs
all_probs <- predict(tree, test,, type="prob")

# Print out all_probs
print(all_probs)

# Select second column of all_probs: probs
probs <- all_probs[,2]


tree <- rpart(income ~ ., train, method = "class")
probs <- predict(tree, test, type = "prob")[,2]
library(ROCR)

# Make a prediction object: pred
pred <- prediction(probs, test$income)

# Make a performance object: perf
# "tpr", "fpr": true positive rate and false positive rate
perf <- performance(pred, "tpr", "fpr")

# Plot this curve
plot(perf)


# Make a performance object: perf
# "auc": area under curve
perf <- performance(pred, "auc")

# Print out the AUC
print(perf@y.values[[1]])


## Comparing the methods
library(ROCR)

# probs_t for the decision tree model, probs_k for k-Nearest Neighbors
# Make the prediction objects for both models: pred_t, pred_k
pred_t <- prediction(probs_t, test$spam)
pred_k <- prediction(probs_k, test$spam)

# Make the performance objects for both models: perf_t, perf_k
perf_t <- performance(pred_t, "tpr", "fpr")
perf_k <- performance(pred_k, "tpr", "fpr")

# Draw the ROC lines using draw_roc_lines()
draw_roc_lines(tree = perf_t, knn = perf_k)
```