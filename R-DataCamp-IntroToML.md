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
2. [Performance measures](#2-performance-measures)
	1. Ratio in confusion matrix
		1. Accuracy: `(TP+TN) / (TP+FP+FN+TN)`
		2. Precision: `TP / (TP+FP)`
		3. Recall: `TP / (TP+FN)`
3. [Classification](#3-classification)
4. [Regression](#4-regression)
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





## 4. Regression
```r
## Simple linear regression
# Plot nose length as function of nose width.
plot(kang_nose, xlab = "nose width", ylab = "nose length")

# Describe the linear relationship between the two variables: lm_kang
lm_kang <- lm(nose_length ~ nose_width, data = kang_nose)

# Print the coefficients of lm_kang
print(lm_kang$coef)

# Predict and print the nose length of the escaped kangoroo
print(predict(lm_kang, nose_width_new))


## Performance measure
abline(lm_kang$coefficients, col = "red")

# Apply predict() to lm_kang: nose_length_est
nose_length_est <- predict(lm_kang)

# Calculate difference between the predicted and the true values: res
res <- kang_nose$nose_length - nose_length_est

# Calculate RMSE, assign it to rmse and print it
rmse <- sqrt(sum(res^2)/nrow(kang_nose))
print(rmse)


# Calculate the residual sum of squares: ss_res
ss_res <- sum(res^2)

# Determine the total sum of squares: ss_tot
ss_tot <- sum((kang_nose$nose_length - mean(kang_nose$nose_length))^2)

# Calculate R-squared and assign it to r_sq. Also print it.
r_sq <- 1 - (ss_res / ss_tot)
print(r_sq)

# Apply summary() to lm_kang
# summary multiple R-squared is the same as r_sq
summary(lm_kang)
```

```r
plot(world_bank_train)

# Set up a linear model between the two variables: lm_wb
lm_wb <- lm(urb_pop ~ cgdp, data=world_bank_train)

# Add a red regression line to your scatter plot
abline(lm_wb$coef, col="red")

# Summarize lm_wb and select R-squared
summary(lm_wb)$r.squared

# Predict the urban population of afghanistan based on cgdp_afg
predict(lm_wb, cgdp_afg)


## Improve model
# Plot: change the formula and xlab
plot(urb_pop ~ log(cgdp), data = world_bank_train,
     xlab = "log(GDP per Capita)",
     ylab = "Percentage of urban population")

# Linear model: change the formula
lm_wb <- lm(urb_pop ~ log(cgdp), data = world_bank_train)

# Add a red regression line to your scatter plot
abline(lm_wb$coefficients, col = "red")

# Summarize lm_wb and select R-squared
summary(lm_wb)$r.squared

# Predict the urban population of afghanistan based on cgdp_afg
predict(lm_wb, cgdp_afg)
```

### Multivariable Linear Regression
```r
plot(sales ~ sq_ft, shop_data)
plot(sales ~ size_dist, shop_data)
plot(sales ~ inv, shop_data)

# Build a linear model for net sales based on all other variables: lm_shop
lm_shop <- lm(sales ~ ., shop_data)

# Summarize lm_shop
summary(lm_shop)


## Are all predictors relevant?
# Plot the residuals in function of your fitted observations
plot(lm_shop$fitted.values, lm_shop$residuals, ylab="Residual Quantiles")

# Make a Q-Q plot of your residual quantiles
qqnorm(lm_shop$residuals, ylab="Residual Quantiles")

# Summarize your model
# small p-values -> every predictor is importan
summary(lm_shop)

# Predict the net sales based on shop_new.
predict(lm_shop, shop_new)
```

```r
# Add a plot:  energy/100g as function of total size. Linearity plausible?
plot(energy ~ protein, choco_data)
plot(energy ~ fat, choco_data)
plot(energy ~ size, choco_data)

# Build a linear model for the energy based on all other variables: lm_choco
lm_choco <- lm(energy ~ ., data=choco_data)

# Plot the residuals in function of your fitted observations
plot(lm_choco$fitted.values, lm_choco$residuals)

# Make a Q-Q plot of your residual quantiles
qqnorm(lm_choco$residuals)

# Summarize lm_choco
# low Pr(>|t|) < 0.05 -> statistically significant
summary(lm_choco)
```

### Generalization in Regression
```r
## log-linear model
lm_wb_log <- lm(urb_pop ~ log(cgdp), data = world_bank_train)

# Calculate rmse_train
rmse_train <- sqrt(mean(lm_wb_log$residuals ^ 2))

# The real percentage of urban population in the test set, the ground truth
world_bank_test_truth <- world_bank_test$urb_pop

# The predictions of the percentage of urban population in the test set
world_bank_test_input <- data.frame(cgdp = world_bank_test$cgdp)
world_bank_test_output <- predict(lm_wb_log, world_bank_test_input)

# The residuals: the difference between the ground truth and the predictions
res_test <- world_bank_test_output - world_bank_test_truth


# Use res_test to calculate rmse_test
rmse_test <- sqrt(sum(res_test^2)/length(res_test))

# Print the ratio of the test RMSE over the training RMSE
# The test's RMSE is only slightly larger than the training RMSE. This means that your model generalizes well to unseen observations
print(rmse_test/rmse_train)


## non-parametric k-NN algorithm
# x_pred: predictor values of the new observations (this will be the cgdp column of world_bank_test)
# x: predictor values of the training set (the cgdp column of world_bank_train)
# y: corresponding response values of the training set (the urb_pop column of world_bank_train)
# k: the number of neighbors (this will be 30)

my_knn <- function(x_pred, x, y, k){
  m <- length(x_pred)
  predict_knn <- rep(0, m)
  for (i in 1:m) {

    # Calculate the absolute distance between x_pred[i] and x
    dist <- abs(x_pred[i] - x)

    # Apply order() to dist, sort_index will contain
    # the indices of elements in the dist vector, in
    # ascending order. This means sort_index[1:k] will
    # return the indices of the k-nearest neighbors.
    sort_index <- order(dist)

    # Apply mean() to the responses of the k-nearest neighbors
    predict_knn[i] <- mean(y[sort_index[1:k]])

  }
  return(predict_knn)
}

# Apply your algorithm on the test set: test_output
test_output <- my_knn(world_bank_test$cgdp, world_bank_train$cgdp, world_bank_train$urb_pop, 30)

# Have a look at the plot of the output
plot(world_bank_train,
     xlab = "GDP per Capita",
     ylab = "Percentage Urban Population")
points(world_bank_test$cgdp, test_output, col = "green")


## Parametric vs non-parametric
# Define ranks to order the predictor variables in the test set
ranks <- order(world_bank_test$cgdp)

# Scatter plot of test set
plot(world_bank_test,
     xlab = "GDP per Capita", ylab = "Percentage Urban Population")

# Predict with simple linear model and add line
test_output_lm <- predict(lm_wb, data.frame(cgdp = world_bank_test$cgdp))
lines(world_bank_test$cgdp[ranks], test_output_lm[ranks], lwd = 2, col = "blue")

# Predict with log-linear model and add line
test_output_lm_log <- predict(lm_wb_log, data.frame(cgdp = world_bank_test$cgdp))
lines(world_bank_test$cgdp[ranks], test_output_lm_log[ranks], lwd = 2, col = "red")

# Predict with k-NN and add line
test_output_knn <- my_knn(world_bank_test$cgdp, world_bank_train$cgdp, world_bank_train$urb_pop, 30)
lines(world_bank_test$cgdp[ranks], test_output_knn[ranks], lwd = 2, col = "green")

# Calculate RMSE for simple linear model
sqrt(mean( (test_output_lm - world_bank_test$urb_pop) ^ 2))

# Calculate RMSE for log-linear model
# log-linear model has the lowest RMSE -> most suitable model
sqrt(mean( (test_output_lm_log - world_bank_test$urb_pop) ^ 2))

# Calculate RMSE for k-NN technique
sqrt(mean( (test_output_knn - world_bank_test$urb_pop) ^ 2))
```





## 5. Clustering
### kmeans
```r
set.seed(100)

# Do k-means clustering with three clusters, repeat 20 times: seeds_km
seeds_km <- kmeans(seeds, 3, nstart=20)

# Print out seeds_km
print(seeds_km)

# Compare clusters with actual seed types. Set k-means clusters as rows
table(seeds_km$cluster, seeds_type)

# Plot the length as function of width. Color by cluster
plot(seeds$width, seeds$length, col=seeds_km$cluster)


## The influence of starting centroids
# If you call kmeans() without specifying your centroids, R will randomly assign them for you.
# To compare the clusters of two cluster models, you can again use table(). 
# If every row and every column has one value, the resulting clusters completely overlap. 
# If this is not the case, some objects are placed in different clusters.
# For consistent and decent results, you should set nstart > 1 or determine a prior estimation of your centroids

set.seed(100)

# Apply kmeans to seeds twice: seeds_km_1 and seeds_km_2
seeds_km_1 <- kmeans(seeds, 5, nstart=1)
seeds_km_2 <- kmeans(seeds, 5, nstart=1)

# Return the ratio of the within cluster sum of squares
print(seeds_km_1$tot.withinss / seeds_km_2$tot.withinss)

# Compare the resulting clusters
table(seeds_km_1$cluster, seeds_km_2$cluster)
```

### scree plot
```r
set.seed(100)
str(school_result)

# Initialise ratio_ss 
ratio_ss <- rep(0, 7)

# Finish the for-loop. 
for (k in 1:7) {
  
  # Apply k-means to school_result: school_km
  school_km <- kmeans(school_result, k, nstart=20)
  
  # Save the ratio between of WSS to TSS in kth element of ratio_ss
  ratio_ss[k] <- school_km$tot.withinss / school_km$totss
  
}

# Make a scree plot with type "b" and xlab "k"
plot(ratio_ss, type="b", xlab="k")
```

### Standardized vs non-standardized clustering
```r
## Non-standardized clustering
set.seed(1)
str(run_record)
summary(run_record)

# Cluster run_record using k-means: run_km. 5 clusters, repeat 20 times
run_km <- kmeans(run_record, 5, nstart = 20)

# Plot the 100m as function of the marathon. Color using clusters
plot(run_record$marathon, run_record$X100m, col = run_km$cluster,
     xlab = "marathon", ylab ="100m", main = "Run Records")

# Calculate Dunn's index: dunn_km. Print it.
dunn_km <- dunn(clusters = run_km$cluster, Data = run_record)
print(dunn_km)


## Standardized clustering
# Standardize run_record, transform to a dataframe: run_record_sc
run_record_sc <- as.data.frame(scale(run_record))

# Cluster run_record_sc using k-means: run_km_sc. 5 groups, let R start over 20 times
run_km_sc <- kmeans(run_record_sc, 5, nstart=20)

# Plot records on 100m as function of the marathon. Color using the clusters in run_km_sc
plot(run_record$marathon, run_record$X100m, col=run_km_sc$cluster)

# Compare the resulting clusters in a nice table
table(run_km$cluster, run_km_sc$cluster)

# Calculate Dunn's index: dunn_km_sc. Print it.
dunn_km_sc <- dunn(cluster=run_km_sc$cluster, Data=run_record_sc)
print(dunn_km_sc)
```

### Hierarchical Clustering
```r
## Single Hierarchical Clustering
run_dist <- dist(run_record_sc)

# Apply hclust() to run_dist: run_single
run_single <- hclust(run_dist, method="single")

# Apply cutree() to run_single: memb_single
memb_single <- cutree(run_single, 5)

# Apply plot() on run_single to draw the dendrogram
plot(run_single)

# Apply rect.hclust() on run_single to draw the boxes
rect.hclust(run_single, 5, border=2:6)


## Complete Hierarchical Clustering
# Code for single-linkage
run_dist <- dist(run_record_sc, method = "euclidean")
run_single <- hclust(run_dist, method = "single")
memb_single <- cutree(run_single, 5)
plot(run_single)
rect.hclust(run_single, k = 5, border = 2:6)

# Apply hclust() to run_dist: run_complete
run_complete <- hclust(run_dist, method="complete")

# Apply cutree() to run_complete: memb_complete
memb_complete <- cutree(run_complete, 5)

# Apply plot() on run_complete to draw the dendrogram
plot(run_complete)

# Apply rect.hclust() on run_complete to draw the boxes
rect.hclust(run_complete, 5, border=2:6)

# table() the clusters memb_single and memb_complete. Put memb_single in the rows
table(memb_single, memb_complete)
```

### Hierarchical vs k-means
```r
set.seed(100)

# Dunn's index for k-means: dunn_km
dunn_km <- dunn(clusters = run_km_sc$cluster, Data = run_record_sc)

# Dunn's index for single-linkage: dunn_single
dunn_single <- dunn(clusters = memb_single, Data = run_record_sc)

# Dunn's index for complete-linkage: dunn_complete
dunn_complete <- dunn(clusters = memb_complete, Data = run_record_sc)

# Compare k-means with single-linkage
table(run_km_sc$cluster, memb_single)

# Compare k-means with complete-linkage
table(run_km_sc$cluster, memb_complete)
```

```r
set.seed(1)

# Scale the dataset: crime_data_sc
crime_data_sc <- scale(crime_data)

# Perform k-means clustering: crime_km
crime_km <- kmeans(crime_data_sc, 4, nstart=20)

# Perform single-linkage hierarchical clustering
## Calculate the distance matrix: dist_matrix
dist_matrix <- dist(crime_data_sc)

## Calculate the clusters using hclust(): crime_single
crime_single <- hclust(dist_matrix, method="single")

## Cut the clusters using cutree: memb_single
memb_single <- cutree(crime_single, 4)

# Calculate the Dunn's index for both clusterings: dunn_km, dunn_single
dunn_km <- dunn(clusters=crime_km$cluster, Data=crime_data_sc)
dunn_single <- dunn(clusters=memb_single, Data=crime_data_sc)

# Print out the results
print(dunn_km)
print(dunn_single)
```