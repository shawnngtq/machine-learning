# Introduction to Machine Learning

Editor: Shawn Ng<br>
Content Author: **Vincent Vankrunkelsven**<br>
Site: https://www.datacamp.com/courses/introduction-to-machine-learning-with-r<br>

1. What is machine learning
	* Algorithms that learn from data
	* Building models for prediction
	1. Classification
		* Predict category of new observation
		* Earlier obs --(estimate)--> classifier
		* Unseen data --(classifier)--> class
	2. Regression
		* Predictors --(regression function)--> response
		* `lmModel <- lm(response~predictor, data=DATA)`
		* `test <- data.frame(predictor=VALUE-2:VALUE)`
		* `predictResult <- predict(lmModel, test)`
		* `plot(response~predictor, xlim=c(1,VALUE))`
		* `points(VALUE-2:VALUE, predictResult, col=COLOR)`
	3. Clustering
		* Grouping objects in clusters
		* kMeans
		* `type <- data$type`
		* `kmeans_data <- kMeans(data, NUMBER OF CLUSTERS)`
		* `table(type, kmeans_data$cluster)`
		* `plot(response~predictor, data=data, col=kmeans_data$cluster)`
	
	1. Supervised learning
		* Function used to assign a class/value to unseen obs
		* Given a set of **labeled** obs
		* Compare real labels with predicted labels
		* Predictions should be similar to real labels
		* `library(rpart)`
		* `decisionTree <- rpart(type ~ pred1 + pred2 + ... + predX, data=data, method='class')`
		* `test <- data.frame(pred1=c(x,x), ..., predX=c(x,x))`
		* `predict(decisionTree, test, type='class')`
	2. Unsupervised learning
		* Clustering: find groups observation that are similar
		* Does not require **labeled** observations
		* No real labels to compare
		* `kmData <- kmeans(data, NUMBER OF CLUSTERS)`
		* `plot(data, col=kmData$cluster)`
		* `points(kmData$centers, pch = 22, bg = c(1, 2), cex = 2)`
	3. Semi-supervised learning
		* Many unlabeled obs, few labeled obs
		* Uses clustering classes of labeled obs to assign class to unlabeled obs -> more obs for supervised learning
2. Performance measures
3. Classification
4. Regression
5. Clustering
