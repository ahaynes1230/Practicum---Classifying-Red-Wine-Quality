#set directory
setwd("/Users/ashanihaynes")

#load dataset
red.wine <- read.csv("winequality-red.csv")
View(red.wine)
head(red.wine)

#any missing values?
#I've also opened up the dataset (in csv format) and checked to see if there were any missing values
is.na(red.wine)

#setting seed
set.seed(20)

#looking at the correlation matrix of the red wine dataset 
red.cor <- cor(red.wine)
View(red.cor)
#visualization of correlation matrix
install.packages("corrplot")
library(corrplot)
corrplot(red.cor)

#remove highly correlated variables
#Im going to remove fixed acidity and total sulfur dioxide 
red.wine2 <- subset(red.wine, select = -c(1, 7))
red.cor2 <- cor(red.wine2)
View(red.cor2)
corrplot(red.cor2)

##Converting quality into a binary factor 
#scale remaining variables the same 
scale.red <- red.wine2
scale.red[,c(1:9)] <- scale(scale.red[,c(1:9)])
View(scale.red)

#convert quality into binary factor
for (i in 1:nrow(scale.red)) {
  if (scale.red$quality[i] > 5)
    scale.red$label[i] <- 1
  else
    scale.red$label[i] <- 0
}
scale.red$label <- factor(scale.red$label, levels = c(0,1), labels = c("Dissatisfactory", "Worth a try"))

#remove quality variable 
scale.red$quality <- NULL

#Creating matrix/table for documentation for each type of prediction 
doc <- matrix(NA, nrow = 4, ncol = 3)
colnames(doc) <- c("Accuracy Rate", "Error Rate", "AUC")
rownames(doc) <- c("tree", "pruned.tree", "full.randomForest", "small.randomForest")
doc

#Splitting data between test and train Set 
test.index <- sample(1:nrow(scale.red), 1000)
test1 <- scale.red[test.index, ]
train1 <- scale.red[-test.index, ]
View(test1)

## DECISION TREE ##

install.packages("tree")
library(tree)
tree <- tree(formula = label ~ ., data = train1, 
             method = "class", 
             control = tree.control(nobs = nrow(train1)))
summary(tree)
# 5 out of 9 predictors were used to build this tree
plot(tree, type = "uniform")
text(tree, pretty = 0, cex = 1, col = "blue")
title("Classification Tree (Pre-Pruning)")

#function that returns the accuracy of a confusion matrix
class.accuracy <- function(conf) {
  sum(diag(conf)) / sum(conf)

}
tree.predict <- predict(tree, test1, type = "class")

#confusion matrix
tree.confus <- table(pred = tree.predict, true = test1$label)
tree.confus

#class.accuracy command is now defined
tree.acc <- class.accuracy(tree.confus)
tree.acc

#misclassification error 
tree.error <- 1 - tree.acc
tree.error

#retreiving matrix of predicted class probabilities 
install.packages("ROCR")
library(ROCR)
all.tree.prob <- as.data.frame(predict(tree, test1, type = "vector"))
tree.prob <- all.tree.prob[,2]
tree.roc.pred <- prediction(tree.prob, test1$label)
tree.roc.perf <- performance(tree.roc.pred, "tpr", "fpr")

#plotting ROC curve for decision tree
plot(tree.roc.perf, col = 2, lwd = 3,
     main = "RPC Curve for tree (Pre pruning)")
abline(0,1)

#AUC (area under curve)
tree.auc.perf <- performance(tree.roc.pred, "auc")
tree.AUC <- tree.auc.perf@y.values[[1]]
tree.AUC

#add results to documentation 
doc[1, ] <- c(tree.acc, tree.error, tree.AUC)
doc

## k-FOLD CROSS VALIDATION ##

set.seed(10)
library(tree)
cv <- cv.tree(tree, FUN = prune.misclass, K = 10)
cv

#best size
best.cv <- cv$size[which.min(cv$dev)]
#plotting misclass error as function of tree size (k)
plot(cv$size, cv$dev, type = "b",
     xlab = "# of Leaves, \'best\' ",
     ylab = "Misclassification Error",
     col = "green", main = "Optimal Tree Size")
abline(v =best.cv, lty = 2)
best.cv
#must prune tree to show it has 9 nodes
tree.pruned <- prune.tree(tree, best = best.cv,
                          method = "misclass")
summary(tree.pruned)
plot(tree.pruned, type = "uniform")
text(tree.pruned, col = "purple")
title("Pruned Classification Tree")

pruned.predict <- predict(tree.pruned, test1, type = "class")
#confusion matrix 
pruned.cm <- table(pred = pruned.predict, true = test1$label)
pruned.cm

pruned.acc <- class.accuracy(tree.confus)
pruned.acc
pruned.error <- 1 - tree.acc
pruned.error

#ROC curve w/ tree object, also retrieving matrix of predicted class probabilities 
all.pruned.prob <- as.data.frame(predict(tree.pruned, test1, type = "vector"))
pruned.prob <- all.pruned.prob[ ,2]

pruned.roc.predict <- prediction(pruned.prob, test1$label)
pruned.roc.performance <- performance(pruned.roc.predict, "tpr", "fpr")

#plotting ROC curve for rpart decision tree
plot(pruned.roc.performance, col = 2, lwd = 3, 
     main = "ROC Curve for Pruned Tree")
abline(0, 1)

pruned.auc.performance <- performance(pruned.roc.predict, "auc")
pruned.AUC <- pruned.auc.performance@y.values[[1]]
pruned.AUC

doc[2, ] <- c(pruned.acc, pruned.error, pruned.AUC)
doc

## Random Forest ##

install.packages("randomForest")
library(randomForest)
random.forest <- randomForest(formula = label ~ .,
                              data = train1,
                              mtry = 9)
print(random.forest)

varImpPlot(random.forest, main = "Variable Importance Plot")

#predicting on test set 
rf.predict <- predict(random.forest, test1, type = "class")

#confusion matrix 
rf.confusion <- table(true = test1$label, pred = rf.predict)
rf.confusion

#accuracy rate
rf.accuracy <- class.accuracy(rf.confusion)
rf.accuracy

#error rate
rf.error <- 1 - rf.accuracy
rf.error

#building ROC Curve
rf.predict <- as.data.frame(predict(random.forest, newdata = test1, type = 'prob'))
rf.predict.prob <- rf.predict[,2]
rf.roc.predict <- prediction(rf.predict.prob, test1$label)
rf.performance <- performance(rf.roc.predict, measure = "tpr",
                              x.measure = "fpr")
#plotting curve
plot(rf.performance, col = 2, lwd = 3,
     main = "ROC Curve for randomForest w/ all 8 varialbles")
abline(0, 1)

#area under curve
rf.performance2 <- performance(rf.roc.predict, measure = "auc")
rf.AUC <- rf.performance2@y.values[[1]]
rf.AUC

doc[3, ] <- c(rf.accuracy, rf.error, rf.AUC)
doc

random.forest2 <- randomForest(formula = label ~ alcohol + volatile.acidity + sulphates,
                               data = train1,
                               mtry = 3)
#predicting on test set
rf.predict2 <- predict(random.forest2, test1, type = "class")

#confusion matrix
rf.confusion2 <- table(test1$label, rf.predict2)
rf.confusion2

#accuracy rate
rf.accuracy2 <- class.accuracy(rf.confusion2)
rf.accuracy2

#error rate
rf.error2 <- 1 - rf.accuracy2
rf.error2

#building ROC Curve
rf.predict2 <- as.data.frame(predict(random.forest2, test1, type = 'prob'))
rf.predict.prob2 <- rf.predict2[,2]
rf.roc.predict2 <- prediction(rf.predict.prob2, test1$label)
rf.performance2 <- performance(rf.roc.predict2,
                               measure = "tpr",
                               x.measure = "fpr")
#plotting curve 
plot(rf.performance2, col = 2, lwd = 3,
     main = "ROC Curve for randomForest w/ 3 variables")
abline(0, 1)

#area under curve 
rf.performance22 <- performance(rf.roc.predict2, measure = "auc")
rf.AUC2 <- rf.performance22@y.values[[1]]
rf.AUC2

doc[4, ] <- c(rf.accuracy2, rf.error2, rf.AUC2)
doc
























































