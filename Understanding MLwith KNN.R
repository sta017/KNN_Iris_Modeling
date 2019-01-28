library(ggplot2)
library(repr)


features <- c('Sepal.Width', 'Sepal.Length', 'Petal.Width', 'Petal.Length')
iris[, features] <- lapply(iris[,features], scale)
print(summary(iris))
print(sapply(iris[, features], sd))


library(dplyr)	### to split using BERNOULLI SAMPLING
set.seed(1234)
train.iris <- sample_frac(iris, 0.7)
test.iris <- iris[-as.numeric(rownames(train.iris)),]   #use as.numeric() as 
									#rownames() returns character
dim(test.iris)

#Training and evaluating the  KNN model
library(kknn)
knn.3 <- kknn(Species ~ .,train = train.iris, test = test.iris, k=3)
summary(knn.3)

test.iris$predicted = predict(knn.3)              #inputs prediction made into the new column
test.iris$correct = test.iris$Species== test.iris$predicted
round(100* sum(test.iris$correct)/nrow(test.iris))






































