## The goal of this example is to try out the "mlr" package.
## Training will involve at least three foundational machine
## leanrning algorithms.
## No cross-validation step since this script does Not
## aim to fine tune and pick the best predictor for the iris 
## dataset that will work well on more unseen iris data

rm(list = ls())
library(dplyr)
library(ggplot2)
library(mlr)

data(iris)
glimpse(iris)
summary(iris)

# how many categories for "Species" variable?
# note: this is also provided by the summary() function
unique(iris$Species)

# look at a few observations
head(iris)
head(iris, n = 12)

# what is the class distribution?
# note: provided by the summary() function
table(iris$Species)


### plots - how are the other variables distributed among the species?
# Sepal.Width vs Sepal.Length
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot1.PNG"))
# Petal.Length vs Sepal.Length
ggplot(iris, aes(Sepal.Length, Petal.Length, color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot2.PNG"))
# Petal.Width vs Sepal.Length
ggplot(iris, aes(Sepal.Length, Petal.Width, color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot3.PNG"))
# Petal.Length vs Sepal.Width
ggplot(iris, aes(Sepal.Width, Petal.Length, color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot4.PNG"))
# Petal.Width vs Sepal.Width
ggplot(iris, aes(Sepal.Width, Petal.Width, color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot5.PNG"))
# Petal.Width vs Petal.Length
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot6.PNG"))

# Area of Petal vs Area of Sepal
ggplot(iris, aes(Sepal.Width*Sepal.Length, Petal.Width*Petal.Length,
                 color = Species)) +
  geom_point()
ggsave(file.path("plots",  "plot7.PNG"))

# observations:
# - evenly distributed class sizes
# - setosa is quite separated from the other two classes
# - versicolor and virginica have partial overlap at their boundaries

#****************************************************************************
set.seed(2342)

### train & predict
# 1) => linear discriminant analysis
# - as seen in example at http://mlr-org.github.io/mlr-tutorial/release/html/
# define the task
lda_task = makeClassifTask(data = iris, target = "Species")
# define the learner
lda_learner = makeLearner("classif.lda")

n = nrow(iris)
train_df = sample(n, size = 2/3 * n)
test_df = setdiff(1:n, train_df)

# fit the model
lda_model = train(lda_learner, lda_task, subset = train_df)
# make predictions
lda_pred = predict(lda_model, task = lda_task, subset = test_df)
# evaluate the learner -> mean misclassification error, and accuracy
performance(lda_pred, measures = list(mmce, acc))
#mmce  acc 
#0.02 0.98 

# clean up
rm(lda_task, lda_learner, lda_model, lda_pred)


# 2) => Naive Bayes, Decision Tree, k-Nearest Neighbors, 
#       support vector machine, and linear discriminant analysis (from 1 above)
# define the task
classify_task = makeClassifTask(data = iris, target = "Species")
# define the learners
chosen_learners = 
  makeLearners(c("naiveBayes", "rpart", "knn", "svm", "lda"), type = "classif")

# for each learner -> fit the model, make predictions,
# evaluate the learner -> mean misclassification error, and accuracy
gen_trainpred <- function(learner) {
  model = train(learner, classify_task, subset = train_df)
  pred = predict(model, task = classify_task, subset = test_df)
  performance(pred, measures = list(mmce, acc))
}

learners_perf <- sapply(chosen_learners, gen_trainpred)
# use the rds object for verification if you like
saveRDS(learners_perf, file = "learners.performace.rds")

# Conclusion: Looks like the chosen classification algorithms do 
# very well on the iris dataset with accuracies of 96% for 
# naive bayes and 98% for the others.


# comments, suggestions, and feedbacks are welcome
