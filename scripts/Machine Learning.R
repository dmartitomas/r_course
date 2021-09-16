### DATA SCIENCE: MACHINE LEARNING

# In this course, you will learn:

# The basics of machine learning.
# How to perform cross-validation to avoid overtraining.
# Several popular machine learning algorithms.
# How to build a recommendation system.
# What regularization is and why it is useful.

# ASSESSMENT: PROGRAMMING SKILLS

# This problem set uses the heights dataset from the dslabs package, which consists
# of actual heights (in inches) of students in 3 Harvard biostatistics courses.

# Install the dslabs package from CRAN, then load the dslabs package into your
# workspace with the library() command.

# After loading the package, load the dataset heights into your workspace:
library(dslabs)
data(heights)
heights

# Q1: Object Classes

# Match each object to its corresponding class.

# heights dataset
class(heights) # data.frame

# sex column
class(heights$sex) # factor

# height column
class(heights$height) # numeric

# "Male"
class("Male") # character

# 75.00000
class(75.00000) # numeric

# Q2: Object Dimensions

# How many rows are in this dataset?
nrow(heights) # 1,050

# Q3: Indexing - 1

# What is the height in row 777?
heights$height[777] # 61

# Q4: Indexing - 2

# Which of these pieces of code returns the sex in row 777?

# heights$sex[777] [X]
# heights[1, 777]
# heights[777, 1] [X]

heights$sex[777]
heights[777, 1]

# Q5: Maximum and Minimum

# What is the maximum height in inches?
max(heights$height) # 82.67

# Which row has the minimum height?
which.min(heights$height) # 1,032

# Q6: Summary statistics

# What is the mean height in inches?
mean(heights$height) # 68.32

# What is the median height in inches?
median(heights$height)

# Q7: Conditional statements - 1

# What proportion of individuals in the dataset are male?
sum(heights$sex == "Male") / nrow(heights) # .773, also mean(heights$sex == "Male")

# Q8: Conditional statements - 2

# How many individuals are taller than 78 inches (roughly 2 meters)?
sum(heights$height > 78) # 9

# Q9: Conditional statements - 3

# How many females in the dataset are taller than 78 inches?
sum(heights$height > 78 & heights$sex == "Female") # 2

## SECTION 1: INTRODUCTION TO MACHINE LEARNING

# INTRODUCTION TO MACHINE LEARNING OVERVIEW

# After completing this section, you will be able to:

# Explain the difference between the outcome and the features.
# Explain when to use classification and when to use prediction.
# Explain the importance of prevalence.
# Explain the difference between sensitivity and specifity.

# 1.1. INTRODUCTION TO MACHINE LEARNING

# NOTATION

# X1,...,Xp denote the features, Y denotes the outcomes, Y_hat denotes the
# predictions.

# Machine learning prediction tasks can be divided into categorical and
# continuous outcomes. We refer to these as classification and prediction,
# respectively.

# AN EXAMPLE

# Yi = an outcome for observation or index i.

# We use boldface for X_i to distinguish the vector of predictors from the
# individual predictors Xi,1,...,Xi,784.

# When referring to an arbitrary set of features and outcomes, we drop the index
# i and use Y and bold X.

# Uppercase is used to refer to variables because we think of predictors as
# random variables.

# Lowercase is used to denote observed values. For example X = x.

# COMPREHENSION CHECK: INTRODUCTION TO MACHINE LEARNING

# Q1

# True or False: A key feature of machine learning is that the algorithms are
# built with data.

# True [X]
# False

# Q2

# True or False: In supervised machine learning, we build algorithms that take
# feature values (X) and train a model using known outcomes (Y) that is then
# used to predict outcomes when presented with features without known outcomes.

# True [X]
# False

## SECTION 2: MACHINE LEARNING BASICS

# In the Machine Learning Basics section, you will learn the basics of machine
# learning.

# After completing this section, you will be able to:

# Start to use the caret package.
# Construct and interpret a confusion matrix.
# Use conditional probabilities in the context of machine learning.

# 2.1. BASICS OF EVALUATING MACHINE LEARNING ALGORITHMS

# CARET PACKAGE, TRAINING AND TEST SETS, AND OVERALL ACCURACY

# Note: the set.seed() function is used to obtain reproducible results. If you
# have R 3.6 or later, please used the sample.kind = "Rounding" argument whenever
# you set the seed for this course.

# To mimic the ultimate evaluation process, we randomly split our data in two -
# a training set and a test set - and act as if we don't know the outcome of the
# test set. We develop algorithms using only the training set; the test set is
# used only for evaluation.

# The createDataPartition() function from the caret package can be used to
# generate indexes for randomly splitting data.

# Note: contrary to what the documentation says, this course will use the argument
# p as the percentage of data that goes to testing. The indexes made from
# createDataPartition() should be used to create the test set. Indexes should be
# created on the outcome and not a predictor.

# The simplest evaluation metric for categorical outcomes is overall accuracy:
# the proportion of cases that were correctly predicted in the test set.

library(tidyverse)
library(caret)
library(dslabs)
data(heights)

# define the outcome and predictors
y <- heights$sex
x <- heights$height

# generate training and test sets
set.seed(2, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index,]
train_set <- heights[-test_index,]

# guess the outcome
y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE)
y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE) %>%
  factor(levels = levels(test_set$sex))

# compute accuracy
mean(y_hat == test_set$sex)
heights %>% group_by(sex) %>% summarize(mean(height), sd(height))
y_hat <- ifelse(x > 62, "Male", "Female") %>% factor(levels = levels(test_set$sex))
mean(y == y_hat)

# examine the accuracy of 10 cutoffs
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x , "Male", "Female") %>%
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})
data.frame(cutoff, accuracy) %>%
  ggplot(aes(cutoff, accuracy)) +
  geom_point() +
  geom_line()
max(accuracy)

best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff

y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>%
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
mean(y_hat == test_set$sex)

# COMPREHENSION CHECK: BASICS ON EVALUATING MACHINE LEARNING ALGORITHMS

# Q1

# For each of the following, indicate whether the outcome is continuous or
# categorical.

# Digit reader # Categorical
# Height # Continuous
# Spam filter # Categorical
# Stock prices # Continuous
# Sex # Categorical

# Q2

# How many features are available to us for prediction in the mnist dataset?
mnist <- read_mnist()
ncol(mnist$train$images) # 784

# CONFUSION MATRIX

# Overall accuracy can sometimes be a deceptive measure because of unbalanced
# classes.

# A general improvement to using overall accuracy is to study sensitivity and
# specitivity separately. Sensitivity, also known as the true positive rate or
# recall, is the proportion of actual positive outcomes correctly identified
# as such. Specificity, also known as the true negative rate, is the proportion
# of actual negative outcomes that are correctly identified as such.

# A confusion matrix tabulates each combination of prediction and actual value.
# You can create a confusion matrix in R using the table() function or the
# confusionMatrix() function from the caret package.

# tabulate each combination of prediction and actual value
table(predicted = y_hat, actual = test_set$sex)
test_set %>%
  mutate(y_hat = y_hat) %>%
  group_by(sex) %>%
  summarize(accuracy = mean(y_hat == sex))
prev <- mean(y == "Male")

confusionMatrix(data = y_hat, reference = test_set$sex)

# BALANCED ACCURACY AND F1 SCORE

# For optimization purposes, sometimes it is more useful to have a one number
# summary than studying both specifity and sensitivity. One preferred metric is
# balanced accuracy. Because specificity and sensitivity are rates, it is more
# appropriate to compute the harmonic average. In fact, the F1-score, a widely
# used one-number summary, is the harmonic average of precision and recall.

# Depending on the context, some type of errors are more costly than others. The
# F1-score can be adapted to weigh specificity and sensitivity differently.

# You can compute the F1-score using the F_meas() function in the caret package.

# maximize F-score
cutoff <- seq(61, 70)
F_1 <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>%
    factor(levels = levels(test_set$sex))
  F_meas(data = y_hat, reference = factor(train_set$sex))
})

data.frame(cutoff, F_1) %>%
  ggplot(aes(cutoff, F_1)) +
  geom_point() +
  geom_line()

max(F_1)

best_cutoff <- cutoff[which.max(F_1)]
best_cutoff

y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>%
  factor(levels = levels(test_set$sex))
sensitivity(data = y_hat, reference = test_set$sex)
specificity(data = y_hat, reference = test_set$sex)

# PREVALENCE MATTERS IN PRACTICE

# A machine learning algorithm with very high sensitivity and specificity may
# not be useful in practice when prevalence is close to either 0 or 1. For example,
# if you develop an algorithm for disease diagnosis with very high sensitivity, but
# the prevalence of the disease is pretty low, then the precision of your algorithm
# is probabily very low based on Bayes' theorem.

# ROC AND PRECISION-RECALL CURVES

# A very common approach to evaluating accuracy and F1-score is to compare them
# graphically by plotting both. A widely used plot that does this is the receiver
# operating characteristic (ROC) curve. The ROC curve plots sensitivity (TPR) versus
# 1 - specificity or the false positive rate (FPR).

# However, ROC curves have one weakness and it is that neither of the measures
# plotted depend on prevalence. In cases in which prevalence matters, we may instead
# make a precision-recall plot, which has a similar idea with ROC curve.

p <- 0.9
n <- length(test_index)
y_hat <- sample(c("Male", "Female"), n, replace = TRUE, prob = c(p, 1 - p)) %>%
  factor(levels = levels(test_set$sex))
mean(y_hat == test_set$sex)

# ROC curve
probs <- seq(0, 1, length.out = 10)
guessing <- map_df(probs, function(p){
  y_hat <-
    sample(c("Male", "Female"), n, replace = TRUE, prob = c(p, 1 - p)) %>%
    factor(levels = c("Female", "Male"))
  list(method = "Guessing",
       FPR = 1 - specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
})
guessing %>% qplot(FPR, TPR, data = ., xlab = "1 - Specificity", ylab = "Sensitivity")

cutoffs <- c(50, seq(60, 75), 80)
height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>%
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       FPR = 1 - specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
})

# plot your curves together
bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(FPR, TPR, color = method)) +
  geom_line() +
  geom_point() +
  xlab("1 - Specificity") +
  ylab("Sensitivity")

library(ggrepel)
map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>%
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       cutoff = x,
       FPR = 1 - specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
}) %>%
  ggplot(aes(FPR, TPR, label = cutoff)) +
  geom_line() +
  geom_point() +
  geom_text_repel(nudge_x = 0.01, nudge_y = -0.01)

# plot precision against recall
guessing <- map_df(probs, function(p){
  y_hat <- sample(c("Male", "Female"), length(test_index), 
                  replace = TRUE, prob=c(p, 1-p)) %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Guess",
       recall = sensitivity(y_hat, test_set$sex),
       precision = precision(y_hat, test_set$sex))
})

height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       recall = sensitivity(y_hat, test_set$sex),
       precision = precision(y_hat, test_set$sex))
})

bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(recall, precision, color = method)) +
  geom_line() +
  geom_point()

guessing <- map_df(probs, function(p){
  y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE,
                    prob = c(p, 1 - p)) %>%
    factor(levels = c("Male", "Female"))
  list(method = "Guess",
       recall = sensitivity(y_hat, relevel(test_set$sex, "Male", "Female")),
       precision = precision(y_hat, relevel(test_set$sex, "Male", "Female")))
})

height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>%
    factor(levels = c("Male", "Female"))
  list(method = "Height cutoff",
       recall = sensitivity(y_hat, relevel(test_set$sex, "Male", "Female")),
       precision = precision(y_hat, relevel(test_set$sex, "Male", "Female")))
})
bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(recall, precision, color = method)) +
  geom_point() +
  geom_line()

# COMPREHENSION CHECK: PRACTICE WITH MACHINE LEARNING, PART 1

# The following questions all ask you to work with the dataset described below.

# The reported_heights and heights datasets were collected from three classes
# taught in the Departments of Computer Science and Biostatistics, as well as
# remotely through the Extension School. On 2016-01-25 at 8:15 AM, during one
# of the lectures, the instructors asked students to fill in the sex and height
# questionnaire that populated the reported_heights dataset. The online students
# filled out the survey during the next few days, after the lecture was posted
# online. We can use this insight to define a variable which we can call type,
# to denote the type of student, inclass or online.

# The code below sets up the dataset for you to analyze in the following exercises:
library(dslabs)
library(dplyr)
library(lubridate)
data(reported_heights)

dat <- mutate(reported_heights, date_time = ymd_hms(time_stamp)) %>%
  filter(date_time >= make_date(2016, 01, 25) & 
           date_time < make_date(2016, 02, 1)) %>%
  mutate(type = ifelse(day(date_time) == 25 & hour(date_time) == 8 &
                         between(minute(date_time), 15, 30), "inclass","online")) %>%
  select(sex, type)

y <- factor(dat$sex, c("Female", "Male"))
x <- dat$type

# Q1

# The type column of dat indicates whether students took classes in person ("inclass")
# or online ("online"). What proportion of the inclass group is female? What
# proportion of the online group is female?

dat %>% group_by(type) %>% summarize(mean(sex == "Female"))

# In class - 0.667

# Online - 0.378

# Q2

# In the course videos, height cutoffs were used to predict sex. Instead of height,
# use the type variable to predict sex. Assume that for each class type the
# students are either all male or all female, based on the most prevalent sex in
# each class type you calculated in Q1. Report the accuracy of your prediction of
# sex based on type. You do not need to split the data into training and test sets.

y_hat <- ifelse(dat$type == "inclass", "Female", "Male") %>%
  factor(levels = c("Female", "Male"))
mean(y_hat == y)

# Q3

# Write a line of code using the table() function to show the confusion matrix
# between y_hat and y. Use the exact format function(a, b) for your answer and do
# not name the columns and rows. Your answer should have exactly one space.

table(y_hat, y)

# Q4

# What is the sensitivity of this prediction? You can use the sensitivity()
# function from the caret package.

sensitivity(y_hat, y)

# Q5

# What is the specificity of this prediction? You can use the specificity()
# function from the caret package.

specificity(y_hat, y)

# Q6

# What is the prevalence (% of females) in the dat dataset defined above?

mean(y == "Female")

# COMPREHENSION CHECK: PRACTICE WITH MACHINE LEARNING, PART 2

# We will practice building a machine learning algorithm using a new dataset, iris,
# that provides multiple predictors for us to use to train. To start, we will
# remove the setosa species and we will focus on the versicolor and virginica iris
# species using the following code:
library(caret)
data(iris)
iris <- iris[-which(iris$Species == "setosa"),]
y <- iris$Species

# Q7

# First let us create an even split of the data into train and test partitions
# using createDataPartition() from the caret package. The code with a missing
# line is given below:

set.seed(2, sample.kind = "Rounding")
# line of code
test <- iris[test_index,]
train <- iris[-test_index,]

# Which code should be used in place of # line of code above?

# 1
test_index <- createDataPartition(y, times = 1, p = 0.5)
# 2
test_index <- sample(2, length(y), replace = FALSE)
# 3
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE) # [X]
# 4
test_index <- rep(1, length(y))

set.seed(2, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]

# Q8

# Next we will figure out the singular feature in the dataset that yields the
# greatest overall accuracy when predicting species. You can use the code from the
# introduction and from Q7 to start your analysis.

# Using only the train iris dataset, for each feature, perform a simple search to
# find the cutoff that produces the highest accuracy, predicting virginica if
# greater than the cutoff and versicolor otherwise. Use the seq function over the
# range of each feature by intervals of 0.1 for this search.
library(tidyverse)

# Which feature produces the highest accuracy?

# Sepal.Length
# Sepal.Width
# Petal.Length [X] 0.96
# Petal.Width

# Sepal.Length

min(train$Sepal.Length) # 5
max(train$Sepal.Length) # 7.9

cutoff <- seq(5, 7.9, 0.1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Sepal.Length > x , "virginica", "versicolor") %>%
    factor(levels = levels(test$Species))
  mean(y_hat == train$Species)
})
data.frame(cutoff, accuracy) %>%
  ggplot(aes(cutoff, accuracy)) +
  geom_point() +
  geom_line()
max(accuracy) # 0.7

# Sepal.Width

min(train$Sepal.Width) # 2
max(train$Sepal.Width) # 3.8

cutoff <- seq(2, 3.8, 0.1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Sepal.Width > x , "virginica", "versicolor") %>%
    factor(levels = levels(test$Species))
  mean(y_hat == train$Species)
})
data.frame(cutoff, accuracy) %>%
  ggplot(aes(cutoff, accuracy)) +
  geom_point() +
  geom_line()
max(accuracy) # 0.62

# Petal.Length

min(train$Petal.Length) # 3
max(train$Petal.Length) # 6.9

cutoff <- seq(3, 6.9, 0.1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Petal.Length > x , "virginica", "versicolor") %>%
    factor(levels = levels(test$Species))
  mean(y_hat == train$Species)
})
data.frame(cutoff, accuracy) %>%
  ggplot(aes(cutoff, accuracy)) +
  geom_point() +
  geom_line()
max(accuracy) # 0.96

# Petal.Width

min(train$Petal.Width) # 1
max(train$Petal.Length) # 6.9

cutoff <- seq(1, 6.9, 0.1)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train$Petal.Width > x , "virginica", "versicolor") %>%
    factor(levels = levels(test$Species))
  mean(y_hat == train$Species)
})
data.frame(cutoff, accuracy) %>%
  ggplot(aes(cutoff, accuracy)) +
  geom_point() +
  geom_line()
max(accuracy) # 0.94

# More efficient solution
foo <- function(x){
  rangedValues <- seq(range(x)[1], range(x)[2], by = 0.1)
  sapply(rangedValues, function(i){
    y_hat <- ifelse(x > i, "virginica", "versicolor")
    mean(y_hat == train$Species)
  })
}
predictions <- apply(train[,-5], 2, foo)
sapply(predictions, max)

# Q9

# For the feature selected in Q8, use the smart cutoff value from the training
# data to calculate the overall accuracy in the test data. What is the overall
# accuracy?

best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff

y_hat <- ifelse(test$Petal.Length > best_cutoff, "virginica", "versicolor") %>%
  factor(levels = levels(test$Species))
mean(y_hat == test$Species) # 0.9

# Optimized solution
predictions <- foo(train[,3])
rangedValues <- seq(range(train[,3])[1], range(train[,3])[2], by = 0.1)
cutoffs <- rangedValues[which(predictions == max(predictions))]

y_hat <- ifelse(test[,3] > cutoffs[1], "virginica", "versicolor")
mean(y_hat == test$Species)

# Q10

# Notice that we had an overall accuracy greater than 96% in the training data,
# but the overall accuracy was lower in the test data. This can happen often if
# we overtrain. In fact, it could be the case that a single feature is not the
# best choice. For example, a combination of features might be optimal. Using a
# single feature and optimizing the cutoff as we did on our training data can
# lead to overfitting.

# Given that we know the test data, we can treat it like we did our training data
# to see if the same feature with a different cutoff will optimize our predictions.
# Repeat the analysis in Q8 but this time using the test data instead of the
# training data.

# Which feature best optimizes our overall accuracy when using the test set?

# Sepal.Length
# Sepal.Width
# Petal.Length
# Petal.Width [X]

# More efficient solution
foo2 <- function(x){
  rangedValues <- seq(range(x)[1], range(x)[2], by = 0.1)
  sapply(rangedValues, function(i){
    y_hat <- ifelse(x > i, "virginica", "versicolor")
    mean(y_hat == test$Species)
  })
}
predictions2 <- apply(test[,-5], 2, foo2)
sapply(predictions2, max)

# Q11

# Now we will perform some exploratory data analysis on the data.
plot(iris, pch = 21, bg = iris$Species)

# Notice that Petal.Length and Petal.Width in combination could potentially be
# more information than either feature alone.

# Optimize the cutoffs for Petal.Length and Petal.Width separately in the train
# dataset by using the seq function with increments of 0.1. Then, report the
# overall accuracy when applied to the test dataset by creating a rule that
# predicts virgincia if Petal.Length is greater than the length cutoff OR
# Petal.Width is greater than the width cutoff, and versicolor otherwise.

# What is the overall accuracy for the test data now?

predictions_length <- foo(train[,3])
rangedValues_length <- seq(range(train[,3])[1], range(train[,3])[2], by = 0.1)
cutoffs_length <- rangedValues_length[which(predictions_length == 
                                              max(predictions_length))]
cutoffs_length

predictions_width <- foo(train[,4])
rangedValues_width <- seq(range(train[,4])[1], range(train[,4])[2], by = 0.1)
cutoffs_width <- rangedValues_width[which(predictions_width ==
                                            max(predictions_width))]

cutoffs_width

y_hat <- ifelse(test[,3] > cutoffs_length[1] | test[,4] > cutoffs_width[1],
                "virginica", "versicolor")
mean(y_hat == test$Species) # 0.88

# 2.2. CONDITIONAL PROBABILITIES

# CONDITIONAL PROBABILITIES

# Conditional probabilities for each class:

# pk(x) = Pr(Y = k | X = x), for k = 1,...,K

# In machine learning, this is referred to as Bayes' Rule. This is a theoretical
# rule because in practice we don't know p(x). Having a good estimate of the p(x)
# will suffice for us to build optimal prediction models, since we can control the
# balance between specificity and sensitivity however we wish. In fact, estimating
# these conditional probabilities can be thought of as the main challenge of
# machine learning.

# CONDITIONAL EXPECTATIONS AND LOSS FUNCTION

# Due to the connection between conditional probabilities and conditional
# expectations:

# pk(x) = Pr(Y = k | X = x), for k = 1,...,K

# we often only use the expectation to denote both the conditional probability and
# the conditional expectation.

# For continuous outcomes, we define a loss function to evaluate the model. The
# most commonly used one is MSE (Mean Squared Error). The reason why we care about
# the conditional expectation in machine learning is that the expected value
# minimizes the MSE:

# Y_hat = E(Y|X = x) minimizes E{(Y_hat - Y)^2 | X = x}

# Due to this property, a succint description of the main task of machine learning
# is that we use data to estimate for any set of features. The main way in which
# competing machine learning algorithms differ is their approach to estimating
# this expectation.

# COMPREHENSION CHECK: CONDITIONAL PROBABILITIES, PART 1

# Q1

# In a previous module, we covered Bayes' theorem and the Bayesian paradigm.
# Conditional probabilities are a fundamental part of this previous covered rule.

# P(A|B) = P(B|A) * P(A) / P(B)

# We first review a simple example to go over conditional probabilities.

# Assume a patient comes into the doctor's office to test whether they have a 
# particular disease.

# The test is positive 85% of the time when tested on a patient with the disease
# (high sensitivity): P(test+ | disease) = 0.85

# The test is negative 90% of the time when tested on a healthy patient (high
# specificity): P(test- | healthy) = 0.90

# The disease is prevalent in about 2% of the community: P(disease) = 0.02

# Using Bayes' theorem, calculate the probability that you have the disease if
# the test is positive.

# P(disease | test+) = P(test+ | disease) * P(disease) / P(test+)
# P(disease | test+) = 0.85 * 0.02 / 0.1

# Hint: P(test+|healthy) = 1 - P(test-|healthy) = 0.1

# P(disease | test+) = P(test+ | disease) * P(disease) / P(test+) =

# (P(test+ | disease) * P(disease)) /
# (P(test+ | disease) * P(disease) + P(test+ | healthy) * P(healthy)) =

# 0.85 * 0.02 / (0.85 * 0.02 + 0.1 * 0.98)

0.85 * 0.02 / (0.85 * 0.02 + 0.1 * 0.98) # 0.1478

# We have a population of 1 million individuals with the following conditional
# probabilities as described below:

# The test is positive 85% of the time when tested on a patient with the disease
# (high sensitivity): P(test+ | disease) = 0.85

# The test is negative 90% of the time when tested on a healthy patient (high
# specificity): P(test- | healthy) = 0.90

# The disease is prevalent in about 2% of the community: P(disease) = 0.02

# Here is some sample code to get you started.
set.seed(1, sample.kind = "Rounding")
disease <- sample(c(0, 1), size = 1e6, replace = TRUE, prob = c(0.98, 0.02))
test <- rep(NA, 1e6)
test[disease == 0] <- sample(c(0, 1), size = sum(disease == 0), replace = TRUE,
                             prob = c(0.90, 0.10))
test[disease == 1] <- sample(c(0, 1), size = sum(disease == 1), replace = TRUE,
                             prob = c(0.15, 0.85))

# Q2

# What is the probability that a test is positive?

mean(test)

# Q3 

# What is the probability that an individual has the disease if the test is
# negative?

mean(disease[test == 0])

# Q4

# What is the probability that you have the disease if the test is positive?

mean(disease[test == 1] == 1)

# Q5

# Compare the prevalence of disease in people who test positive to the overall
# prevalence of the disease.

# If a patient's test is positive, by how many times does that increase their
# risk of having the disease?

mean(disease[test == 1]) / mean(disease)

# Q6

# We are now going to write code to compute conditional probabilities for being
# male in the heights dataset. Round the heights to the closest inch. Plot the
# estimated conditional probability: P(x) = Pr(Male | height = x) for each x.

# Part of the code is provided here:
library(dslabs)
data("heights")
# MISSING CODE
  qplot(height, p, data = .)

# Which of the following blocks of code can be used to replace # MISSING CODE to
# make the correct plot?

# 1
heights %>%
  group_by(height) %>%
  summarize(p = mean(sex == "Male")) %>%
  qplot(height, p, data = .)

# 2
heights %>%
  mutate(height = round(height)) %>%
  group_by(height) %>%
  summarize(p = mean(sex == "Female")) %>%
  qplot(height, p, data = .)

# 3
heights %>%
  mutate(height = round(height)) %>%
  summarize(p = mean(sex == "Male")) %>%
  qplot(height, p, data = .)

# 4
heights %>%
  mutate(height = round(height)) %>%
  group_by(height) %>%
  summarize(p = mean(sex == "Male")) %>%
  qplot(height, p, data = .) # [X]

# Q7

# In the plot we just made in Q6 we see high variability for low values of height.
# This is because we have few data points. This time use the quantile 0.1, 0.2,...,
# 0.9 and the cut() function to assure each group has the same number of points.
# Note that for any numeric vector x, you can create groups based on quantiles like
# this: cut(quantile(x, seq(0, 1, 0.1)), include.lowest = TRUE).

# Part of the code is provided here:
ps <- seq(0, 1, 0.1)
heights %>%
  # MISSING CODE
  group_by(g) %>%
  summarize(p = mean(sex == "Male"), height = mean(height)) %>%
  qplot(height, p, data = .)

# Which of the following lines of code can be used to replace # MISSING CODE to
# make the correct plot?

# 1
# mutate(g = cut(male, quantile(height, ps), include.lowest = TRUE)) %>%

# 2
# mutate(g = cut(height, quantile(height, ps), include.lowest = TRUE)) %>% [X]

# 3
# mutate(g = cut(female, quantile(height, ps), include.lowest = TRUE)) %>%

# 4
# mutate(g = cut(height, quantile(height, ps))) %>%

heights %>%
  mutate(g = cut(height, quantile(height, ps), include.lowest = TRUE)) %>%
  group_by(g) %>%
  summarize(p = mean(sex == "Male"), height = mean(height)) %>%
  qplot(height, p, data = .)

# Q8

# You can generate data from a bivariate normal distribution using the MASS package
# using the following code:
Sigma <- 9*matrix(c(1, 0.5, 0.5, 1), 2, 2)
dat <- MASS::mvrnorm(n = 10000, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

# And you can make a quick plot using plot(dat).
plot(dat)

# Using an approach similar to that used in the previous exercise, let's estimate
# the conditional expectations and make a plot. Part of the code has again been
# provided for you:

ps <- seq(0, 1, 0.1)
dat %>%
  # MISSING CODE
  qplot(x, y, data = .)

# Which of the following blocks of code can be used to replace # MISSING CODE to
# make the correct plot?

# 1
# mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
# group_by(g) %>%
# summarize(y = mean(y), x = mean(x)) %>% [X]

# 2
# mutate(g = cut(x, quantile(x, ps))) %>%
# group_by(g) %>%
# summarize(y = mean(y), x = mean(x)) %>%

# 3
# mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
# summarize(y = mean(y), x = mean(x)) %>%

# 4
# mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
# group_by(g) %>%
# summarize(y = (y), x = (x)) %>%

dat %>%
  mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
  group_by(g) %>%
  summarize(y = mean(y), x = mean(x)) %>%
  qplot(x, y, data = .)

## SECTION 3: LINEAR REGRESSION FOR PREDICTION, SMOOTHING, AND WORKING
# WITH MATRICES

# LINEAR REGRESSION FOR PREDICTION, SMOOTHING, AND WORKING WITH MATRICES OVERVIEW

# In this section, you will learn why linear regression is a useful baseline
# approach but is often insufficiently flexible for more complex analyses, how to
# smooth noisy data, and how to use matrices for machine learning.

# After completing this section, you will be able to:

# Use linear regression for prediction as a baseline approach.
# Use logistic regression for categorical data.
# Detect trends in noisy data using smoothing (also known as curve fitting or
# low pass filtering).
# Convert predictors to matrices and outcomes to vectors when all predictors are
# numeric (or can be converted to numerics in a meaningful way).
# Perform basic matrix algebra calculations.

# 3.1. LINEAR REGRESSION FOR PREDICTION

# Linear regression can be considered a machine learning algorithm. Although it
# can be too rigid to be useful, it works rather well for some challenges. It also
# serves as a baseline approach: if you can't beat it with a more complex approach,
# your probably want to stick to linear regression.

library(tidyverse)
library(HistData)

galton_heights <- GaltonFamilies %>%
  filter(childNum == 1 & gender == "male") %>%
  select(father, childHeight) %>%
  rename(son = childHeight)

library(caret)
y <- galton_heights$son
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)

train_set <- galton_heights %>% slice(-test_index)
test_set <- galton_heights %>% slice(test_index)

avg <- mean(train_set$son)
avg

# fit linear regression model
fit <- lm(son ~ father, data = train_set)
fit$coef

y_hat <- fit$coef[1] + fit$coef[2] * test_set$father
mean((y_hat - test_set$son)^2)

# PREDICT FUNCTION

# The predict() function takes a fitted object from functions such as lm() and
# glm() and a data frame with the new predictors for which to predict. We can use
# predict() like this:
y_hat <- predict(fit, test_set)

# predict() is a generic function in R that calls on other functions depending on
# what kind of object it receives. To learn more about the specifics, you can
# read the help files using code like this:
?predict.lm # or ?predict.glm

y_hat <- predict(fit, test_set)
mean((y_hat - test_set$son)^2)

# COMPREHENSION CHECK: LINEAR REGRESSION

# Q1

# Create a dataset using the following code:
library(tidyverse)
library(caret)

set.seed(1, sample.kind = "Rounding")
n <- 100
Sigma <- 9 * matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

# We will build 100 linear models using the data above and calculate the mean and
# standard deviation of the combined models. First, set the seed to 1 again (make
# sure to use sample.kind = "Rounding" if your R version is 3.6 or later). Then,
# within a replicate() loop, (1) partition the dataset into test and training sets
# with p = 0.5 and using dat$y to generate your indices, (2) train a linear model
# predicting y from x, (3) generate predictions on the test set, and (4) calculate
# the RMSE of that model. Then, report the mean and standard deviation (SD) or the
# RMSEs from all 100 models.

set.seed(1, sample.kind = "Rounding")
rmse <- replicate(n, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, test_set)
  sqrt(mean((y_hat - test_set$y)^2))
})
mean(rmse) # 2.488
sd(rmse) # 0.124

# Q2

# Now we will repeat the exercise above but using larger datasets. Write a function
# that takes a size n, then (1) builds a dataset using the code provided at the
# top of Q1 but with n observations instead of 100 and without the set.seed(1), (2)
# runs the replicate() loop that you write to answer Q1, which builds 100 linear
# models and returns a vector of RMSEs, and (3) calculates the mean and standard
# deviation of the 100 RMSEs.

# Set the seed to 1 and then use sapply() or map() to apply your new function to
# n <- c(100, 500, 1000, 5000, 10000).

set.seed(1, sample.kind = "Rounding")
results <- function(n){
  dat <- MASS::mvrnorm(n, c(69, 69), Sigma) %>%
    data.frame() %>% setNames(c("x", "y"))
  rmse <- replicate(n = 100, {
    test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
    train_set <- dat %>% slice(-test_index)
    test_set <- dat %>% slice(test_index)
    fit <- lm(y ~ x, data = train_set)
    y_hat <- predict(fit, test_set)
    sqrt(mean((y_hat - test_set$y)^2))
  })
  rmse %>% data.frame() %>% summarize(mean = mean(rmse), sd = sd(rmse))  
}

n <- c(100, 500, 1000, 5000, 10000)
sapply(n, results)

# Optimized solution
set.seed(1, sample.kind = "Rounding")
n <- c(100, 500, 1000, 5000, 10000)
res <- sapply(n, function(n){
  Sigma <- 9 * matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
  dat <- MASS::mvrnorm(n, c(69, 69), Sigma) %>%
    data.frame() %>% setNames(c("x", "y"))
  rmse <- replicate(100, {
    test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
    train_set <- dat %>% slice(-test_index)
    test_set <- dat %>% slice(test_index)
    fit <- lm(y ~ x, data = train_set)
    y_hat <- predict(fit, newdata = test_set)
    sqrt(mean((y_hat - test_set$y)^2))
  })
  c(avg = mean(rmse), sd = sd(rmse))
})
res

# Q3

# What happens to the RMSE as the size of the dataset becomes larger?

# On average, the RMSE does not change much as n gets larger, but the variability
# of the RMSE decreases. [X]

# Because of the law of large numbers the RMSE decreases; more data means more
# precise estimates.

# n = 10000 is not sufficiently large. To see a decrease in the RMSE we would need
# to make it larger.

# The RMSE is not a random variable.

# Q4

# Now repeat the exercise from Q1, this time making the correlation between x and
# y larger, as in the following code:

set.seed(1, sample.kind = "Rounding")
n <- 100
Sigma <- 9 * matrix(c(1.0, 0.95, 0.95, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))
set.seed(1, sample.kind = "Rounding")
rmse <- replicate(n, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, test_set)
  sqrt(mean((y_hat - test_set$y)^2))
})
mean(rmse) # 0.909
sd(rmse) # 0.062

# Q5

# Which of the following best explains why the RMSE in question 4 is so much lower
# than the RMSE in question 1?

# It is just luck. If we do it again, it will be larger.

# The central limit theorem tells us that the RMSE is normal.

# When we increase the correlation between x and y, x has more predictive power and
# thus provides a better estimate of y. [X]

# These are both examples of regression so the RMSE has to be the same.

# Q6

# Create a dataset using the following code:
set.seed(1, sample.kind = "Rounding")
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.25, 0.75, 0.25, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

# Note that y is correlated with both x_1 and x_2 but the two predictors are
# independent of each other, as seen by:
cor(dat)

# Set the seed to 1, then use the caret package to partition into test and
# training sets with p = 0.5. Compare the RMSE when using just x_1, just x_2 and
# both x_1 and x_2. Train a single linear model for each (not 100 like in the
# previous questions).

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)
fit_x1 <- lm(y ~ x_1, data = train_set)
fit_x2 <- lm(y ~ x_2, data = train_set)
fit_x1_x2 <- lm(y ~ x_1 + x_2, data = train_set)
y_hat_x1 <- predict(fit_x1, test_set)
y_hat_x2 <- predict(fit_x2, test_set)
y_hat_x1_x2 <- predict(fit_x1_x2, test_set)
res <- c(x_1 = sqrt(mean((y_hat_x1 - test_set$y)^2)),
         x_2 = sqrt(mean((y_hat_x2 - test_set$y)^2)),
         x_1_x_2 = sqrt(mean((y_hat_x1_x2 - test_set$y)^2)))
res

# Which of the three models performs the best (has the lowest RMSE)?

# x_1
# x_2
# x_1 and x_2 [X]

# Q7

# Report the lowest RMSE of the three models tested in Q6.

# Q8

# Repeat the exercise from Q6 but now create an example in which x_1 and x_2
# are highly correlated.
set.seed(1, sample.kind = "Rounding")
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.95, 0.75, 0.95, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

# Set the seed to 1, then use the caret package to partition into a test and training
# set of equal size. Compare the RMSE when using just x_1, just x_2, and both x_1
# and x_2.

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)
fit_x1 <- lm(y ~ x_1, data = train_set)
fit_x2 <- lm(y ~ x_2, data = train_set)
fit_x1_x2 <- lm(y ~ x_1 + x_2, data = train_set)
y_hat_x1 <- predict(fit_x1, test_set)
y_hat_x2 <- predict(fit_x2, test_set)
y_hat_x1_x2 <- predict(fit_x1_x2, test_set)
res <- c(x_1 = sqrt(mean((y_hat_x1 - test_set$y)^2)),
         x_2 = sqrt(mean((y_hat_x2 - test_set$y)^2)),
         x_1_x_2 = sqrt(mean((y_hat_x1_x2 - test_set$y)^2)))
res

# Compare the results from Q6 and Q8. What can you conclude?

# Unless we include all predictors we have no predictive power.

# Adding extra predictors improves RMSE regardless of whether the added predictors
# are correlated with other predictors or not.

# Adding extra predictors results in overfitting.

# Adding extra predictors can improve RMSE substantially, but not when the added
# predictors are highly correlated with other predictors. [X]

# REGRESSION FOR A CATEGORICAL OUTCOME

# The regression approach can be extended to categorical data. For example, we
# can try regression to estimate the conditional probability:

# p(x) = Pr(Y = 1 | X = x) = `beta`0 + `beta`1 * x

# Once we have estimates `beta`0 and `beta`1, we can obstain an actual prediction
# p(x). Then we can define a specific decision rule to form a prediction.
library(dslabs)
data("heights")
y <- heights$height

set.seed(2, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

train_set %>%
  mutate(x = round(height)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(sex == "Female")) %>%
  ggplot(aes(x, prop)) +
  geom_point()
lm_fit <- mutate(train_set, y = as.numeric(sex == "Female")) %>%
  lm(y ~ height, data = .)
p_hat <- predict(lm_fit, test_set)
y_hat <- ifelse(p_hat > 0.5, "Female", "Male") %>% factor()
confusionMatrix(y_hat, test_set$sex)$overall["Accuracy"]

# LOGISTIC REGRESSION

# Logistic regression is an extension of linear regression that assures that
# the estimate of conditional probability Pr(Y = 1 | X = x) is between 0 and 1.
# This approach makes use of the logistic transformation:

# g(p) = log(p / (1 - p))

# With logistic regression, we model the conditional probability directly with:

# g{Pr(Y = 1 | X = x)} = `beta`0 + `beta`1 * x

# Note that with this model, we can no longer use least squares. Instead we compute
# the maximum likelihood estimate (MLE).

# In R, we can fit the logistic regression model with the function glm() (generalized
# linear models). If we want to compute the conditional probabilities, we want 
# type = "response" since the default is to return the logistic transformed values.

heights %>%
  mutate(x = round(height)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(sex == "Female")) %>%
  ggplot(aes(x, prop)) +
  geom_point() +
  geom_abline(intercept = lm_fit$coef[1], slope = lm_fit$coef[2])

range(p_hat)

# fit logistic regression model
glm_fit <- train_set %>%
  mutate(y = as.numeric(sex == "Female")) %>%
  glm(y ~ height, data = ., family = "binomial")

p_hat_logit <- predict(glm_fit, newdata = test_set, type = "response")

tmp <- heights %>%
  mutate(x = round(height)) %>%
  group_by(x) %>%
  filter(n() >= 10) %>%
  summarize(prop = mean(sex == "Female"))
logistic_curve <- data.frame(x = seq(min(tmp$x), max(tmp$x))) %>%
  mutate(p_hat = plogis(glm_fit$coef[1] + glm_fit$coef[2] * x))
tmp %>%
  ggplot(aes(x, prop)) +
  geom_point() +
  geom_line(data = logistic_curve, mapping = aes(x, p_hat), lty = 2)

y_hat_logit <- ifelse(p_hat_logit > 0.5, "Female", "Male") %>% factor
confusionMatrix(y_hat_logit, test_set$sex)$overall["Accuracy"]

# CASE STUDY: 2 OR 7

# In this case study we apply logistic regression to classify whether a digit is
# a two or seven. We are interested in estimating a conditional probability that
# depends on two variables:

# g{p(x1,x2)}=g{Pr(Y=1|X1=x1,X2=x2)}=`beta`0+`bet`1*x1+`beta`2*x2

# Through this case, we know that logistic regression forces our estimates to be
# a plane and our boundary to be a line. This implies that a logistic regression
# approach has no change of capturing the non-linear nature of the true p(x1, x2).
# Therefore, we need other more flexible methods that permit other shapes.
mnist <- read_mnist()
is <- mnist_27$index_train[c(which.min(mnist_27$train$x_1),
                             which.max(mnist_27$train$x_1))]
titles <- c("smallest", "largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%
    mutate(label = titles[i],
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
tmp %>% ggplot(aes(Row, Column, fill = value)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) +
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5)

data("mnist_27")
mnist_27$train %>% ggplot(aes(x_1, x_2, color = y)) + geom_point()

is <- mnist_27$index_train[c(which.min(mnist_27$train$x_2), which.max(mnist_27$train$x_2))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%
    mutate(label=titles[i],
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
tmp %>% ggplot(aes(Row, Column, fill=value)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) +
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5)

fit_glm <- glm(y ~ x_1 + x_2, data = mnist_27$train, family = "binomial")
p_hat_glm <- predict(fit_glm, mnist_27$test)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, 7, 2))
confusionMatrix(data = y_hat_glm, reference = mnist_27$test$y)$overall["Accuracy"]

mnist_27$true_p %>% ggplot(aes(x_1, x_2, fill = p)) +
  geom_raster()

mnist_27$true_p %>% ggplot(aes(x_1, x_2, z = p, fill = p)) +
  geom_raster() +
  scale_fill_gradientn(colors = c("#F8766D", "white", "#00BFC4")) +
  stat_contour(breaks = c(0.5), color = "black")

p_hat <- predict(fit_glm, newdata = mnist_27$true_p)
mnist_27$true_p %>%
  mutate(p_hat = p_hat) %>%
  ggplot(aes(x_1, x_2, z = p_hat, fill = p_hat)) +
  geom_raster() +
  scale_fill_gradientn(colors = c("#F8766D", "white", "#00BFC4")) +
  stat_contour(breaks = c(0.5), color = "black")

p_hat <- predict(fit_glm, newdata = mnist_27$true_p)
mnist_27$true_p %>%
  mutate(p_hat = p_hat) %>%
  ggplot() +
  stat_contour(aes(x_1, x_2, z = p_hat), breaks = c(0.5), color = "black") +
  geom_point(mapping = aes(x_1, x_2, color = y), data = mnist_27$test)

# COMPREHENSION CHECK: LOGISTIC REGRESSION

# Q1

# Define a dataset using the following code
set.seed(2, sample.kind = "Rounding")
make_data <- function(n = 1000, p = 0.5,
                      mu_0 = 0, mu_1 = 2,
                      sigma_0 = 1, sigma_1 = 1){
  y <- rbinom(n, 1, p)
  f_0 <- rnorm(n, mu_0, sigma_0)
  f_1 <- rnorm(n, mu_1, sigma_1)
  x <- ifelse(y == 1, f_1, f_0)
  
  test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
  
  list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
       test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}
dat <- make_data()

# Note that we have defined a variable x that is predictive of a binary outcome y:
dat$train %>% ggplot(aes(x, color = y)) +geom_density()

# Set the seed to 1, then use the make_data() function defined above to generate
# 25 different datasets with mu_1 <- seq(0, 3, len = 25). Perform logistic
# regression on each of the 25 different datasets (predict 1 if p > 0.5) and plot
# accuracy (res in the figures) vs mu_1 (delta in the figures).

# Which is the correct plot? # 1

set.seed(1, sample.kind = "Rounding")
delta <- seq(0, 3, len = 25)
res <- sapply(delta, function(d){
  dat <- make_data(mu_1 = d)
  fit_glm <- dat$train %>% glm(y ~ x, family = "binomial", data = .)
  y_hat_glm <- ifelse(predict(fit_glm, dat$test) > 0.5, 1, 0) %>% 
    factor(levels = c(0, 1))
  mean(y_hat_glm == dat$test$y)
})
qplot(delta, res)

# 3.2. SMOOTHING

# INTRODUCTION TO SMOOTHING

# Smoothing is a very powerful technique used all across data analysis. It is
# designed to detect trends in the presence of noisy data in cases in which the
# shape of the trend is unknown.

# The concepts behind smoothing techniques are extremely useful in machine learning
# because conditional expectations/probabilities can be thought of as trends of
# unknown shapes that we need to estimate in the presence of uncertainty.
data("polls_2008")
qplot(day, margin, data = polls_2008)

# BIN SMOOTHING AND KERNELS

# The general idea of smoothing is to group data points into strata in which the
# value of f(x) can be assumed to be constant. We can make this assumption because
# we think f(x) changes slowly and, as a result, f(x) is almost constant in small
# windows of time.

# This assumption implies that a good estimate of f(x) is the average of the Yi
# values in the window. The estimate is:

# f_hat(x0) = 1 / N0 * sum(Yi)i`in`A0

# In smoothing, we call the size pf the inverval |x - x0| satisfying the particular
# condition the window size, bandwidth or span.

# bin smoothers
span <- 7
fit <- with(polls_2008, ksmooth(day, margin, x.points = day, kernel = "box",
                                bandwidth = span))
polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "gray") +
  geom_line(aes(day, smooth), color = "red")

# kernel
span <- 7
fit <- with(polls_2008, ksmooth(day, margin,  
                                x.points = day, kernel="normal", bandwidth = span))
polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = 0.5, color = "gray") +
  geom_line(aes(day, smooth), color = "red")

# LOCAL WEIGHTED REGRESSION (LOESS)

# A limitation of the bin smoothing approach is that we need small windows for the
# approximately constant assumptions to hold which may lead to imprecise estimates
# of f(x). Local weighted regression (loess) permits us to consider larger window
# sizes.

# One important difference between loess and bin smoother is that we assume the
# smooth function is locally linear in a window instead of a constant.

# The result of loess is a smoother fit than bin smoothing because we use larger
# sample sizes to estimate our local parameters.

polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth(color = "red", span = 0.15, method = "loess",
              method.args = list(degree = 1))

# COMPREHENSION CHECK: SMOOTHING

# Q1

# In the Wrangling course of this series, we used the following code to obtain
# mortality counts for Puerto Rico for 2015-2018:
library(tidyverse)
library(lubridate)
library(purrr)
library(pdftools)

fn <- system.file("extdata", "RD-Mortality-Report_2015-18-180531.pdf", package="dslabs")
dat <- map_df(str_split(pdf_text(fn), "\n"), function(s){
  s <- str_trim(s)
  header_index <- str_which(s, "2015")[1]
  tmp <- str_split(s[header_index], "\\s+", simplify = TRUE)
  month <- tmp[1]
  header <- tmp[-1]
  tail_index  <- str_which(s, "Total")
  n <- str_count(s, "\\d+")
  out <- c(1:header_index, which(n==1), which(n>=28), tail_index:length(s))
  s[-out] %>%
    str_remove_all("[^\\d\\s]") %>%
    str_trim() %>%
    str_split_fixed("\\s+", n = 6) %>%
    .[,1:5] %>%
    as_data_frame() %>% 
    setNames(c("day", header)) %>%
    mutate(month = month,
           day = as.numeric(day)) %>%
    gather(year, deaths, -c(day, month)) %>%
    mutate(deaths = as.numeric(deaths))
}) %>%
  mutate(month = recode(month, "JAN" = 1, "FEB" = 2, "MAR" = 3, "APR" = 4, "MAY" = 5, "JUN" = 6, 
                        "JUL" = 7, "AGO" = 8, "SEP" = 9, "OCT" = 10, "NOV" = 11, "DEC" = 12)) %>%
  mutate(date = make_date(year, month, day)) %>%
  dplyr::filter(date <= "2018-05-01")

# Use the loess() function to obtain a smooth estimate of the expected number of
# deaths as a function of date. Plot this resulting smooth function. Make the
# span about two months long and use degree = 1.

# Which of the following plots is correct?

span <- 60 / as.numeric(diff(range(dat$date)))
fit <- dat %>% mutate(x = as.numeric(date)) %>%
  loess(deaths ~ x, data = ., span = span, degree = 1)
dat %>% mutate(smooth = predict(fit, as.numeric(date))) %>%
  ggplot() +
  geom_point(aes(date, deaths)) +
  geom_line(aes(date, smooth), lwd = 2, col = "red")

# Q2

# Work with the same data as in Q1 to plot smooth estimates against day of the
# year, all on the same plot, but with different colors for each year.

# Which code produces the desired plot?

# 1
dat %>% 
  mutate(smooth = predict(fit), day = yday(date), year = as.character(year(date))) %>%
  ggplot(aes(day, smooth, col = year)) +
  geom_line(lwd = 2)

# 2
dat %>% 
  mutate(smooth = predict(fit, as.numeric(date)), day = mday(date), year = as.character(year(date))) %>%
  ggplot(aes(day, smooth, col = year)) +
  geom_line(lwd = 2)

# 3
dat %>% 
  mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
  ggplot(aes(day, smooth)) +
  geom_line(lwd = 2)

# 4
dat %>% 
  mutate(smooth = predict(fit, as.numeric(date)), day = yday(date), year = as.character(year(date))) %>%
  ggplot(aes(day, smooth, col = year)) +
  geom_line(lwd = 2) # [X]

# Q3

# Suppose we want to predict 2s and 7s in the mnist_27 dataset with just the second
# covariate. Can we do this? On first inspection it appears the data does not have
# much predictive power.

# In fact, if we fit a regular logistic regression coefficient for x_2 is not
# significant!

# This can be seen using this code:
library(broom)
mnist_27$train %>% glm(y ~ x_2, family = "binomial", data = .) %>% tidy()

# Plotting a scatterplot here is not useful since y is binary:
qplot(x_2, y, data = mnist_27$train)

# Fit a loess line to the data above and plot the results. What do you observe?

fit <- mnist_27$train %>% loess(as.numeric(y) ~ x_2, data = .)
mnist_27$train %>% mutate(smooth = predict(fit, x_2)) %>%
  ggplot() +
  geom_line(aes(x_2, smooth), lwd = 2, col = "red")

# There is no predictive power and the conditional probability is linear.
# There is no predictive power and the conditional probability is non-linear.
# There is predictive power and the conditional probability is linear.
# There is predictive power and the conditional probability is non-linear. [X]

# Optimized response
mnist_27$train %>%
  mutate(y = ifelse(y == "7", 1, 0)) %>%
  ggplot(aes(x_2, y)) +
  geom_smooth(method = "loess")

# 3.3. WORKING WITH MATRICES

# MATRICES

# The main reason for using matrices is that certain mathematical operations needed
# to develop efficient code can be performed using techniques from a branch of
# mathematics called linear algebra.

# Linear algebra and matrix notation are key elements of the language used in
# academic papers describing machine learning techniques.

library(tidyverse)
library(dslabs)
if(!exists("mnist")) mnist <- read_mnist()

class(mnist$train$images)

x <- mnist$train$images[1:1000,]
y <- mnist$train$labels[1:1000]

# MATRIX NOTATION

# In matrix algebra, we have three main types of objects: scalars, vectors, and
# matrices.

# Scalar: `alpha` = 1
# Vector: X1 = (X1,1 ... XN,1)T
# Matrix: X = [X1X2] = ((X1,1 ... XN,1)T (X1,2 ... XN,2)T

# In R, we can extract the dimension of matrix with the function dim(). We can
# convert a vector into a matrix using the function as.matrix().

length(x[,1])
x_1 <- 1:5
x_2 <- 6:10
cbind(x_1, x_2)
dim(x)
dim(x_1)
dim(as.matrix(x_1))
dim(x)

# CONVERTING A VECTOR TO MATRIX

# In R, we can convert a vector into a matrix with the matrix() function. The
# matrix is filled in by column, but we can fill by row using the by row argument.
# The function t() can be used to directly transpose a matrix.

# Note that the matrix function recycles values in the vector without warning if
# the product of columns and rows does not match the length of the vector.
my_vector <- 1:15

# fill the matrix by column
mat <- matrix(my_vector, 5, 3)
mat

# fill by row
mat_t <- matrix(my_vector, 3, 5, byrow = TRUE)
mat_t
identical(t(mat), mat_t)
matrix(my_vector, 5, 5)
grid <- matrix(x[3,], 28, 28)
image(1:28, 1:28, grid)

# flip the image back
image(1:28, 1:28, grid[,28:1])

# ROW AND COLUMN SUMMARIES AND APPLY

# The function rowSums() computes the sum of each row.

# The function rowMeans() computes the average of each row.

# We can compute the column sums and averages using the functions colSums() and
# colMeans().

# The matrixStats package adds functions that perform operations on each row or
# column very efficiently, including the functions rowSds() and colSds().

# The apply() function lets you apply any function to a matrix. The first argument
# is the matrix, the second is the dimension (1 for rows, 2 for columns), and the
# third is the function.

sums <- rowSums(x)
avg <- rowMeans(x)

data_frame(labels = as.factor(y), row_averages = avg) %>%
  qplot(labels, row_averages, data = ., geom = "boxplot")

avgs <- apply(x, 1, mean)
sds <- apply(x, 2, sd)

# FILTERING COLUMNS BASED ON SUMMARIES

# The operations used to extract columns: x[, c(351, 352)]

# The operations used to extract rows: x[c(2, 3),]

# We can also use logical indexes to determine which columns or rows to keep:
# new_x <- x[, colSds(x) > 60].

# Important note: If you select one column or only one row, the result is no
# longer a matrix but a vector. We can preserve the matrix class by using the
# argument drop = FALSE.

library(matrixStats)
sds <- colSds(x)
qplot(sds, bins = "30", color = I("black"))
image(1:28, 1:28, matrix(sds, 28, 28)[, 28:1])

# extract columns and rows
x[, c(351, 352)]
x[c(2, 3),]
new_x <- x[, colSds(x) > 60]
dim(new_x)
class(x[,1])
dim(x[1,])

# preserve the matrix class
class(x[, 1, drop = FALSE])
dim(x[, 1, drop = FALSE])

# INDEXING WITH MATRICES AND BINARIZING THE DATA

# We can use logical operations with matrices:
mat <- matrix(1:15, 5, 3)
mat[mat > 6 & mat < 12] <- 0
mat

# We can also binarize the data using just matrix operations:
bin_x <- x
bin_x[bin_x < 255/2] <- 0
bin_x[bin_x > 255/2] <- 1

# index with matrices
mat <- matrix(1:15, 5, 3)
as.vector(mat)
qplot(as.vector(x), bins = 30, color = I("black"))
new_x <- x
new_x[new_x < 50] <- 0

mat <- matrix(1:15, 5, 3)
mat[mat < 3] <- 0
mat

mat <- matrix(1:15, 5, 3)
mat[mat > 6 & mat < 12] <- 0
mat

# binarize the data
bin_x <- x
bin_x[bin_x < 255/2] <- 0
bin_x[bin_x > 255/2] <- 1
bin_X <- (x > 255/2) * 1

# VECTORIZATION FOR MATRICES AND THE MATRIX ALGEBRA OPERATIONS

# We can scale each row of a matrix using this line of code:
(x - rowMeans(x)) / rowSds(x)

# To scale each column of a matrix, we use this code:
t(t(x) - colMeans(x))

# We can also use a function called sweep() that works similarly to apply(). It
# takes each entry of a vector and substracts it from the corresponding row or
# column.
X_mean_0 <- sweep(x, 2, colMeans(x))

# Matrix multiplication: t(x) %*% x
# The cross product: crossprod(x)
# The inverse of a function: solve(crossprod(x))
# The QR decomposition: qr(x)

# scale each row of a matrix:
(x - rowMeans(x)) / rowSds(x)

# scale each column:
t(t(x) - colMeans(x))

# take each entry of a vector and substract it from the corresponding row or column:
x_mean_0 <- sweep(x, 2, colMeans(x))

# divide by the standard deviation
x_mean_0 <- sweep(x, 2, colMeans(x))
x_standardized <- sweep(x_mean_0, 2, colSds(x), FUN = "/")

# COMPREHENSION CHECK: WORKING WITH MATRICES

# Q1

# Which line of code correctly creates a 100 by 10 matrix of randomly generated
# normal numbers and assigns it to x?

# 1
x <- matrix(rnorm(1000), 100, 100)
# 2
x <- matrix(rnorm(100*10), 100, 10) # [X]
# 3
x <- matrix(rnorm(100*10), 10, 10)
# 4
x <- matrix(rnorm(100*10), 10, 100)

# Q2

# Write the line of code that would give you the specified information about the
# matrix x that you generated in Q1. Do not include any spaces in your line of code.

# Dimension of x
dim(x)

# Number of rows of x
nrow(x)

# Number of columns of x
ncol(x)

# Q3

# Which of the following lines of code would add the scalar 1 to row 1, the scalar 2
# to row 2, and so on, for the matrix x? Select all that apply.

# 1
x <- x + seq(nrow(x)) # [X]
# 2
x <- 1:nrow(x)
# 3
x <- sweep(x, 2, 1:nrow(x), "+")
# 4
x <- sweep(x, 1, 1:nrow(x), "+") # [X]

# Q4

# Which of the following lines of code would add the scalar 1 to column 1, the
# scalar 2 to column 2, and so on, for the matrix x? Select all that apply.

# 1
x <- 1:ncol(x)
# 2
x <- 1:col(x)
# 3
x <- sweep(x, 2, 1:ncol(x), FUN = "+") # [X]
# 4
x <- -x

# Q5

# Which code correctly computes the average of each row of x?

# 1
mean(x)
# 2
rowMedians(x)
# 3
sapply(x, mean)
# 4
rowSums(x)
# 5
rowMeans(x) # [X]

# Which code correctly computes the average of each column of x?

# 1
mean(x)
# 2
sapply(x, mean)
# 3
colMeans(x) # [X]
# 4
colMedians(x)
# 5
colSums(x)

# Q6

# For each observation in the mnist training data, compute the proportion of pixels
# that are in the grey area, defined as values between 50 and 205 (but not including
# 50 and 205). (To visualize this, you can make a boxplot by digit class).

# What proportion of the 60000 * 784 pixels in the mnist training data are in the
# gray area overall, defined as values between 50 and 205?

mnist <- read_mnist()
head(mnist$train)

mnist <- mnist$train$images
class(mnist)
dim(mnist)
mnist[1:5, 1:5]
range(mnist)
mnist[mnist > 50 & mnist < 205] <- 1
mnist[mnist <= 50 | mnist >= 205] <- 0
mean(mnist)

# Optimal solution
mnist <- read_mnist()
y <- rowMeans(mnist$train$images > 50 & mnist$train$images < 205)
qplot(as.factor(mnist$train$labels), y, geom = "boxplot")
mean(y)

## SECTION 4: DISTANCE, KNN, CROSS VALIDATION, AND GENERATIVE MODELS

# In this section you will learn about different types of discriminative and
# generative approaches for machine learning algorithms.

# After completing this section, you will be able to:

# Use the k-nearest neighbors (KNN) algorithm.

# Understand the problems of overfitting and oversmoothing.

# Use cross-validation to reduce the true error and the apparent error.

# Use generative models such as naive Bayes, quadratic discriminant analysis (qda),
# and linear discriminant analysis (lda) for machine learning.

# 4.1. NEAREST NEIGHBORS

# DISTANCE

# Most clustering and machine learning techniques rely on being able to define
# distance between observations, using features or predictors.

# With high dimensional data, a quick way to compute all the distances at once is
# to use the function dist(), which computes the distance between each row and
# produces an object of class dist():
d <- dist(x)

# We can also compute distances between predictors. If N is the number of
# observations, the distance between two predictors, say 1 and 2, is:

# dist(1,2) = sqrt(sum(i=1 to N, (xi,1 - xi,2)^2))

# To compute the distance between all pairs of 784 predictors, we can transpose
# the matrix first and then use dist():
d <- dist(t(x))

library(tidyverse)
library(dslabs)

if(!exists("mnist")) mnist <- read_mnist()
set.seed(0, sample.kind = "Rounding")
ind <- which(mnist$train$labels %in% c(2,7)) %>% sample(500)

# the predictors are in x and the labels in y
x <- mnist$train$images[ind,]
y <- mnist$train$labels[ind]

y[1:3]

x_1 <- x[1,]
x_2 <- x[2,]
x_3 <- x[3,]

# distance between two numbers
sqrt(sum((x_1 - x_2)^2))
sqrt(sum((x_1 - x_3)^2))
sqrt(sum((x_2 - x_3)^2))

# compute distance using matrix algebra
sqrt(crossprod(x_1 - x_2))
sqrt(crossprod(x_1 - x_3))
sqrt(crossprod(x_2 - x_3))

# compute distance between each row
d <- dist(x)
class(d)
as.matrix(d)[1:3, 1:3]

# visualize these distances
image(as.matrix(d))

# order the distance by labels
image(as.matrix(d)[order(y), order(y)])

# compute distance between predictors
d <- dist(t(x))
dim(as.matrix(d))

d_492 <- as.matrix(d)[492,]

image(1:28, 1:28, matrix(d_492, 28, 28))

# COMPREHENSION CHECK: DISTANCE

# Q1

# Load the following dataset
library(dslabs)
data("tissue_gene_expression")

# This dataset includes a matrix x:
dim(tissue_gene_expression$x)

# This matrix has the gene expression levels of 500 genes from 189 biological
# samples representing seven different tissues. The tissue type is stored in y:
table(tissue_gene_expression$y)

# Which of the following lines of code computes the Euclidean distance between
# each observation and stores it in the object d?

# 1
d <- dist(tissue_gene_expression$x, distance = "maximum")

# 2
d <- dist(tissue_gene_expression)

# 3
d <- dist(tissue_gene_expression$x) # [X]

# 4
d <- cor(tissue_gene_expression$x)

# Q2

# Using the dataset from Q1, compare the distances between observations 1 and 2
# (both cerebellum), observations 39 and 40 (both colon), and observations 73
# and 74 (both endometrium).

d[1:2]
d[39:40]
d[73:74]

# Distance-wise, are samples from tissues of the same type closer to each other
# than tissues of different type?

# No, the samples from the same tissue type are not necessarily closer.

# The two colon samples are close to each other, but the samples from the other
# two tissues are not.

# The two cerebellum samples are close to each other, but the samples from the
# other two tissues are not.

# Yes, the samples from the same tissue type are closer to each other. [X]

# Also...
ind <- c(1, 2, 39, 40, 73, 74)
as.matrix(d)[ind, ind]

# Q3

# Make a plot of all the distances using the image() function to see if the pattern
# you observed in Q2 is general.

# Which code would correctly make the desired plot?

# 1
image(d)

# 2
image(as.matrix(d)) # [X]

# 3
d

# 4
image()

# KNN

# K-nearest neighbors (KNN) estimates the conditional probabilities in a similar
# way to bin smoothing. However, KNN is easier to adapt to multiple dimensions.

# Using KNN, for any point(x1,x2) for which we want an estimate of p(x1,x2), we
# look for the k nearest points to (x1,x2) and take an average of the 0s and 1s
# associated with these points. We refer to the set of points used to compute the
# average as the neighborhood. Larger values of k result in smoother estimates,
# while smaller values of k result in more flexible and more wiggly estimates.

# To implement the algorithm, we can use the knn3() function from the caret
# package. There are two ways to call this function:

# 1. We need to specify a formula and a data frame. The formula looks like this:
# outcome ~ predictor1 + predictor2 + predictor3. The predict() function for knn3
# produces a probability for each class.

# 2. We can also call the function with the first argument being the matrix
# predictors and the second a vector of outcomes, like this:
x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y
knn_fit <- knn3(x, y)

library(tidyverse)
library(dslabs)
data("mnist_27")
mnist_27$test %>% ggplot(aes(x_1, x_2, color = y)) + geom_point()

# logistic regression
library(caret)
fit_glm <- glm(y ~ x_1 + x_2, data = mnist_27$train, family = "binomial")
p_hat_logistic <- predict(fit_glm, mnist_27$test)
y_hat_logistic <- factor(ifelse(p_hat_logistic > 0.5, 7, 2))
confusionMatrix(data = y_hat_logistic, reference = mnist_27$test$y)$overall[1]

# fit knn model
knn_fit <- knn3(y ~ ., data = mnist_27$train)

x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y
knn_fit <- knn3(x, y)

knn_fit <- knn3(y ~ ., data = mnist_27$train, k = 5)

y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn, reference = mnist_27$test$y)$overall[1]

# OVERTRAINING AND OVERSMOOTHING

# Over-training is the reason that we have higher accuracy in the train set
# compared to the test set. Over-training is at its worst when we set k = 1. With
# k = 1, the estimate for each (x1,x2) in the training set is obtained with just
# the y corresponding to that point.

# When we try a larger k, the k might be so large that it does not permit enough
# flexibility. We call this over-smoothing.

# Note that if we use the test set to pick this k, we should not expect the
# accompanying accuracy estimate to extrapolate to the real world. This is because
# even here we broke a golden rule of machine learning: we selected the k using the
# test set. Cross validation also provides an estimate that takes this into account.

y_hat_knn <- predict(knn_fit, mnist_27$train, type = "class")
confusionMatrix(data = y_hat_knn, reference = mnist_27$train$y)$overall["Accuracy"]
y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn, reference = mnist_27$test$y)$overall["Accuracy"]

# fit knn with k = 1
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k = 1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type = "class")
confusionMatrix(data = y_hat_knn_1, reference = mnist_27$train$y)$overall[["Accuracy"]]

y_hat_knn_1 <- predict(knn_fit_1, mnist_27$test, type = "class")
confusionMatrix(data = y_hat_knn_1, reference = mnist_27$test$y)$overall[["Accuracy"]]

#fit knn with k=401
knn_fit_401 <- knn3(y ~ ., data = mnist_27$train, k = 401)
y_hat_knn_401 <- predict(knn_fit_401, mnist_27$test, type = "class")
confusionMatrix(data=y_hat_knn_401, reference=mnist_27$test$y)$overall["Accuracy"]

# pick the k in knn
ks <- seq(3, 251, 2)
library(purrr)
accuracy <- map_df(ks, function(k){
  fit <- knn3(y ~ ., data = mnist_27$train, k = k)
  
  y_hat <- predict(fit, mnist_27$train, type = "class")
  cm_train <- confusionMatrix(data = y_hat, reference = mnist_27$train$y)
  train_error <- cm_train$overall["Accuracy"]
  
  y_hat <- predict(fit, mnist_27$test, type = "class")
  cm_test <- confusionMatrix(data = y_hat, reference= mnist_27$test$y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})

# pick the k that maximizes accuracy using the estimates built on the test data
ks[which.max(accuracy$test)]
max(accuracy$test)

# COMPREHENSION CHECK: NEAREST NEIGHBORS

# Q1

# Previously, we used logistic regression to predict sex based on height. Now we
# are going to use knn to do the same. Set the seed to 1, then use the caret
# package to partition the dslabs heights data into a training and test set of 
# equal size. Use the sapply() funtion to perform knn with k values of
# seq(1, 101, 3) and calculate F1 scores with the F_meas() function using the
# default value of the relevant argument.
data(heights)
set.seed(1, sample.kind = "Rounding")

y <- heights$sex
x <- heights$height

test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index,]
train_set <- heights[-test_index,]

ks <- seq(1, 101, 3)

F_1 <- sapply(ks, function(k){
  fit <- knn3(sex ~ height, data = train_set, k = k)
  y_hat <- predict(fit, test_set, type = "class")
  F_meas(data = y_hat, reference = factor(test_set$sex))
})

# What is the max value of F_1?
max(F_1) # 0.6019

# At what value of k does the max occur? If there are multiple values of k with
# the maximum value, report the smallest such k.
ks[which.max(F_1)]

# Also...
library(dslabs)
library(tidyverse)
library(caret)
data("heights")

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(heights$sex, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index]
train_set <- heights[-test_index]

ks <- seq(1, 101, 3)
F_1 <- sapply(ks, function(k){
  fit <- knn3(sex ~ height, data = train_set, k = k)
  y_hat <- predict(fit, test_set, type = "class") %>%
    factor(levels = levels(train_set$sex))
  F_meas(data = y_hat, reference = test_set$sex)
})

# Q2

# Next we will use the same gene expression example used in Comprehension Check:
# Distance exercises. You can load it like this:
library(dslabs)
library(caret)
data(tissue_gene_expression)


# First, set the seed to 1 and split the data into training and test sets with
# p = 0.5. Then, report the accuracy you obtain from predicting tissue type using
# KNN with k = seq(1, 11, 2) using sapply() or map_df(). Note: Use the 
# createDataPartition() function outside of sapply() or map_df().

set.seed(1, sample.kind = "Rounding")
x <- tissue_gene_expression$x
y <- tissue_gene_expression$y
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)

ks <- seq(1, 11, 2)
accuracy <- sapply(ks, function(k){
  fit <- knn3(y~., data.frame(x = x[-test_index, ], y = y[-test_index]), k = k)
  y_hat <- predict(fit, data.frame(x = x[test_index, ], y = y[test_index]),
                   type = "class")
  accuracy <- confusionMatrix(y_hat, y[test_index])$overall[["Accuracy"]]
})
data.frame(ks, accuracy)

# 4.2. CROSS-VALIDATION

# K-FOLD CROSS-VALIDATION

# For k-fold cross validation, we divide the dataset into a training set and a
# test set. We train our algorithm exclusively on the training set and use the
# test set only for evaluation purposes.

# For each set of algorithm parameters being considered, we want an estimate of
# the MSE and then we will choose the parameters with the smallest MSE. In k-fold
# cross validation, we randomly split the observations into k non-overlapping
# sets, and repeat the calculation for MSE for each of these sets. Then, we
# compute the average MSE and obtain an estimate of our loss. Finally, we can
# select the optimal parameter that minimized the MSE.

# In terms of how to select k for cross validation, larger values of k are
# preferable but they will also take much more computational time. For this
# reason, the choices of k = 5 and k = 10 are common.

# COMPREHENSION CHECK: CROSS-VALIDATION

# Q1

# Generate a set of random predictors and outcomes using the following code:
library(tidyverse)
library(caret)

set.seed(1996, sample.kind = "Rounding")
n <- 1000
p <- 10000
x <- matrix(rnorm(n*p), n, p)
colnames(x) <- paste("x", 1:ncol(x), sep = "_")
y <- rbinom(n, 1, 0.5) %>% factor()

x_subset <- x[ , sample(p, 100)]

# Because x and y are completely independent, you should not be able to predict
# y using x with accuracy greater than 0.5. Confirm this by running cross-validation
# using logistic regression to fit the model. Because we have so many predictors,
# we selected a random sample x_subset. Use the subset when training the model.

# Which code correctly performs cross-validation?

# 1
fit <- train(x_subset, y)
fit$results

# 2 [X]
fit <- train(x_subset, y, method = "glm")
fit$results

# 3
fit <- train(y, x_subset, method = "glm")
fit$results

# 4
fit <- test(x_subset, y, method = "glm")
fit$results

# Q2

# Now, instead of using a random selection of predictors, we are going to search
# for those that are most predictive of the outcome. We can do this by comparing
# the values for the y = 1 group to those in the y = 0 group, for each predictor,
# using a t-test. You can perform this step like this:
install.packages("BiocManager")
BiocManager::install("genefilter")
library(genefilter)
tt <- colttests(x, y)

# Which of the following lines of code correctly creates a vector of the p-values
# called pvals?

# 1
pvals <- tt$dm

# 2
pvals <- tt$statistic

# 3
pvals <- tt

# 4
pvals <- tt$p.value # [X]

# Q3

# Create an index ind with the column numbers of the predictors that were
# statistically significant associated with y. Use a p-value cutoff of
# 0.01 to define statistically significant.

# How many predictors survive this cutoff?

ind <- tt %>% mutate(n = row_number()) %>%
  filter(p.value < 0.01) %>%
  select(n)
nrow(ind)

# Also...
ind <- which(pvals <= 0.01)
length(ind)

# Q4

# Now re-run the cross-validation after redefining x_subset to be the
# subset of x defined by the columns showing statistically significant
# association with y.

# What is the accuracy now?

x_subset <- x[ , ind]
fit <- train(x_subset, y, method = "glm")
fit$results # 0.753

# Q5

# Re-run the cross-validation again, but this time using kNN. Try out the
# following grid k = seq(101, 301, 25) of tuning parameters. Make a plot
# of the resulting accuracies.

# Which code is correct?

# 1
fit <- train(x_subset, y, method = "knn",
             tuneGrid = data.frame(k = seq(101, 301, 25)))
ggplot(fit) # [X]

# 2
fit <- train(x_subset, y, method = "knn")
ggplot(fit)

# 3
fit <- train(x_subset, y, method = "knn", tuneGrid = 
               data.frame(k = seq(103, 301, 25)))
ggplot(fit)

# 4
fit <- train(x_subset, y, method = "knn",
             tuneGrid = data.frame(k = seq(101, 301, 5)))
ggplot(fit)

# Q6

# In the previous exercises, we see that despite the fact that x and y are completely
# independent, we were able to predict y with accuracy higher than 70%. We must be doing
# something wrong then.

# What is it?

# The function train() estimates accuracy on the same data it uses to train the
# algorithm.

# We are overfitting the model by including 100 predictors.

# We used the entire dataset to select the columns used in the model. [X]

# The high accuracy is just due to random variability.

# Q7

# Use the train() function with kNN to select the best k for predicting tissue from
# gene expression on the tissue_gene_expression dataset from dslabs. Try k =
# seq(1, 7, 2) for tuning parameters. For this question, do not split the data into
# test and train sets (understand this can lead to overfitting, but ignore this for
# now).

# What value of k results in highest accuracy?
library(dslabs)
data("tissue_gene_expression")

fit <- train(tissue_gene_expression$x, tissue_gene_expression$y, method = "knn",
             tuneGrid = data.frame(k = seq(1, 7, 2)))
ggplot(fit)

# Also...
fit <- with(tissue_gene_expression, train(x, y, method = "knn",
                                          tuneGrid = data.frame(k = seq(1, 7, 2))))
ggplot(fit)
fit$results

# BOOTSTRAP

# When we don't have access to the entire population, we can use bootstrap to estimate
# the population median m.

# The bootstrap permits us to approximate a Monte Carlo simulation without access to
# the entire distribution. The general idea is relatively simple. We act as if the
# observed sample is the population. We then sample datasets (with replacement) of the
# same sample size as the original dataset. Then we compute the summary statistic, in
# this case the median, on this bootstrap sample.

# Note that we can use ideas similar to those used in the bootstrap in cross validation:
# instead of dividing the data into equal partitions, we simply bootstrap many times.
n <- 10^6
income <- 10^rnorm(n, log10(45000), log10(3))
qplot(log10(income), bins = 30, color = I("black"))

m <- median(income)
m

set.seed(1, sample.kind = "Rounding")
N <- 250
X <- sample(income, N)
M <- median(X)
M

library(gridExtra)
B <- 10^5
M <- replicate(B, {
  X <- sample(income, N)
  median(X)
})
p1 <- qplot(M, bins = 30, color = I("black"))
p2 <- qplot(sample = scale(M)) + geom_abline()
grid.arrange(p1, p2, ncol = 2)

mean(M)
sd(M)

B <- 10^5
M_star <- replicate(B, {
  X_star <- sample(X, N, replace = TRUE)
  median(X_star)
})

tibble(monte_carlo = sort(M), bootstrap = sort(M_star)) %>%
  qplot(monte_carlo, bootstrap, data = .) +
  geom_abline()

quantile(M, c(0.05, 0.95))
quantile(M_star, c(0.05, 0.95))

median(X) + 1.96 * sd(X) / sqrt(N) * c(-1 , 1)
mean(M) + 1.96 * sd(M) * c(-1, 1)

mean(M_star) + 1.96 * sd(M_star) * c(-1, 1)

# COMPREHENSION CHECK: BOOTSTRAP

# Q1

# The createResample() function can be used to create bootstrap samples. For example,
# we can create the indexes for 10 bootstrap samples for the mnist_27 dataset like
# this:
library(dslabs)
library(caret)
data(mnist_27)
set.seed(1995, sample.kind = "Rounding")
indexes <- createResample(mnist_27$train$y, 10)

# How many times to 3, 4, and 7 appear in the first resampled index?

sum(indexes$Resample01 == 3) # 1
sum(indexes$Resample01 == 4) # 4
sum(indexes$Resample01 == 7) # 0

# Q2

# We see that some numbers appear more than once and others appear no times. This has
# to be this way for each dataset to be independent. Repeat the exercise for all the
# resampled indexes.

# What is the total number of times that 3 appears in all of the resampled indexes?

m <- sapply(1:10, function(i){
  x <- sum(indexes[[i]] == 3)
  y <- sum(indexes[[i]] == 4)
  z <- sum(indexes[[i]] == 7)
  c(x, y, z)
})
sum(m[1,]) # 11

# Also
x <- sapply(indexes, function(ind){
  sum(ind == 3)
})
sum(x)

# Q3

# A random dataset can be generated with the following code:
y <- rnorm(100, 0, 1)

# Estimate the 75th quantile, which we know is qnorm(0.75), with the sample quantile:
# quantile(y, 0.75).
quantile(y, 0.75)

# Now set the seed to 1 and perform a Monte Carlo simulation with 10,000 repetitions,
# generating the random dataset and estimating the 75th quantile each time. What is
# the expected value and standard error of the 75th quantile?

set.seed(1, sample.kind = "Rounding")
B <- 10^5
res <- replicate(B, {
  y <- rnorm(100, 0, 1)
  quantile(y, 0.75)
})

# Expected value
mean(res)

# Standard error
sd(res)

# Q4

# In practice, we can't run a Monte Carlo simulation. Use the sample:
set.seed(1, sample.kind = "Rounding")
y <- rnorm(100, 0, 1)

# Set the seed to 1 again after generating y and use 10 bootstrap samples to estimate
# the expected value and standard error of the 75th quantile.
set.seed(1, sample.kind = "Rounding")
indexes <- createResample(y, 10)

# Expected value

q_75_star <- sapply(indexes, function(ind){
  quantile(y[ind], 0.75)  
})

mean(q_75_star)

# Standard error
sd(q_75_star)

# Q5

# Repeat the exercise from Q4 but with 10,000 bootstrap samples instead of 10. Set the
# seed to 1 first.

set.seed(1, sample.kind = "Rounding")
indexes <- createResample(y, 10000)

q_75_star <- sapply(indexes, function(ind){
  quantile(y[ind], 0.75)
})

# Expected value
mean(q_75_star)

# Standard error
sd(q_75_star)

# Q6

# When doing bootstrap sampling, the simulated samples are drawn from the empirical 
# distribution of the original data.

# True or False: The bootstrap is particularly useful in situations when we do not
# have access to the distribution or it is unknown.

# True [X]
# False

# 4.3. GENERATIVE MODELS

# Discriminative approaches estimate the conditional probability directly and do not
# consider the distribution of predictors.

# Generative models are methods that model the joint distribution of X (we model
# how the entire data, X and Y, are generated)

# NAIVE BAYES

# Bayes' rule:

# p(x) = Pr(Y=1|X=x) = fx|y=1(X)Pr(Y=1) / (fx|y=0(X)Pr(Y=0)+fx|y=1(X)Pr(Y=1))

# with fx|y=1 and fx|y=0 representing the distribution functions of the predictor X
# for the two classes Y = 1 and Y = 0.

# The Naive Bayes approach is similar to the logistic regression prediction mathematically.
# However, we leave the demonstration to a more advanced text.

# Generating train and test set
library(caret)
data(heights)
y <- heights$height
set.seed(2, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
train_set <- heights %>% slice(-test_index)
test_set <- heights %>% slice(test_index)

# Estimating averages and standard deviations
params <- train_set %>%
  group_by(sex) %>%
  summarize(avg = mean(height), sd = sd(height))
params

# Estimating the prevalence
pi <- train_set %>% summarize(pi = mean(sex == "Female")) %>% pull(pi)
pi

# Getting an actual rule
x <- test_set$height
f0 <- dnorm(x, params$avg[2], params$sd[2])
f1 <- dnorm(x, params$avg[1], params$sd[1])
p_hat_bayes <- f1 * pi / (f1 * pi + f0 * (1 - pi))

# CONTROLLING PREVALENCE

# The Naive Bayes approach includes a parameter to account for differences in prevalence
# pi = Pr(Y=1). If we use hats to denote the estimates, we can write p(x)_hat as:

# p(x)_hat=Pr(Y=1|X=x)=f_hat,x|y=1(x)pi_hat/(f_hat,x|y=0(x)(1-pi_hat) +
# f_hat,x|y=1(x)Pr(Y=1))

# The Naive Bayes approach gives us a direct way to correct the imbalance between
# sensitivity and specificity by simply forcing pi_hat to be whatever value we want it
# to be in order to better balance specificity and sensitivity.

# computing sensitivity
y_hat_bayes <- ifelse(p_hat_bayes > 0.5, "Female", "Male")
sensitivity(data = factor(y_hat_bayes), reference = factor(test_set$sex))

# computing specificity
specificity(data = factor(y_hat_bayes), reference = factor(test_set$sex))

# changing the cutoff of the decision rule
p_hat_bayes_unbiased <- f1 * 0.5 / (f1 * 0.5 + f0 * (1 - 0.5))
y_hat_bayes_unbiased <- ifelse(p_hat_bayes_unbiased > 0.5, "Female", "Male")
sensitivity(data = factor(y_hat_bayes_unbiased), reference = factor(test_set$sex))
specificity(data = factor(y_hat_bayes_unbiased), reference = factor(test_set$sex))

# Draw plot
qplot(x, p_hat_bayes_unbiased, geom = "line") +
  geom_hline(yintercept = 0.5, lty = 2) +
  geom_vline(xintercept = 67, lty = 2)

# QDA AND LDA

# Quadratic discrimnant analysis (QDA) is a version of Naive Bayes in which we assume
# that the distributions px|y=1(x) and px|y=0(x) are multivariate normal.

# QDA can work well with a few predictors, but it becomes harder to use as the number
# of predictors increases. Once the number of parameters appraches the size of our
# data, the method becomes impractical due to overfitting.

# Forcing the assumption that all predictors share the same standard deviations and
# correlations, the boundary will be a line, just as with logistic regression. For
# this reason, we call the method linear discriminant analysis (LDA).

# In the case of LDA, the lack of flexibility does not permit us to capture the
# non-linearity in the true conditional probability function.

# QDA

# load data
data(mnist_27)

# Estimate parameters from the data
params <- mnist_27$train %>%
  group_by(y) %>%
  summarize(avg_1 = mean(x_1), avg_2 = mean(x_2),
            sd_1 = sd(x_1), sd_2 = sd(x_2),
            r = cor(x_1, x_2))

# contour plots
mnist_27$train %>% mutate(y = factor(y)) %>%
  ggplot(aes(x_1, x_2, fill = y, color = y)) +
  geom_point(show.legend = FALSE) +
  stat_ellipse(type = "norm", lwd = 1.5)

# Fit model
library(caret)
train_qda <- train(y ~ ., method = "qda", data = mnist_27$train)
# Obtain predictors and accuracy
y_hat <- predict(train_qda, mnist_27$test)
confusionMatrix(data = y_hat, reference = mnist_27$test$y)$overall["Accuracy"]

# Draw separate plots for 2s and 7s
mnist_27$train %>% mutate(y = factor(y)) %>%
  ggplot(aes(x_1, x_2, fill = y, color = y)) +
  geom_point(show.legend = FALSE) +
  stat_ellipse(type = "norm") +
  facet_wrap(~ y)

# LDA

params <- mnist_27$train %>%
  group_by(y) %>%
  summarize(avg_1 = mean(x_1), avg_2 = mean(x_2),
            sd_1 = sd(x_1), sd_2 = sd(x_2),
            r = cor(x_1, x_2))
params <- params %>% mutate(sd_1 = mean(sd_1), sd_2 = mean(sd_2), r = mean(r))
train_lda <- train(y ~ ., method = "lda", data = mnist_27$train)
y_hat <- predict(train_lda, mnist_27$test)
confusionMatrix(data = y_hat, reference = mnist_27$test$y)$overall["Accuracy"]

# CASE STUDY: MORE THAN THREE CLASSES

# In this case study, we will briefly give a slightly more complex example: one with
# 3 classes instead of 2. Then we will fit QDA, LDA, and KNN models for prediction.

# Generative models can be very powerful, but only when we are able to successfully
# approximate the joint distribution of predictors conditioned on each class.

if(!exists("mnist")) mnist <- read_mnist()

set.seed(3456, sample.kind = "Rounding")
index_127 <- sample(which(mnist$train$labels %in% c(1, 2, 7)), 2000)
y <- mnist$train$labels[index_127]
x <- mnist$train$images[index_127,]
index_train <- createDataPartition(y, p = 0.8, list = FALSE)

# get the quadrants
# temporary object to help figure out the quadrants
row_column <- expand.grid(row = 1:28, col = 1:28)
upper_left_ind <- which(row_column$col <= 14 & row_column$row <= 14)
lower_right_ind <- which(row_column$col > 14 & row_column$row > 14)

# binarize the values. Above 200 is ink, below is no ink
x <- x > 200

# cbind proportion of pixels in upper right quadrant and proportion of pixels in
# lower right
x <- cbind(rowSums(x[, upper_left_ind]) / rowSums(x),
           rowSums(x[, lower_right_ind]) / rowSums(x))

train_set <- data.frame(y = factor(y[index_train]),
                        x_1 = x[index_train, 1],
                        x_2 = x[index_train, 2])

test_set <- data.frame(y = factor(y[-index_train]),
                       x_1 = x[-index_train, 1],
                       x_2 = x[-index_train, 2])

train_set %>% ggplot(aes(x_1, x_2, color = y)) + geom_point()

train_qda <- train(y ~ ., method = "qda", data = train_set)
predict(train_qda, test_set, type = "prob") %>% head()
predict(train_qda, test_set) %>% head()
confusionMatrix(predict(train_qda, test_set), test_set$y)$table
confusionMatrix(predict(train_qda, test_set), test_set$y)$overall["Accuracy"]
train_lda <- train(y ~ ., method = "lda", data = train_set)
confusionMatrix(predict(train_lda, test_set), test_set$y)$overall["Accuracy"]
train_knn <- train(y ~ ., method = "knn", tuneGrid = data.frame(
  k = seq(15, 51, 2)), data = train_set)
confusionMatrix(predict(train_knn, test_set), test_set$y)$overall["Accuracy"]
train_set %>% mutate(y = factor(y)) %>% ggplot(aes(x_1, x_2, fill = y, color = y)) +
  geom_point(show.legend = FALSE) + stat_ellipse(type = "norm")

# COMPREHENSION CHECK: GENERATIVE MODELS

# Q1

# Create a dataset of samples from just cerebellum and hippocampus, two parts of
# the brain, and a predictor matrix with 10 randomly selected columns using the
# following code:
library(dslabs)
library(caret)
library(tidyverse)
data(tissue_gene_expression)

set.seed(1993, sample.kind = "Rounding")
ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]

# Use the train() function to estimate the accuracy of LDA. For this question, use
# the version of x and y created with the code above: do not split them or
# tissue_gene_expression into training and test sets (understand this can lead to
# overfitting). Report the accuracy from the train() results (do not make predictions).

train_lda <- train(y ~ ., method = "lda", data = data.frame(x, y))
train_lda$results["Accuracy"]

train_lda <- train(x, y, method = "lda")
train_lda$results["Accuracy"]

# Q2

# In this case, LDA fits two 10-dimensional normal distributions. Look at the fitted
# model by looking at the finalModel component of the result of train(). Notice there
# is a component called means that includes the estimated means of both distributions.
# Plot the mean vectors against each other and determine which predictors (genes)
# appear to be driving the algorithm.

# Which TWO genes appear to be driving the algorithm (i.e., the two genes with the
# highest means)?

# PLCB1
# RAB1B [X]
# MSH4
# OAZ2 [X]
# SPI1
# SAPCD1
# HEMK1

train_lda$finalModel
train_lda$finalModel$means

t(train_lda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()

# Q3

# Repeat the exercise in Q1 with QDA.

# Create a dataset of samples from just cerebellum and hippocampus, two parts of the
# brain, and a predictor matrix with 10 randomly selected columns using the following
# code:
library(dslabs)
library(caret)
data(tissue_gene_expression)

set.seed(1993, sample.kind = "Rounding")
ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]

# Use the train() function to estimate the accuracy of QDA. For this question, use the
# version of x and y created above instead of the default from tissue_gene_expression.
# Do not split them into training and test sets (understand this can lead to overfitting).

# What is the accuracy?

fit_qda <- train(x, y, method = "qda")
fit_qda$results["Accuracy"]

# Q4

# Which two genes drive the algorithm when using QDA instead of LDA (i.e., the two
# genes with the highest means)? The same genes.

fit_qda$finalModel$means
t(fit_qda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()

# Q5

# One thing we saw in the previous plots is that the values of the predictors correlate
# in both groups: some predictors are low in both groups and others high in both groups.
# The mean value of each predictor found in colMeans(x) is not informative or useful
# for prediction and often for purposes of interpretation, it is useful to center or
# scale each column. This can be achieved with the preProcess argument in train().
# Re-run LDA with preProcess = "center". Note that accuracy does not change, but it is
# now easier to identify the predictors that differ more between groups than based on
# the plot made in Q2.

# Which TWO genes drive the algorithm after performing the scaling?

# C21orf62
# PLCB1
# RAB1B
# MSH4
# OAZ2 [X]
# SPI1 [X]
# SAPCD1
# IL18R1

set.seed(1993, sample.kind = "Rounding")
ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]

fit_lda <- train(x, y, method = "lda", preProcess = "center")
fit_lda$results["Accuracy"]

t(fit_lda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()

# You can see that it is different genes driving the algorithm now. This is because the
# predictor means change. In the previous exercises we saw that both LDA and QDA approaches
# worked well. For further exploration of the data, you can plot the predictor values for
# the two genes with the largest differences between the two groups in a scatter plot to
# see how they appear to follow a bivariate distribution as assumed by the LDA and QDA
# approaches, coloring the points by the outcome, using the following code:
d <- apply(fit_lda$finalModel$means, 2, diff)
ind <- order(abs(d), decreasing = TRUE)[1:2]
plot(x[, ind], col = y)

# Q6

# Now we are going to increase the complexity of the challenge slightly. Repeat the LDA
# analysis from Q5 but using all tissue types. Use the following code to create your
# dataset:
library(dslabs)
library(caret)
data(tissue_gene_expression)

set.seed(1993, sample.kind = "Rounding")
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
x <- x[, sample(ncol(x), 10)]

# What is the accuracy using LDA?

set.seed(1993, sample.kind = "Rounding")
fit_lda <- train(x, y, method = "lda", preProcess = "center")
fit_lda$results["Accuracy"]

## SECTION 5. CLASSIFICATION WITH MORE THAN TWO CLASSES AND THE CARET PACKAGE

# OVERVIEW

# In the classification with More than Two Classes and the Caret Package section, you will
# learn how to overcome the curse of dimensionality using methods that adapt to higher
# dimensions and how to use the caret package to implement many different machine learning
# algorithms.

# After this section, you will be able to:

# Use classification and regression trees.
# Use classification (decision) trees.
# Apply random forests to address the shortcomings of decision trees.
# Use the caret package to implement a variety of machine learning algorithsm.

