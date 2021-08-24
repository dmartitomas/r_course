### DATA SCIENCE: LINEAR REGRESSION

# In this course, you will learn:

# How linear regression was originally developed by Galton
# What confounding is and how to detect it
# How to examine the relationships between variables by implementing linear#
# regression in R

## SECTION 1: INTRODUCTION TO REGRESSION

# In the Introduction to Regression section, you will learn the basics of linear
# regression.

# You will learn, 

# Understand how Galton developed linear regression
# Calculate and interpret the sample correlation
# Stratify a dataset when appropriate
# Understand what bivariate normal distribution is
# Explain what the term variance explained means
# Interpret two regression lines

# 1.1 BASEBALL AS A MOTIVATING EXAMPLE

# MOTIVATING EXAMPLE: MONEYBALL

# Bill James was the originator of the sabermetrics, the approach of using data
# to predict what outcomes best predicted if a team would win.

# BASEBALL BASICS

# The goal of a baseball game is to score more runs than the other team.

# Each team has 9 batters who have an opportunity to hit a ball with a bat
# in a predetermined order.

# Each time a batter has an opportunity to bat, we call it a plate appearance (PA).

# The PA ends with a binary outcome: the batter either makes an out (failure) and
# returns to the bench or the batter doesn't (success) and can around the bases,
# and potentially score a run (reach all 4 bases).

# We are simplifying a bit, but there are five ways a batter can succeed (not make
# an out):

# 1. Bases on balls (BB): the pitcher fails to throw the ball through a predefined
# area considered to be hittable (the strike zone), so the batter is permitted to
# go to first base.

# 2. Single: the batter hits the ball and gets to first base.

# 3. Double (2B): the batter hits the ball and gets to second base.

# 4. Triple (3B): the batter hits the ball and gets to third base.

# 5. Home Run (HR): the batter hits the ball and goes all the way home and
# scores a run.

# Historically, the batting average has been considered the most important offensive
# statistic. To define this average, we define a hit (H) and an at bat (AB). Singles,
# doubles, triples, and home runs are hits. The fifth way to be successful, a walk
# (BB), is not a hit. An AB is the number of times you either get a hit or make an
# out; BBs are excluded. The batting average is simply H/AB and is considered the
# main measure of a success rate.

# BASES ON BALLS OR STOLEN BASES?

# The visualization of choice when exploring the relationship between two variables
# like home runs and runs is a scatterplot.

# Scatterplot of the relationship between HRs and wins
library(Lahman)
library(tidyverse)
library(dslabs)
ds_theme_set()

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(HR_per_game = HR / G, R_per_game = R / G) %>%
  ggplot(aes(HR_per_game, R_per_game)) +
  geom_point(alpha = 0.5)

# Scatterplot of the relationship between stolen bases and wins

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(SB_per_game = SB / G, R_per_game = R / G) %>%
  ggplot(aes(SB_per_game, R_per_game)) +
  geom_point(alpha = 0.5)

# Scatterplot of the relationship between bases on balls and runs

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(BB_per_game = BB / G, R_per_game = R / G) %>%
  ggplot(aes(BB_per_game, R_per_game)) +
  geom_point(alpha = 0.5)

# ASSESSMENT: BASEBALL AS A MOTIVATING EXAMPLE

# Question 1

# What is the application of statistics and data science to baseball called?

# Moneyball
# Sambermetrics [X]
# The "Oakland A's Approach"
# There is no specific name for this; it's just data science

# Question 2

# Which of the following outcomes is not included in the batting average?

# A home run
# A base on balls [X]
# An out
# A single

# Question 3

# Why do we consider team statistics as well as individual player statistics?

# The success of any individual player also depends on the strength of their team. [X]
# Team statistics can be easier to calculate.
# The ultimate goal of sabermetrics is to rank teams, not players.

# Question 4

# You want to know whether teams with more at-bats per game have more runs per
# game.

# What R code below correctly makes a scatterplot for this relationship?

# 1
Teams %>% filter(yearID %in% 1961:2001) %>%
  ggplot(aes(AB, R)) +
  geom_point(alpha = 0.5)

# 2
Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(AB_per_game = AB/G, R_per_game = R/G) %>%
  ggplot(aes(AB_per_game, R_per_game)) +
  geom_point(alpha = 0.5) # [X]

# 3
Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(AB_per_game = AB/G, R_per_game = R/G) %>%
  ggplot(aes(AB_per_game, R_per_game)) +
  geom_line()

# 4
Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(AB_per_game = AB/G, R_per_game = R/G) %>%
  ggplot(aes(R_per_game, AB_per_game)) +
  geom_point()

# Question 5

# What does the variable "SOA" stand for in the Teams table?

# sacrifice out
# slides or attempts
# strikeouts by pitchers [X]
# accumulated singles

# Question 6

# Load the Lahman library. Filter the Teams data frame to include years from
# 1961 to 2001. Make a scatterplot of runs per game versus at bats (AB) per game.

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(R_per_game = R/G, AB_per_game = AB/G) %>%
  ggplot(aes(AB_per_game, R_per_game)) +
  geom_point(alpha = 0.5)

# Which of the following is true?

# There is no clear relationship between runs and at bats per game.

# As the number of at bats per game increases, the number of runs per game
# tends to increase. [X]

# As the number of at bats per game increases, the number of runs per game
# tends to decrease.

# Question 7

# Use the filtered Teams data frame from Question 6. Make a scatterplot of win
# rate (number of wins per game) versus the number of fielding errors (E) per
# game.

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(W_per_game = W/G, E_per_game = E/G) %>%
  ggplot(aes(E_per_game, W_per_game)) +
  geom_point(alpha = 0.5)

# Which of the following is true?

# There is no relationship between win rate and errors per game.

# As the number of errors per game increases, the win rate tends to increase.

# As the number of errors per game decreases, the win rate tends to decrease. [X]

# Question 8

# Use the filtered Teams data frame from Question 6. Make a scatterplot of triples
# (X3B) per game versus doubles (X2B) per game.

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(X3B_per_game = X3B/G, X2B_per_game = X2B/G) %>%
  ggplot(aes(X2B_per_game, X3B_per_game)) +
  geom_point(alpha = 0.5)

# There is no clear relationship between doubles per game and triples per game. [X]

# As the number of doubles per game increases, the number of triples per game
# tends to increase.

# As the number of doubles per game increases, the number of triples per game
# tends to increase.

# 1.2 CORRELATION

# CORRELATION

# Galton tried to predict sons' heights based on fathers' heights.

# The mean and standard errors are insufficient for describing an important
# characteristic of the data: the trend that the taller the father, the taller
# the son.

# The correlation coefficient is an informative summary of how two variables
# move together that can be used to predict one variable using the other.

# create the dataset
library(tidyverse)
library(HistData)
data("GaltonFamilies")
set.seed(1983, sample.kind = "Rounding")
galton_heights <- GaltonFamilies %>%
  filter(gender == "male") %>%
  group_by(family) %>%
  sample_n(1) %>%
  ungroup() %>%
  select(father, childHeight) %>%
  rename(son = childHeight)

# means and standard deviations
galton_heights %>%
  summarize(mean(father), sd(father), mean(son), sd(son))

# scatterplot of father and son heights
galton_heights %>%
  ggplot(aes(father, son)) +
  geom_point(alpha = 0.5)

# CORRELATION COEFFICIENT

# The correlation coefficient is defined for a list of pairs (x1, y1),...,(xn, yn)
# as the product of the standardized values:

# (xi - `mu`x / `sigma`x) * (yi - `mu`y / `sigma`y)

# The correlation coefficient essentially conveys how two variables move together.

# The correlation coefficient is always between -1 and 1.

rho <- mean(scale(x)*scale(y))
galton_heights %>% summarize(r = cor(father, son)) %>% pull(r)

# SAMPLE CORRELATION IS A RANDOM VARIABLE

# The correlation that we compute and use as a summary is a random variable.

# When interpreting correlations, it is important to remember that correlations
# derived from samples are estimates containing uncertainty.

# Because the sample correlation is an average of independent draws, the central
# limit theorem applies.

# compute sample correlation
R <- sample_n(galton_heights, 25, replace = TRUE) %>%
  summarize(r = cor(father, son))
R

# Monte Carlo simulation to show distribution of sample correlation
B <- 1000
N <- 25
R <- replicate(B, {
  sample_n(galton_heights, N, replace = TRUE) %>%
  summarize(r = cor(father, son)) %>%
  pull(r)
})
qplot(R, geom = "histogram", binwidth = 0.05, color = I("black"))

# expected value and standard error
mean(R)
sd(R)

# QQ-plot to evaluate whether N is large enough
data.frame(R) %>%
  ggplot(aes(sample = R)) +
  stat_qq() +
  geom_abline(intercept = mean(R), slope = sqrt((1-mean(R)^2)/(N-2)))

# ASSESSMENT: CORRELATION

# Question 1

# While studying heredity, Francis Galton developed what important statistical
# concept?

# Standard deviation
# Normal distribution
# Correlation [X]
# Probability

# Question 2

# The correlation coefficient is a summary of what?

# The trend between two variables [X]
# The dispersion of a variable
# The central tendency of a variable
# The distribution of a variable

# Question 3

# Below is a scatter plot showing the relationship between two variables, x and y.

# From the figure, the correlation between x and y appears to be about:

# -0.9 [X]
# -0.2
# 0.2
# 2

# Question 4

# Instead of running a Monte Carlo simulation with a sample size of 25 from the
# 179 father-son pairs described in the videos, now run our simulation with a
# sample size of 50.

# Would you expect the `mean` of our sample correlation to increase, decrease,
# or stay approximately the same?

# Increase
# Decrease
# Stay approximately the same [X]

# Question 5

# Instead of running a Monte Carlo simulation with a sample size of 25 from the
# 179 father-son pairs described in the videos, now run our simulation with a
# sample size of 50.

# Would you expect the `standard deviation` of our sample correlation to increase, 
# decrease, or stay approximately the same?

# Increase
# Decrease [X]
# Stay approximately the same

# Question 6

# If X and Y are completely independent, what do you expect the value of the
# correlation coefficient to be?

# -1
# -0.5
# 0 [X]
# 0.5
# 1
# Not enough information to answer this question

# Question 7

# Load the Lahman library. Filter the Teams data frame to include years from 1961
# to 2001
library(Lahman)

# What is the correlation coefficient between number of runs per game and 
# number of at bats per game?

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(rpg = R / G, abpg = AB / G) %>%
  summarize(r = cor(abpg, rpg)) # 0.658

# Question 8

# Use the filtered Teams data frame from Question 7.

# What is the correlation coefficient between win rate (number of wins per game)
# and number of errors per game?

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(wpg = W / G, epg = E / G) %>%
  summarize(r = cor(wpg, epg)) # -0.339


# Question 9

# Use the filtered Teams data frame from Question 7.

# What is the correlation coefficient between doubles (X2B) per game
# and triples (X3B) per game?

Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(X2Bpg = X2B / G, X3Bpg = X3B / G) %>%
  summarize(r = cor(X2Bpg, X3Bpg)) # -0.011

# 1.3 STRATIFICATION AND VARIANCE EXPLAINED

# ANSCOMBE'S QUARTET/STRATIFICATION

# Correlation is not always a good summary of the relationship between two
# variables.

# The general idea of conditional expectation is that we stratify a population
# into groups and compute summaries in each group.

# A practical way to improve the estimates of the conditional expectations is to
# define strata of with similar values of x.

# If there is perfect correlation, the regression line predicts an increase that 
# is the same number of SDs for both variables. If there is 0 correlation, then
# we don't use x at all for the prediction and simply predict the average `mu`y.
# For values between 0 and 1, the prediction is somewhere in between. If the
# correlation is negative, we predict a reduction instead of an increase.

# number of fathers with height 72 or 72.5 inches.
sum(galton_heights$father == 72)
sum(galton_heights$father == 72.5)

# predicted height of a son with a 72 inch tall father
conditional_avg <- galton_heights %>%
  filter(round(father) == 72) %>%
  summarize(avg = mean(son)) %>%
  pull(avg)
conditional_avg

# stratify fathers' heights to make a boxplot of son heights
galton_heights %>% mutate(father_strata = factor(round(father))) %>%
  ggplot(aes(father_strata, son)) +
  geom_boxplot() +
  geom_point()

# center of each boxplot
galton_heights %>%
  mutate(father = round(father)) %>%
  group_by(father) %>%
  summarize(son_conditional_avg = mean(son)) %>%
  ggplot(aes(father, son_conditional_avg)) +
  geom_point()

# calculate values to plot regression line on original data
mu_x <- mean(galton_heights$father)
mu_y <- mean(galton_heights$son)
s_x <- sd(galton_heights$father)
s_y <- sd(galton_heights$son)
r <- cor(galton_heights$father, galton_heights$son)
m <- r * s_y / s_x
b <- mu_y - m * mu_x

# add regression line to the plot
galton_heights %>%
  ggplot(aes(father, son)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = b, slope = m)

# BIVARIATE NORMAL DISTRIBUTION

# When a pair of random variables are approximated by the bivariate normal
# distribution, scatterplots look like ovals. They can be thin (high correlation)
# or circle-shaped (no correlation).

# Whent wo variables follow a bivariate normal distribution, computing the
# regression line is equivalent to computing conditional expectations.

# We can obtain a much more stable estimate of the conditional expectation by
# finding the regression line and using it to make predictions.

galton_heights %>%
  mutate(z_father = round((father - mean(father)) / sd(father))) %>%
  filter(z_father %in% -2:2) %>%
  ggplot() +
  stat_qq(aes(sample = son)) +
  facet_wrap(~ z_father)

# VARIANCE EXPLAINED

# Conditioning on a random variable X can help to reduce variance of response
# variable Y.

# The standard deviation of the conditional distribution is 
# SD (Y|X=x) = `sigma`y * sqrt(1-`ro`^2), which is smaller than the standard
# deviation without conditioning `sigma`y.

# Because variance is the standard deviation squared, the variance of the
# conditional distribution is `sigma`y^2 * (1 - `ro`^2).

# In the statement "X explains such and such percent of variability," the percent
# value refers to the variance. The variance decreases by `ro`^2 percent.

# The "variance explained" statement only makes sense when the data is
# approximated by a bivariate normal distribution.

# THERE ARE TWO REGRESSION LINES

# There are two different regression lines depending on whether we are taking the
# expectation of Y given X or taking the expectation of X given Y.

# compute a regression line to predict the son's height from the father's height
mu_x <- mean(galton_heights$father)
mu_y <- mean(galton_heights$son)
s_x <- sd(galton_heights$father)
s_y <- sd(galton_heights$son)
r <- cor(galton_heights$father, galton_heights$son)
m_1 <- r * s_y / s_x
b_1 <- mu_y - m_1 * mu_x

# compute a regression line to predict the father's height from the son's height
m_2 <- r * s_x / s_y
b_2 <- mu_x - m_2 * mu_y

# ASSESSMENT: STRATIFICATION AND VARIANCE EXPLAINED PART 1

# Question 1

# Look at the figure below.

# The slope of the regression line in this figure is equal to what, in words?

# Slope = (correlation coefficient of son and father heights) * (standard deviation
# of sons' heights / standard deviation of fathers' heights) [X]

# Slope = (correlation coefficient of son and father heights) * (standard deviation
# of fathers' heights / standard deviation of son's heights)

# Slope = (correlation coefficient of son and father heights) / (standard deviation
# of sons' heights * standard deviation of fathers' heights)

# Slope = (mean height of fathers) - (correlation coefficient of son and father
# heihts * mean height of sons)

# Question 2

# Why does the regression line simplify to a line with intercept zero and slope
# `ro` when we standardize our x and y variables?

# When we standardize variables, both x and y will have a mean of one and a 
# standard deviation of zero. When you substitute this into the formula for the
# regression line, the terms cancel out until we have the following equation:
# yi = `ro`xi

# When we standardize variables, both x and y will have a mean of zero and a
# standard deviation of one. When you substitute this into the formula for the
# regression line, the terms cancel out until we have the following equation:
# yi = `ro`xi [X]

# When we standardize variables, both x and y will have a mean of zero and a
# standard deviation of one. When you substitute this into the formula for the
# regression line, the terms cancel out until we have the following equation:
# yi = `ro` + xi

# Regression equation: yi = b + m * xi
# Replacing b and m:   yi = mu_y - (`ro` * s_y / s_x) * mu_x + (`ro` * s_y / s_x) * xi
# Repl. standardized:  yi = 0 - (`ro`) * 0 + (`ro`) * xi = `ro` * xi

# Question 3

# What is a limitation of calculating conditional means? Select all that apply.

# Each stratum we condition on (e.g., a specific father's height) may not have
# many data points. [X]

# Because there are limited data points for each stratum, our average values
# have large standard errors. [X]

# Conditional means are less stable than a regression line. [X]

# Conditional means are a useful theoretical tool but cannot be calculated.

# Question 4

# A regression line is the best prediction of Y given we know the value of X
# when:

# X and Y follow a bivariate normal distribution. [X]

# Both X and Y are normally distributed.

# Both X and Y have been standardized.

# There are at least 25 X-Y pairs.

# Question 5

# Which one of the following scatterplots depicts an x and y distribution that
# is NOT well-approximated by the bivariate normal distribution.

# U-shaped scatter plot [X]
# Positive, low correlation distribution
# Circle, zero-correlation distribution
# Negative, high correlation

# Question 6

# We previously calculated that the correlation coefficient `ro` between
# fathers' and sons' heights is 0.5.

# Given this, what percent of the variation in son's heights is explained by
# fathers' heights?

# 0%
# 25% [X] `ro`^2 * 100
# 50%
# 75%

# Question 7

# Suppose the correlation between father and son's height is 0.5, the standard
# deviation of fathers' heights is 2 inches, and the standard deviation of
# sons' heights is 3 inches.

# Given a one inch increase in a father's height, what is the predicted change
# in the son's height?

# 0.333
# 0.5
# 0.667
# 0.75 [X]
# 1
# 1.5

# m = `ro` * s_y / s_x
0.5 * 3 / 2

# ASSESSMENT: STRATIFICATION AND VARIANCE EXPLAINED PART 2

# In the second part of this assessment, you'll analyze a set of mother and
# daughter heights, also from GaltonFamilies.

# Define female_heights, a set of mother and daughter heights sampled from
# GaltonFamilies as follows:
set.seed(1989, sample.kind = "Rounding")
library(HistData)
data("GaltonFamilies")

female_heights <- GaltonFamilies %>%
  filter(gender == "female") %>%
  group_by(family) %>%
  sample_n(1) %>%
  ungroup() %>%
  select(mother, childHeight) %>%
  rename(daughter = childHeight)

# Question 8

# Calculate the mean and standard deviation of mothers' heights, the mean
# and standard deviation of daughters' heights, and the correlation coefficient
# between mother and daughter heights
mean(female_heights$mother) # 64.125
mean(female_heights$daughter) # 64.28011
sd(female_heights$mother) # 2.289
sd(female_heights$daughter) # 2.394
cor(female_heights$mother, female_heights$daughter) # 0.324

# Question 9

# Calculate the slope and intercept of the regression line predicting daughters'
# heights given mothers' heights. Given an increase in mother's height by 1 inch,
# how many inches is the daughter's height expected to change?

mu_x <- mean(female_heights$mother)
mu_y <- mean(female_heights$daughter)
s_x <- sd(female_heights$mother)
s_y <- sd(female_heights$daughter)
r <- cor(female_heights$mother, female_heights$daughter)

# Slope of regression line predicting daughters' height from mothers' heights
m <- r * s_y / s_x
m # 0.339

# Intercept of regression line predicting daughters' height from mothers' heights
b <- mu_y - m * mu_x
b # 42.517

# Change in daughter's height in inches given a 1 inch increase in the mother's
# height.
m # 0.339

# y = b + m * x; y = (b + m * (x+1)) - (b + m * x);
# y = m * (x + 1) - m * x; y = m * x + m - m * x; y = m

# Question 10

# What percent of the variability in daughter heights is explained by the
# mother's height?
r^2 * 100 # 10.531

# Question 11

# A mother has a height of 60 inches.

# Using the regression formula, what is the conditional expected value of her
# daughter's height given the mother's height?

# y = b + m * 60
b + m * 60 # 62.88

## SECTION 2: LINEAR MODELS

# LINEAR MODELS OVERVIEW

# In the linear models section, you will learn:

# Use multivariate regression to adjust for confounders.

# Write linear models to describe the relationship between two or more variables.

# Calculate the least squares estimates for a regression model using the lm function.

# Understand the differences between tibbles and data frames.

# Use the do() function to bridge R functions and the tidyverse.

# Use the tidy(), glance(), and augment() functions from the broom package.

# Apply linear regression to measurement error models.

# 2.1. INTRODUCTION TO LINEAR MODELS

# CONFOUNDING: ARE BBs MORE PREDICTIVE?

# Association is not causation!

# Although it may appear that BB cause runs, it is actually the HR that cause
# most of these runs. We say that BB are confounded with HR.

# Regression can help us account for confounding.

# find regression line for predicting runs from BBs
library(tidyverse)
library(Lahman)
bb_slope <- Teams %>%
  filter(yearID %in% 1961:2001) %>%
  mutate(BB_per_game = BB/G, R_per_game = R/G) %>%
  lm(R_per_game ~ BB_per_game, data = .) %>%
  .$coef %>%
  .[2]
bb_slope

# compute regression line for predicting runs from singles
singles_slope <- Teams %>%
  filter(yearID %in% 1961:2001) %>%
  mutate(Singles_per_game = (H-HR-X2B-X3B)/G, R_per_game = R/G) %>%
  lm(R_per_game ~ Singles_per_game, data = .) %>%
  .$coef %>%
  .[2]
singles_slope

# calculate correlation between HR, BB and Singles
Teams %>%
  filter(yearID %in% 1961:2001) %>%
  mutate(Singles = (H-HR-X2B-X3B)/G, BB = BB/G, HR = HR/G) %>%
  summarize(cor(BB, HR), cor(Singles, HR), cor(BB, Singles))

# STRATIFICATION AND MULTIVARIATE REGRESSION

# A first approach to check confounding is to keep HRs fixed at a certain value
# and then examine the relationship between BB and runs.

# The slopes of BB after stratifying on HR are reduced, but they are not 0, which
# indicates that BB are helpful for producing runs, just not as much as previously
# thought.

# stratify HR per game to nearest 10, filter out strata with few points
dat <- Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(HR_strata = round(HR/G, 1),
         BB_per_game = BB/G,
         R_per_game = R/G) %>%
  filter(HR_strata >= 0.4 & HR_strata <= 1.2)

# scatterplot for each HR stratum
dat %>%
  ggplot(aes(BB_per_game, R_per_game)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  facet_wrap(~ HR_strata)

# calculate slope of regression line after stratifying by HR
dat %>%
  group_by(HR_strata) %>%
  summarize(slope = cor(BB_per_game, R_per_game) * sd(R_per_game) / sd(BB_per_game))

# stratify by BB
dat <- Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(BB_strata = round(BB/G, 1),
         HR_per_game = HR/G,
         R_per_game = R/G) %>%
  filter(BB_strata >= 2.8 & BB_strata <= 3.9)

# scatterplot for each BB stratum
dat %>% ggplot(aes(HR_per_game, R_per_game)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  facet_wrap(~ BB_strata)

# slope of regression line after stratifying by BB
dat %>%
  group_by(BB_strata) %>%
  summarize(slope = cor(HR_per_game, R_per_game) * sd(R_per_game) / sd(HR_per_game))

# LINEAR MODELS

# "Linear" here does not refer to lines, but rather to the fact that the
# conditional expectation is a linear combination of known quantities.

# In Galton's model, we assume Y (son's height) is a linear combination of a
# constant and X (father's height) plus random noise. We further assume that
# `epsilon`i are independent from each other, have expected value of 0 and the
# standard deviation `sigma` does not depend on i.

# Note that if we further assume that `epsilon` is normally distributed, then
# the model is exactly the same one we derived earlier by assuming bivariate
# normal data.

# We can substract the mean from X to make `beta`0 more interpretable.

# ASSESSMENT: INTRODUCTION TO LINEAR MODELS

# Question 1

# As described in the videos, when we stratified our regression lines for runs
# per game vs. bases on balls by the number of home runs, what happened?

# The slope of runs per game vs. bases on balls within each stratum was reduced
# because we removed confounding by home runs. [X]

# The slope of runs per game vs. bases on balls within each stratum was reduced
# because there were fewer data points.

# The slope of runs per game vs. bases on balls within each stratum increased
# after we removed confounding by home runs.

# The slope of runs per game vs. bases on balls within each stratum stayed
# about the same as the original slope.

# Question 2

# We run a linear model for sons' heights vs. fathers' heights using the Galton
# height data, and get the following results:

# >lm(son ~ father, data = galton_heights)
# Call:
# lm(formula = son ~ father, data = galton_heights)
# Coefficients:
# (Intercept) (father)
# 35.71        0.50

# Interpret the numeric coefficient for "father".

# For every inch we increase the son's height, the predicted father's height
# increases by 0.5 inches.

# For every inch we increase the father's height, the predicted son's height
# grows by 0.5 inches. [X]

# For every inch we increase the father's height, the predicted son's height
# is 0.5 times greater.

lm(son ~ father, data = galton_heights)
lm(father ~ son, data = galton_heights)

# Question 3

# We want the intercept term for our model to be more interpretable, so we run
# the same model as before but now we substract the mean of fathers' heights
# from each individual father's height to create a new variable centered at zero.
galton_heights <- galton_heights %>%
  mutate(father_centered = father - mean(father))

# We run a linear model using this centered fathers' height variable.

# >lm(son ~ father_centered, data = galton_heights)
# Call:
# lm(formula = son ~ father_centered, data = galton_heights)
# Coefficients:
# (Intercept) father_centered
# 70.45       0.50

# Interpret the numeric coefficient for the intercept.

# The height of a son of a father of average height is 70.45 inches. [X]

# The height of a son when a father's height is zero is 70.45 inches.

# The height of an average father is 70.45 inches.

# Question 4

# Suppose we fit a multivariate regression model for expected runs based on
# BB and HR:

# E[R|BB = x1, HR = x2] = `beta`0 + `beta`1 * x1 + `beta`2 * x2

# Suppose we fix BB = x1. Then we observe a linear relationship between runs
# and HR with intercept of: 

# `beta`0
# `beta`0 + `beta`2 * x2
# `beta`0 + `beta`1 * x1 [X]
# `beta`0 + `beta`2 * x1

# Question 5

# Which of the following are assuptions for the errors `epsilon`i in a linear
# regression model? Select all that apply.

# The `epsilon`i are independent of each other [X]
# The `epsilon`i have expected value of 0 [X]
# The variance of `epsilon`i is a constant [X]

# 2.2. LEAST SQUARES ESTIMATES

# LEAST SQUARES ESTIMATES

# For regression data, we aim to find the coefficient values that minimize the distance of the fitted
# model to the data.

# Residual sum of squares (RSS) measures the distance between the true value and the predicted value
# given by the regression line. The values that minimize the RSS are called the least squares estimates (LSE).

# We can use partial derivatives to get the values for `beta`0 and `beta`1 in Galton's data.

# compute RSS for any pair of `beta`0 and `beta`1 in Galton's data
library(HistData)
data("GaltonFamilies")
set.seed(1983, sample.kind = "Rounding")
galton_heights <- GaltonFamilies %>%
  filter(gender == "male") %>%
  group_by(family) %>%
  sample_n(1) %>%
  ungroup() %>%
  select(father, childHeight) %>%
  rename(son = childHeight)
rss <- function(beta0, beta1){
  resid <- galton_heights$son - (beta0+beta1*galton_heights$father)
  return(sum(resid^2))
}

# plot RSS as a function of `beta`1 when `beta`0 = 25
beta1 = seq(0,1,len=nrow(galton_heights))
results <- data.frame(beta1 = beta1,
                      rss = sapply(beta1, rss, beta0 = 25))
results %>% ggplot(aes(beta1, rss)) + geom_line() +
  geom_line(aes(beta1, rss))

# THE lm FUNCTION

# When calling the lm function, the variable that we want to predict is put to the left of the ~ symbol,
# and the variables we use to predict are put on the right of the ~ symbol. The intercept is added
# automatically.

# LSEs are random variables.

# fit regression line to predict son's height from father's height
fit <- lm(son ~ father, data = galton_heights)
fit

# summary statistics
summary(fit)

# LSE ARE RANDOM VARIABLES

# Because they are derived from the samples, LSE are random variables.

# `beta`0 and `beta`1 appear to be normally distributed because the central limit theorem plays a role.

# The t-statistic depends on the assumption that `epsilon` follows a normal distribution.

# Monte Carlo simulation
B <- 1000
N <- 50
lse <- replicate(B, {
  sample_n(galton_heights, N, replace = TRUE) %>%
    lm(son ~ father, data = .) %>%
    .$coef
})
lse <- data.frame(beta_0 = lse[1,], beta_1 = lse[2,])

# Plot the distribution of `beta`0 and `beta`1
library(gridExtra)
p1 <- lse %>% ggplot(aes(beta_0)) + geom_histogram(binwidth = 5, color = "black")
p2 <- lse %>% ggplot(aes(beta_1)) + geom_histogram(binwidth = 0.1, color = "black")
grid.arrange(p1, p2, ncol = 2)

# summary statistics
sample_n(galton_heights, N, replace = TRUE) %>%
  lm(son ~ father, data = .) %>%
  summary %>%
  .$coef

# ADVANCED NOTE ON LSE

# Although interpretation is not straightforward, it is also useful to know that the LSE can be
# strongly correlated, which can be seen using this code.
lse %>% summarize(cor(beta_0, beta_1))

# However, the correlation depends on how the predictors are defined or transformed.

# Here we standardize the father heights, which changes xi to xi - x_hat.
B <- 1000
N <- 50
lse <- replicate(B, {
  sample_n(galton_heights, N, replace = TRUE) %>%
    mutate(father = father - mean(father)) %>%
    lm(son ~ father, data = .) %>% .$coef
})

# Observe what happens to the correlation in this case.
cor(lse[1,], lse[2,])

# PREDICTED VARIABLES ARE RANDOM VARIABLES

# The predicted value is often denoted as Y^, which is a random variable. Mathematical theory tells us
# what the standard error of the predicted value is. 

# The predict() function in R can give us predictions directly.

# plot predictions and confidence intervals
galton_heights %>% ggplot(aes(father, son)) +
  geom_point() +
  geom_smooth(method = "lm")

# predict Y directly
fit <- galton_heights %>% lm(son ~ father, data = .)
Y_hat <- predict(fit, se.fit = TRUE)
names(Y_hat)

# plot best fit line
galton_heights %>%
  mutate(Y_hat = predict(lm(son ~ father, data = .))) %>%
  ggplot(aes(father, Y_hat)) +
  geom_line()

# ASSESSMENT: LEAST SQUARES ESTIMATES, PART 1

# Question 1

# The following code was used in the video to plot RSS with `beta`0 = 25.
beta1 = seq(0, 1, len=nrow(galton_heights))
results <- data.frame(beta1 = beta1,
                      rss = sapply(beta1, rss, beta0 = 25))
results %>% ggplot(aes(beta1, rss)) + geom_line() +
  geom_line(aes(beta1, rss), color = 2)

# In a model for sons' heights vs fathers' heights, what is the least square estimate (LSE) for
# `beta`1 if we assume that `beta`0 is 36?

resultsQ1 <- data.frame(beta1 = beta1,
                        rss = sapply(beta1, rss, beta0 = 36))
resultsQ1 %>% ggplot(aes(beta1, rss)) + geom_line() +
  geom_line(aes(beta1, rss), color = 2) # 0.5

# Question 2

# The least squares estimates for the parameters `beta`0, `beta`1,...,`beta`n ??? the residual sum of
# squares.

# maximize
# minimize [X]
# equal

# Question 3

# Load the Lahman library and filter the Teams dataset to the years 1961-2001. Run a linear model in R
# predicting the number of runs per game based on both the number of bases on balls per game and the 
# number of home runs per game.

library(Lahman)
Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(Rpg = R / G, BBpg = BB / G, HRpg = HR / G) %>%
  lm(Rpg ~ BBpg + HRpg, data = .) %>% .$coef

# What is the coefficient for bases on balls?

# 0.39 [X]
# 1.56
# 1.74
# 0.027

# Question 4

# We run a Monte Carlo simulation where we repeatedly take samples of N = 100 from the Galton heights
# data and compute the regression slope coefficients for each sample.
B <- 1000
N <- 100
lse <- replicate(B, {
  sample_n(galton_heights, N, replace = TRUE) %>%
    lm(son ~ father, data = .) %>% .$coef
})
lse <- data.frame(beta_0 = lse[1,], beta_1 = lse[2,])

# What does the central limit theorem tell us about the variables beta_0 and beta_1? Select all that
# apply.

# They are approximately normally distributed. [X]
# The expected value of each is the true value of `beta`0 and `beta`1 (assuming the Galton heights
# data is a complete population) [X]
# The central limit theorem does not apply in this situation.
# It allows us to test the hypothesis that `beta`0 = 0 and `beta`1 = 0

# Question 5

# Which R code(s) below would properly plot the predictions and confidence intervals for our lineal
# model of sons' heights?

# 1
galton_heights %>% ggplot(aes(father, son)) +
  geom_point() +
  geom_smooth() # INCORRECT, uses loess method

# 2
galton_heights %>% ggplot(aes(father, son)) +
  geom_point() +
  geom_smooth(method = "lm") # [X]

# 3
model <- lm(son ~ father, data = galton_heights)
predictions <- predict(model, interval = c("confidence"), level = 0.95)
data <- as_tibble(predictions) %>% bind_cols(father = galton_heights$father)

ggplot(data, aes(x = father, y = fit)) +
  geom_line(color = "blue", size = 1) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), alpha = 0.2) +
  geom_point(data = galton_heights, aes(x = father, y = son)) # [X]

# 4
model <- lm(son ~ father, data = galton_heights)
predictions <- predict(model)
data <- as_tibble(predictions) %>% bind_cols(father = galton_heights$father)

ggplot(data, aes(x = father, y = fit)) +
  geom_line(color = "blue", size = 1) +
  geom_point(data = galton_heights, aes(x = father, y = son)) # INCORRECT

# ASSESSMENT: LEAST SQUARES ESTIMATES, PART 2

# In questions 7 and 8, you'll look again at female heights from GaltonFamilies.

# Define female_heights, a set of mother and daughter heights sampled from the GaltonFamilies, as
# follows:
set.seed(1989, sample.kind = "Rounding")
library(HistData)
data("GaltonFamilies")
options(digits = 3)

female_heights <- GaltonFamilies %>%
  filter(gender == "female") %>%
  group_by(family) %>%
  sample_n(1) %>%
  ungroup() %>%
  select(mother, childHeight) %>%
  rename(daughter = childHeight)

# Question 7

# Fit a linear regression model predicting the mothers' height using daughters' heights.

lm(mother ~ daughter, data = female_heights)

# What is the slope of the model? 0.31

# What is the intercept of the model? 44.18

# Question 8

# Predict mothers' heights using the model.

fit <- lm(mother ~ daughter, data = female_heights)
fit$coef[1] # intercept
fit$coef[2] # slope

# What is the predicted height of the first mother in the data set?

fit$coef[1] + fit$coef[2] * female_heights$daughter[1] # 65.6

# Also...
predict(fit)[1]

# What is the actual height of the first mother in the data set?

female_heights$mother[1] # 67

# We have shown how BB and singles have similar predictive power for scoring runs. Another way to compare
# the usefulness of these baseball metrics is by assessing how stable they are across the years. Because
# we have to pick players based on their previous performances, we will prefer metrics that are more stable.
# In these exercises, we will compare the stability of singles and BB. 

# Before we get started, we want to generate two tables: one for 2002 and another for the average of the
# 1999-2001 seasons. We want to define per plate appearance statistics, keeping only players with more
# than 100 plate appearances. Here is how we create the 2002 table.
library(Lahman)
bat_02 <- Batting %>% filter(yearID == 2002) %>%
  mutate(pa = AB + BB, singles = (H - X2B - X3B - HR)/pa, bb = BB/pa) %>%
  filter(pa > 100) %>%
  select(playerID, singles, bb)

# Question 9

# Now compute a similar table but with rates computed over 1999-2001. Keep only rows from 1999-2001 where
# players have 100 or more plate appearances, calculate each player's single rate and BB rate per stint
# where each row is one stint - a player can have multiple stints per season), then calculate the 
# average single rate (mean_singles) and average BB rate (mean_bb) per player over the three year period.

bat_99_01 <- Batting %>% filter(yearID %in% 1999:2001) %>%
  mutate(pa = AB + BB) %>%
  filter(pa >= 100) %>%
  mutate(singles = (H - X2B - X3B - HR)/pa, bb = BB/pa) %>%
  select(playerID, yearID, stint, pa, singles, bb)

# How many players had a single rate mean_singles of greater than 0.2 per plate appearance over 1999-2001?

bat_99_01 %>%
  group_by(playerID) %>%
  summarize(mean_singles = mean(singles), mean_bb = mean(bb)) %>%
  summarize(n = sum(mean_singles > 0.2)) %>% pull(n) # 46

# How many players had a bb rate mean_bb of greater than 0.2 per plate appearance over 1999-2001?

bat_99_01 %>%
  group_by(playerID) %>%
  summarize(mean_singles = mean(singles), mean_bb = mean(bb)) %>%
  summarize(n = sum(mean_bb > 0.2)) %>% pull(n) # 3

# Recreate table for question 10

bat_99_01 <- Batting %>% filter(yearID %in% 1999:2001) %>%
  mutate(pa = AB + BB, singles = (H - X2B - X3B - HR)/pa, bb = BB/pa) %>%
  filter(pa >= 100) %>%
  group_by(playerID) %>%
  summarize(mean_singles = mean(singles), mean_bb = mean(bb))

# Question 10

# Use inner_join() to combine the bat_02 table with the table of 1999-2001 rate averages you created in
# the previous question.

table <- bat_02 %>% inner_join(bat_99_01)

# What is the correlation between 2002 singles rates and 1999-2001 average singles rates?

cor(table$singles, table$mean_singles) # 0.551

# What is the correlation between 2002 BB rates and 1999-2001 average BB rates?

cor(table$bb, table$mean_bb) # 0.717

# Question 11

# Make scatterplots of mean_singles versus singles and mean_bb versus bb.
library(gridExtra)
p1 <- table %>% ggplot(aes(mean_singles, singles)) + geom_point()
p2 <- table %>% ggplot(aes(mean_bb, bb)) + geom_point()
grid.arrange(p1, p2, ncol = 2)

# Are either of these distributions bivariate normal?

# Neither distribution is bivariate normal.
# singles and mean_singles are bivariate normal, but bb and mean_bb are not.
# bb and mean_bb are bivariate normal, but singles and mean_singles are not.
# Both distributions are bivariate normal. [X]

# Question 12

# Fit a linear model to predict 2002 singles given 1999-2001 mean_singles.

fit_singles <- lm(singles ~ mean_singles, data = table)

# What is the coefficient of mean_singles, the slope of the fit?

fit_singles$coef[2] # 0.588

# Fit a linear model to predict 2002 bb given 1999-2001 mean_bb.

fit_bb <- lm(bb ~ mean_bb, data = table)

# What is the coefficient of mean_bb, the slope of the fit?

fit_bb$coef[2]

# 2.3. TIBBLES, DO, AND BROOM

# ADVANCED DPLYR: TIBBLES

# Tibbles can be regarded as a modern version of data frames and are the default data structure
# in the tidyverse.

# Some functions that do not work properly with data frames do work with tibbles.

# stratify by HR
dat <- Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(HR = round(HR/G, 1),
         BB = BB/G,
         R = R/G) %>%
  select(HR, BB, R) %>%
  filter(HR >= 0.4 & HR <= 1.2)

# calculate the slope of regression lines to predict runs by BB in different HR strata
dat %>%
  group_by(HR) %>%
  summarize(slope = cor(BB,R)*sd(R)/sd(BB))

# use lm to get estimated slopes - lm does not work with grouped tibbles
dat %>%
  group_by(HR) %>%
  lm(R ~ BB, data = .) %>%
  .$coef

# inspect grouped tibble
dat %>% group_by(HR) %>% head()
dat %>% group_by(HR) %>% class()

# TIBBLES: DIFFERENCES FROM DATA FRAMES

# Tibbles are more readable than data frames.

# If you subset a data frame, you may not get a data frame. If you subset a tibble, you always
# get a tibble.

# Tibbles can hold more complex objects such as lists or functions.

# Tibbles can be grouped.

# inspect data frame and tibble
Teams
as_tibble(Teams)
# Note than the function was formerly called as.tibble()

# subsetting a data frame sometimes generates vectors
class(Teams[,20])

# subsetting a tibble always generates tibbles
class(as_tibble(Teams[,20]))

# pulling a vector out of tibble
class(as_tibble(Teams)$HR)

# access a non-existing column in a data frame or a tibble
Teams$hr
as_tibble(Teams)$hr

# create a tibble with complex objects
tibble(id = c(1, 2, 3), func = c(mean, median, sd))

# DO

# The do() function serves as a bridge between R functions, such as lm(), and the tidyverse.

# We have to specify a column when using the do() function, otherwise we will get an error.

# If the data frame being returned has more than one row, the rows will be concatenated appropriately.

# use do to fit a regression line to each HR stratum
dat %>%
  group_by(HR) %>%
  do(fit = lm(R ~ BB, data = .))

# using without a column name gives an error
dat %>%
  group_by(HR) %>%
  do(lm(R ~ BB, data = .))

# define a function to extract the slope from lm
get_slope <- function(data){
  fit <- lm(R ~ BB, data = data)
  data.frame(slope = fit$coefficients[2],
             se = summary(fit)$coefficient[2,2])
}

# return the desired data frame
dat %>%
  group_by(HR) %>%
  do(get_slope(.))

# not the desired output: a column containing data frames
dat %>%
  group_by(HR) %>%
  do(slope = get_slope(.))

# data frames with multiple rows will be concatenated appropriately
get_lse <- function(data){
  fit <- lm(R ~ BB, data = data)
  data.frame(term = names(fit$coefficients),
             estimate = fit$coefficients,
             se = summary(fit)$coefficients[,2])
}

dat %>%
  group_by(HR) %>%
  do(get_lse(.))

# BROOM

# The broom package has three main functions, all of which extract information from the object returned
# by lm and return it in a tidyverse friendly data frame.

# The tidy() function returns estimates and related information as a data frame.

# The functions glance() and augment() relate to model specific and observation specific outcomes
# respectively.

# use tidy to return lm estimates and related information as a data frame
library(broom)
fit <- lm(R ~ BB, data = dat)
tidy(fit)

# add confidence intervals with tidy
tidy(fit, conf.int = TRUE)

# pipeline with lm, do, tidy
dat %>%
  group_by(HR) %>%
  do(tidy(lm(R ~ BB, data = .), conf.int = TRUE)) %>%
  filter(term == "BB") %>%
  select(HR, estimate, conf.low, conf.high)

# make ggplots
dat %>%
  group_by(HR) %>%
  do(tidy(lm(R ~ BB, data = .), conf.int = TRUE)) %>%
  filter(term == "BB") %>%
  select(HR, estimate, conf.low, conf.high) %>%
  ggplot(aes(HR, y = estimate, ymin = conf.low, ymax = conf.high)) +
  geom_errorbar() +
  geom_point()

# ASSESSMENT: TIBBLES, DO, AND BROOM, PART 1

# Question 1

# As seen in the videos, what problem do we encounter when we try to run a linear model on our
# baseball data, grouping by home runs?

# There is not enough data in some levels to run the model.
# The lm() function does not know how to handle group tibbles. [X]
# The results of the lm() function cannot be put into a tidy format.

# Question 2

# Tibbles are similar to what other class in R?

# Vectors
# Matrices
# Data frames [X]
# Lists

# Question 3

# What are some advantages of tibbles compared to data frames? Select all that apply.

# Tibbes display better. [X]
# If you subset a tibble, you always get back a tibble. [X]
# Tibbles can have complex entries. [X]
# Tibbles can be grouped. [X]

# Question 4

# What are two advantages of the do() command, when applied to the tidyverse? Select two.

# It is faster than normal functions.
# It returns useful error messages.
# It understands grouped tibbles. [X]
# It always returns a data.frame. [X]

# Question 5

# You want to take the tibble dat, which we used in the video on the do() function, and run a linear
# model R ~ BB for each strata of HR. Then you want to add three new columns to your grouped tibble:
# the coefficients, standard error, and p-value for the BB term in the model.

# You've already written the function get_slope(), shown below:
get_slope <- function(data){
  fit <- lm(R ~ BB, data = data)
  sum.fit <- summary(fit)
  data.frame(slope = sum.fit$coefficients[2, "Estimate"],
             se = sum.fit$coefficients[2, "Std. Error"],
             pvalue = sum.fit$coefficients[2, "Pr(>|t|)"])
}

# What additional code could you write to accomplish your goal?

# 1
dat %>%
  group_by(HR) %>%
  do(get_slope)

# 2
dat %>%
  group_by(HR) %>%
  do(get_slope(.)) # [X]

# 3
dat %>%
  group_by(HR) %>%
  do(slope = get_slope(.))

# 4
dat %>%
  do(get_slope(.))

# Question 6

# The output of a broom function is always what?

# A data frame
# A list
# A vector

# Question 7

# You want to know whether the relationship between home runs and runs per game varies varies by
# baseball league. You create the following data set:
dat <- Teams %>% filter(yearID %in% 1961:2001) %>%
  mutate(HR = HR / G,
         R = R / G) %>%
  select(lgID, HR, BB, R)

# What code would quickly help you answer this question?

# 1
dat %>%
  group_by(lgID) %>%
  do(tidy(lm(R ~ HR, data = .), conf.int = T)) %>%
  filter(term == "HR") # [X]

# 2
dat %>%
  group_by(lgID) %>%
  do(glance(lm(R ~ HR, data = .)))

# 3
dat %>%
  do(tidy(lm(R ~ HR, data = .), conf.int = T)) %>%
  filter(term == "HR")

# 4
dat %>%
  group_by(lgID) %>%
  do(mod = lm(R ~ HR, data = .))

# ASSESSMENT: TIBBLES, DO, AND BROOM, PART 2

# We have investigated the relationship between fathers' height and sons' heights. But what about other
# parent-child relationships? Does one parent's height have a stronger association with child height? How
# does the child gender's affect this relationship in heights? Are any differences that we observe
# statistically significant?

# The galton dataset is a sample of one male and one female child from each family in the GaltonFamilies
# dataset. The pair column denotes whether the pair is father and daugther, father and son, mother and
# daughter, or mother and son.

# Create the galton dataset using the code below:
library(tidyverse)
library(HistData)
data("GaltonFamilies")
set.seed(1, sample.kind = "Rounding")
galton <- GaltonFamilies %>%
  group_by(family, gender) %>%
  sample_n(1) %>%
  ungroup() %>%
  gather(parent, parentHeight, father:mother) %>%
  mutate(child = ifelse(gender == "female", "daughter", "son")) %>%
  unite(pair, c("parent", "child"))
galton

# Question 8

# Group by pair and summarize the number of observations in each group.

galton %>%
  group_by(pair) %>%
  summarize(n())

# How many father-daughter pairs are in the dataset? 176
# How many mother-son pairs are in the dataset? 179

# Question 9

# Calculate the correlation coefficients for fathers and daughters, fathers and sons, mothers and 
# daughters, and mothers and sons.

galton %>% group_by(pair) %>%
  summarize(cor(parentHeight, childHeight))

# Which pair has the strongest correlation in heights?

# fathers and daughters
# fathers and sons [X] .430
# mothers and daughters
# mothers and sons

# Which pair has the weakest correlation in heights?

# fathers and daughters
# fathers and sons
# mothers and daughters
# mothers and sons [X] .343

# Question 10 has two parts. The information here applies to both parts.

# Use lm() and the broom package to fit regression lines for each parent-child pair type. Compute the
# least squares estimates, standard errors, confidence intervals, and p-values for the parentHeight
# coefficient for each pair.

galton %>%
  group_by(pair) %>%
  do(tidy(lm(childHeight ~ parentHeight, data = .), conf.int = TRUE)) %>%
  filter(term == "parentHeight")

# Question 10a

# What is the estimate of the father-daughter coefficient? 0.345

# For every 1-inch increase in mother's height, how many inches does the typical son's height increase? 0.381

# Question 10b

# Which sets of parent-child heights are significantly correlated at a p-value cut off of .05? Select all
# that apply.

# father-daughter [X]
# father-son [X]
# mother-daughter [X]
# mother-son [X]

galton %>%
  group_by(pair) %>%
  do(tidy(lm(childHeight ~ parentHeight, data = .), conf.int = TRUE)) %>%
  filter(term == "parentHeight", p.value < 0.5)

# When considering the estimates, which of the following statements are true? Select all that apply.

# All of the confidence intervals overlap with each other. [X]

# At least one confidence interval covers zero.

# The confidence intervals involving mothers' heights are larger than the confidence intervals involving
# fathers' heights. [X]

# The confidence intervals involving daughters' heights are larger than the confidence intervals involving
# sons' heights.

# The data are consistent with inheritance of height being independent of the child's gender. [X]

# The data are consistent with inheritance of height being independent of the parent's gender. [X]

galton %>%
  group_by(pair) %>%
  do(tidy(lm(childHeight ~ parentHeight, data = .), conf.int = TRUE)) %>%
  filter(term == "parentHeight") %>%
  mutate(interval = conf.high - conf.low) %>% select(pair, conf.low, conf.high, interval)

# 2.4. REGRESSION AND BASEBALL

# BUILDING A BETTER OFFENSIVE METRIC FOR BASEBALL

# linear regression with two variables
fit <- Teams %>%
  filter(yearID %in% 1961:2001) %>%
  mutate(BB = BB/G, HR = HR/G, R = R/G) %>%
  lm(R ~ BB + HR, data = .)
tidy(fit, conf.int = TRUE)

# regression with BB, singles, doubles, triples, HR
fit <- Teams %>%
  filter(yearID %in% 1961:2001) %>%
  mutate(BB = BB / G,
         singles = (H - X2B - X3B - HR) / G,
         doubles = X2B / G,
         triples = X3B / G,
         HR = HR / G,
         R = R / G) %>%
  lm(R ~ BB + singles + doubles + triples + HR, data = .)
coefs <- tidy(fit, conf.int = TRUE)
coefs

# predict number of runs for each team in 2002 and plot
Teams %>%
  filter(yearID %in% 2002) %>%
  mutate(BB = BB / G,
         singles = (H - X2B - X3B - HR) / G,
         doubles = X2B / G,
         triples = X3B / G,
         HR = HR / G,
         R = R / G) %>%
  mutate(R_hat = predict(fit, newdata = .)) %>%
  ggplot(aes(R_hat, R, label = teamID)) +
  geom_point() +
  geom_text(nudge_x = 0.1, cex = 2) +
  geom_abline()

# average number of team plate appearances per game
pa_per_game <- Batting %>% filter(yearID == 2002) %>%
  group_by(teamID) %>% 
  summarize(pa_per_game = sum(AB+BB)/max(G)) %>%
  pull(pa_per_game) %>%
  mean

# compute per-plate-appearance rates for players available in 2002 using previous data
players <- Batting %>% filter(yearID %in% 1999:2001) %>%
  group_by(playerID) %>%
  mutate(PA = BB + AB) %>%
  summarize(G = sum(PA) / pa_per_game,
            BB = sum(BB) / G,
            singles = sum(H - X2B - X3B - HR) / G,
            doubles = sum(X2B) / G,
            triples = sum(X3B) / G,
            HR = sum(HR) / G,
            AVG = sum(H) / G,
            PA = sum(PA)) %>%
  filter(PA >= 300) %>%
  select(-G) %>%
  mutate(R_hat = predict(fit, newdata = .))

# plot player-specific predicted runs
qplot(R_hat, data = players, geom = "histogram", binwidth = 0.5, color = I("black"))

# add 2002 salary for each player
players <- Salaries %>%
  filter(yearID == 2002) %>%
  select(playerID, salary) %>%
  right_join(players, by = "playerID")

# add defensive position
position_names <- c("G_p", "G_c", "G_1b", "G_2b", "G_3b", "G_ss", "G_lf", "G_cf", "G_rf")
tmp_tab <- Appearances %>%
  filter(yearID == 2002) %>%
  group_by(playerID) %>%
  summarize_at(position_names, sum) %>%
  ungroup()
pos <- tmp_tab %>%
  select(position_names) %>%
  apply(., 1, which.max)
players <- data.frame(playerID = tmp_tab$playerID, POS = position_names[pos]) %>%
  mutate(POS = str_to_upper(str_remove(POS, "G_"))) %>%
  filter(POS != "P") %>%
  right_join(players, by = "playerID") %>%
  filter(!is.na(POS) & !is.na(salary))

# add players first and last names
players <- Master %>%
  select(playerID, nameFirst, nameLast, debut) %>%
  mutate(debut = as.Date(debut)) %>%
  right_join(players, by = "playerID")

# top 10 players
players %>% select(nameFirst, nameLast, POS, salary, R_hat) %>%
  arrange(desc(R_hat)) %>%
  top_n(10)

# Remake plot without players that debuted after 1998
library(lubridate)
players %>% filter(year(debut) < 1998) %>%
  ggplot(aes(salary, R_hat, color = POS)) +
  geom_point() +
  scale_x_log10()

# BUILDING A BETTER OFFENSIVE METRIC FOR BASEBALL: LINEAR PROGRAMMING

# A way to actually pick the players for the team can be done using what computer scientists call
# linear programming. Although we don't go into this topic in detail in this course, we include the
# code anyway.
library(reshape2)
library(lpSolve)
players <- players %>% filter(debut <= "1997-01-01" & debut > "1988-01-01")
constraint_matrix <- acast(players, POS ~ playerID, fun.aggregate = length)
npos <- nrow(constraint_matrix)
constraint_matrix <- rbind(constraint_matrix, salary = players$salary)
constraint_dir <- c(rep("==", npos), "<=")
constraint_limit <- c(rep(1, npos), 50*10^6)
lp_solution <- lp("max", players$R_hat,
                  constraint_matrix, constraint_dir, constraint_limit,
                  all.int = TRUE) 

# This algorithm chooses these 9 players:
our_team <- players %>%
  filter(lp_solution$solution == 1) %>%
  arrange(desc(R_hat))
our_team %>% select(nameFirst, nameLast, POS, salary, R_hat)

# ON BASE PLUS SLUGGING

# The on-base-percentage plus slugging percentage (OPS) metric is:

# (BB/PA) + ((Singles+Doubles+Triples+HR) / AB)

# REGRESSION FALLACY

# Regression can bring about errors in reasoning, especially when interpreting individual observations.
# The example showed in the video demonstrates that the "sophomore slump" observed in the data is caused
# by regressing to the mean.

# The code to create a table with playerID, their names, and their most played position:
library(Lahman)
playerInfo <- Fielding %>%
  group_by(playerID) %>%
  arrange(desc(G)) %>%
  slice(1) %>%
  ungroup() %>%
  left_join(Master, by = "playerID") %>%
  select(playerID, nameFirst, nameLast, POS)

# The code to create a table with only the ROY award winners and add their batting statistics:
ROY <- AwardsPlayers %>%
  filter(awardID == "Rookie of the Year") %>%
  left_join(playerInfo, by = "playerID") %>%
  rename(rookie_year = yearID) %>%
  right_join(Batting, by = "playerID") %>%
  mutate(AVG = H / AB) %>%
  filter(POS != "P")

# The code to keep only the rookie and sophomore seasons and remove players who did not play
# sophomore seasons:
ROY <- ROY %>%
  filter(yearID == rookie_year | yearID == rookie_year+1) %>%
  group_by(playerID) %>%
  mutate(rookie = ifelse(yearID == min(yearID), "rookie", "sophomore")) %>%
  filter(n() == 2) %>%
  ungroup() %>%
  select(playerID, rookie_year, rookie, nameFirst, nameLast, AVG)

# The code to use the spread() function to have one column for the rookie and sophomore years batting
# averages:
ROY <- ROY %>% spread(rookie, AVG) %>% arrange(desc(rookie))
ROY

# The code to calculate the proportion of players who have a lower batting average their sophomore year:
mean(ROY$sophomore - ROY$rookie <= 0)

# The code to do the similar analysis on all players that played the 2013 and 2014 seasons and batted
# more than 130 times:
two_years <- Batting %>%
  filter(yearID %in% 2013:2014) %>%
  group_by(playerID, yearID) %>%
  filter(sum(AB) >= 130) %>%
  summarize(AVG = sum(H) / sum(AB)) %>%
  ungroup() %>%
  spread(yearID, AVG) %>%
  filter(!is.na(`2013`) & !is.na(`2014`)) %>%
  left_join(playerInfo, by = "playerID") %>%
  filter(POS != "P") %>%
  select(-POS) %>%
  arrange(desc(`2013`)) %>%
  select(nameFirst, nameLast, `2013`, `2014`)
two_years

# The code to see what happens to the worst performers of 2013:
arrange(two_years, `2013`)

# The code to see the correlation for performance in two separate years:
qplot(`2013`, `2014`, data = two_years)

summarize(two_years, cor(`2013`, `2014`))

# MEASUREMENT ERROR MODELS

# Up to now, all our linear regression examples have been applied to two or more random variables. We
# assume the pairs are bivariate normal and use this to motivate a linear model.

# Another use of linear regression is with measurement error models, where it is common to have a non-random
# covariate (such as time). Randomness is introduced  from measurement error rather than sampling or
# natural variability.

# The code to use dslabs function rfalling_object() to generate simulations of dropping balls:
library(dslabs)
falling_object <- rfalling_object()

# The code to draw the trajectory of the ball:
falling_object %>%
  ggplot(aes(time, observed_distance)) +
  geom_point() +
  ylab("Distance in meters") +
  xlab("Time in seconds")

# The code to use the lm() function to estimate the coefficients:
fit <- falling_object %>%
  mutate(time_sq = time^2) %>%
  lm(observed_distance ~ time + time_sq, data = .)
tidy(fit)

# The code to check if the estimated parabola fits the data:
augment(fit) %>%
  ggplot() +
  geom_point(aes(time, observed_distance)) +
  geom_line(aes(time, .fitted), col = "blue")

# The code to see the summary statistic of the regression:
tidy(fit, conf.int = TRUE)

# ASSESSMENT: REGRESSION AND BASEBALL, PART 1

# Question 1

# What is the final linear model (in the video "Building a better offensive metric for baseball") we
# used to predict runs scored per game?

# lm(R ~ BB + HR)
# lm(HR ~ BB + singles + doubles + triples)
# lm(R ~ BB + singles + doubles + triples + HR) [X]
# lm(R ~ singles + doubles + triples + HR)

# Question 2

# We want to estimate runs per game scored by individual players, not just by teams. What summary
# metric do we calculate to help estimate this?

# Look at the code from the video "Building a better offensive metric for baseball" for a hint.
pa_per_game <- Batting %>%
  filter(yearID == 2002) %>%
  group_by(teamID) %>%
  summarize(pa_per_game = sum(AB+BB) / max(G)) %>%
  .$pa_per_game %>%
  mean

# The summary metric used is:

# pa_per_game: the mean number of plate appearances per team per game for each team
# pa_per_game: the mean number of plate appearances per game for each player
# pa_per_game: the number of plate appearances per team per game, averaged across all teams [X]

# Question 3

# Imagine you have two teams. Team A is comprised of batters who, on average, get two bases on balls, four
# singles, one double, no triples, and one home run. Team B is comprised of batters who, on average, get
# one base on balls, six singles, two doubles, one triple, and no home runs.

# Which team scores more runs, as predicted by our model?

teamA <- c(1, 2, 4, 1, 0, 1)
teamB <- c(1, 1, 6, 2, 1, 0)
sum(teamA * coefs$estimate) # 2.26 runs
sum(teamB * coefs$estimate) # 3.5 runs, team B wins

# Question 4

# The On-base-percentage plus slugging percentage (OPS) metric gives the most weight to:

# singles
# doubles
# triples
# home runs [X]

# Question 5

# What statistical concept properly explains the "sophomore slump"?

# Regression to the mean [X]
# Law of averages
# Normal distribution

# Question 6

# In our model of time vs. observed_distance in the video "Measurement error models", the randomness
# of our data was due to:

# sampling
# natural variability
# measurement error [X]

# Question 7

# Which of the following are important assumptions about the measurement errors in the experiment
# presented in the video "Measurement error models"? Select all that apply.

# The measurement error is random. [X]
# The measurement error is independent. [X]
# The measurement error has the same distribution for each time i.[X]

# Question 8

# Which of the following scenarios would violate an assumption of our measurement error model?

# The experiment was conducted on the moon.
# There was one position were it was particularly difficult to see the dropped ball. [X]
# The experiment was only repeated 10 times, not 100 times.

# Question 9 has two parts. Use the information below to answer both parts.

# Use the teams data frame from the Lahman package. Fit a multivariate linear regression model to obtain
# the effects of BB and HR on Runs (R) in 1971. Use the tidy() function in the broom package to obtain
# the results in a data frame.

fitQ9 <- Teams %>%
  filter(yearID == 1971) %>%
  mutate(BB = BB / G,
         HR = HR / G,
         R = R / G) %>%
  lm(R ~ BB + HR, data = .)
tidy(fitQ9, conf.int = TRUE)

# Question 9a

# What is the estimate for the effect of BB on runs? 0.412

# What is the estimate for the effect of HR on runs? 1.29

# Question 9b

# Interpret the p-values for the estimates using a cutoff of 0.05.

# Which of the following is the correct interpretation?

# Both BB and HR have a nonzero effect on runs.
# HR has a significant effect on runs, but the evidence is not strong enough to suggest BB also does. [X]
# BB has a significant effect on runs, but the evidence is not strong enough to suggest HR also does.
# Neither BB nor HR have a statistically significant effect on runs.

# Question 10

# Repeat the above exercise to find the effect of BB and HR on runs (R) for every year from 1961 to 2018
# using do() and the broom package.

fitQ10 <- Teams %>% filter(yearID %in% 1961:2018) %>%
  group_by(yearID) %>%
  do(tidy(lm(R ~ BB + HR, data = .), conf.int = TRUE))

# Make a scatterplot of the estimate for the effect of BB on runs over time and add a trend line with 
# confidence intervals.

fitQ10 %>% filter(term == "BB") %>%
  ggplot(aes(yearID, estimate)) +
  geom_point() +
  geom_smooth(method = "lm")

# The effect of BB on runs has ??? over time.

# decreased
# increased [X]
# remained the same

# Question 11

# Fit a linear model on the results from Question 10 to determine the effect of year on the impact of
# BB. 

fitQ11 <- fitQ10 %>% filter(term == "BB") %>%
  lm(estimate ~ yearID, data = .)
tidy(fitQ11)

# For each additional year, by what value does the impact of BB on runs change? 0.0035

# What is the p-value for this effect? 0.00807

## ASSESSMENT: LINEAR MODELS

# This assessment has 6 multi-part questions that will all use the setup below.

# Game attendance in baseball varies partly as a function of how well a team is playing.

# Load the Lahman library. The Teams data frame contains an attendance column. This is the total attendance
# for the season. To calculate the average attendance, divide by the number of games played, as follows:
library(tidyverse)
library(broom)
library(Lahman)
Teams_small <- Teams %>%
  filter(yearID %in% 1961:2001) %>%
  mutate(avg_attendance = attendance/G)

# Use linear models to answer the following 3-part question about Teams_small.

# Question 1a

# Use runs (R) per game to predict average attendance.

fit <- Teams_small %>%
  mutate(Rpg = R / G) %>%
  lm(avg_attendance ~ Rpg, data = .)

# For every 1 run scored per game, average attendance increases by how much? # 4,117

tidy(fit)

# Use home runs (HR) per game to predict average attendance.

fit <- Teams_small %>%
  mutate(HRpg = HR / G) %>%
  lm(avg_attendance ~ HRpg, data = .)

# For every 1 home run hit per game, average attendance increases by how much? # 8,113

tidy(fit)

# Question 1b

# Use number of wins to predict average attendance; do not normally for number of games.

# For every game won in a season, how much does average attendance increase?

Teams_small %>%
  lm(avg_attendance ~ W, data = .) %>%
  .$coef # 121

# Suppose a team won zero games in a season.

# Predict the average attendance. # Intercept = 1,129

# Question 1c

# Use year to predict average attendance.

# How much does average attendance increase each year?

Teams_small %>%
  lm(avg_attendance ~ yearID, data = .) %>%
  .$coef # 244

# Question 2

# Game wins, runs per game and home runs per game are positively correlated with attendance. We saw in
# the course material that runs per game and home runs per game are correlated with each other. Are
# wins and runs per game or wins and home runs per game correlated? Use the Teams_small data once
# again.

Teams_small %>%
  mutate(rpg = R / G, hrpg = HR / G) %>%
  summarize(cor1 = cor(W, rpg), cor2 = cor(W, hrpg))

# What is the correlation coefficient for runs per game and wins? # 0.411

# What is the correlation coefficient for home runs per game and wins? # 0.274

# Stratify Teams_small by wins: divide number of wins by 10 and then round to the nearest integer. Keep
# only strata 5 through 10, which have 20 or more data points.

Teams_small <- Teams_small %>%
  mutate(wins = round(W / 10, 0)) %>%
  filter(wins %in% 5:10)

# Use the stratified dataset to answer this three-part questions.

# Question 3a

# How many observations are in the 8 win strata?

sum(Teams_small$wins == 8)

# Question 3b

# Calculate the slope of the regression line predicting average attendance given runs per game for each
# of the win strata.

Teams_small %>%
  mutate(R_per_game = R / G) %>%
  group_by(wins) %>%
  do(tidy(lm(avg_attendance ~ R_per_game, data = .))) %>%
  filter(term == "R_per_game")

# Which win stratum has the largest regression line slope? # 5

# Calculate the slope of the regression line predicting average attendance given home runs per game
# for each of the win strata.

Teams_small %>%
  mutate(HR_per_game = HR / G) %>%
  group_by(wins) %>%
  do(tidy(lm(avg_attendance ~ HR_per_game, data = .))) %>%
  filter(term == "HR_per_game")

# Which win stratum has the largest regression line slope? # 5

# Also...
Teams_small %>%  
  group_by(wins) %>%
  summarize(slope = cor(HR/G, avg_attendance)*sd(avg_attendance)/sd(HR/G))

# Question 3c

# Which of the following are true about the effect of win strata on average attendance?

# Across all win strata, runs per game are positively correlated with average attendance. [X]

Teams_small %>%
  group_by(wins) %>%
  summarize(cor = cor(R/G, avg_attendance))

# Runs per game has the strongest effect on attendance when a team wins many games.

# After controlling for number of wins, home runs per game are not correlated with attendance.

Teams_small %>%
  group_by(wins) %>%
  summarize(cor = cor(HR/G, avg_attendance))

# Home runs per game has the strongest effect on attendance when a team does not win many games. [X]

Teams_small %>%
  mutate(HR_per_game = HR / G) %>%
  group_by(wins) %>%
  do(tidy(lm(avg_attendance ~ HR_per_game, data = .))) %>%
  filter(term == "HR_per_game")

# Among teams with similar number of wins, teams with more home runs per game have larger average
# attendance. [X]

# Question 4

# Fit a multivariate regression determining the effects of runs per game, home runs per game, wins, 
# and year on average attendance. Use the original Teams_small wins column, not the win strata from
# question 3.

Teams_small %>%
  mutate(R_per_game = R / G, HR_per_game = HR / G) %>%
  lm(avg_attendance ~ R_per_game + HR_per_game + W + yearID, data = .) %>%
  .$coef

# What is the estimate of the effect of runs per game on average attendance? 322

# What is the estimate of the effect of home runs per game on average attendance? 1,798

# WHat is the estimate of the effect of wins in a season on average attendance? 117

# Question 5

# Use the multivariate regression model from question 4. Suppose a team averaged 5 runs per game, 1.2
# home runs per game, and won 80 games in a season.

fit <- Teams_small %>%
  mutate(R_per_game = R / G, HR_per_game = HR / G) %>%
  lm(avg_attendance ~ R_per_game + HR_per_game + W + yearID, data = .)

# What would this team's average attendance be in 2002?

Y_hat <- fit$coefficients[1] + 5*fit$coefficients[2] + 1.2*fit$coefficients[3] + 80*fit$coefficients[4] +
  2002*fit$coefficients[5]
Y_hat # 16149

# What would this team's average attendance be in 1960? 

Y_hat <- fit$coefficients[1] + 5*fit$coefficients[2] + 1.2*fit$coefficients[3] + 80*fit$coefficients[4] +
  1960*fit$coefficients[5]
Y_hat # 6505

# Also with predict()
predict(fit, data.frame(R_per_game = 5, HR_per_game = 1.2, W = 80, yearID = 1960))

# Question 6

# Use your model from Question 4 to predict average attendance for teams in 2002 in the original Teams
# data frame.

Teams %>%
  filter(yearID == 2002) %>%
  mutate(avg_attendance = attendance / G,
         R_per_game = R /G,
         HR_per_game = HR / G,
         Y_hat = predict(lm(avg_attendance ~ R_per_game + HR_per_game + W + yearID, data = .))) %>%
  summarize(cor = cor(avg_attendance, Y_hat)) %>% .$cor

# What is the correlation between the predicted attendance and actual attendance? # 0.524

## SECTION 3: CONFOUNDING

# In the Confounding section, you will learn what is perhaps the most important lesson of statistics:
# that correlation is not causation.

# Identify examples of spurious correlation and explain how data dredging can lead to spurious correlation.
# Explain how outliers can drive correlation and learn to adjust for outliers using Spearman correlations.
# Explain how reversing cause and effect can lead to associations being confused with causation.
# Understand how confounders can lead to the misinterpretation of associations.
# Explain and give examples of Simpson's paradox.

# CORRELATION IS NOT CAUSATION: SPURIOUS CORRELATION

# Association/correlation is not causation
# p-hacking is a topic of much discussion because it is a problem in scientific publications. Because
# publishers tend to reward statistically significant results over negative results, there is an incentive
# to report significant results.

# generate Monte Carlo simulation
library(tidyverse)
N <- 25
g <- 1000000
sim_data <- tibble(group = rep(1:g, each = N), x = rnorm(N * g), y = rnorm(N * g))

# calculate correlation between X, Y for each group
res <- sim_data %>%
  group_by(group) %>%
  summarize(r = cor(x, y)) %>%
  arrange(desc(r))
head(res)

# plot points from the group with maximum correlation
sim_data %>% filter(group == res$group[which.max(res$r)]) %>%
  ggplot(aes(x, y)) +
  geom_point() +
  geom_smooth(method = "lm")

# histogram of correlations in Monte Carlo simulations
res %>% ggplot(aes(x = r)) + geom_histogram(binwidth = 0.1, color = "black")

# linear regression on group with maximum correlation
library(broom)
sim_data %>%
  filter(group == res$group[which.max(res$r)]) %>%
  do(tidy(lm(y ~ x, data = .)))

# CORRELATION IS NOT CAUSATION: OUTLIERS

# Correlations can be caused by outliers.
# The Spearman correlation is calculated based on the ranks of data.


# simulate independent X, Y and standardize all except entry 23
set.seed(1985, sample.kind = "Rounding")
x <- rnorm(100, 100, 1)
y <- rnorm(100, 84, 1)
x[-23] <- scale(x[-23])
y[-23] <- scale(y[-23])

# plot shows the outlier
qplot(x, y, alpha = 0.5)

# outlier makes it appear there is a correlation
cor(x, y)
cor(x[-23], y[-23])

# use rank instead
qplot(rank(x), rank(y))
cor(rank(x), rank(y))

# Spearman correlation with cor function
cor(x, y, method = "spearman")

# CORRELATIO IS NOT CAUSATION: REVERSING CAUSE AND EFFECT

# Another way association can be confused with causation is when the cause and effect are reversed.
# As discussed in the video, in the Galton data, when father and son were reversed in the regression,
# the model was technically correct. The estimates and p-values were obtained correctly as well. What
# was incorrect was the interpretation of the model.

# cause and effect reversal using son heights to predict father heights
library(HistData)
data("GaltonFamilies")
GaltonFamilies %>%
  filter(childNum == 1 & gender == "male") %>%
  select(father, childHeight) %>%
  rename(son = childHeight) %>%
  do(tidy(lm(father ~ son, data = .)))

# CORRELATION IS NOT CAUSATION: CONFOUNDERS

# If X and Y are correlated, we call Z a confounder if changes in Z causes changes in both X and Y.

# UC-Berkeley admission data
library(dslabs)
data(admissions)
admissions

# percent men and women accepted
admissions %>% group_by(gender) %>%
  summarize(percentage =
              round(sum(admitted*applicants)/sum(applicants),1))

# test whether gender and admissions are independent
admissions %>% group_by(gender) %>%
  summarize(total_admitted = round(sum(admitted / 100 * applicants)),
            not_admitted = sum(applicants) - sum(total_admitted)) %>%
  select(-gender) %>%
  do(tidy(chisq.test(.)))

# percent admissions by major
admissions %>% select(major, gender, admitted) %>%
  spread(gender, admitted) %>%
  mutate(women_minus_men = women - men)

# plot total percent admitted to major versus percent women applicants
admissions %>%
  group_by(major) %>%
  summarize(major_selectivity = sum(admitted * applicants) / sum(applicants),
            percent_women_applicants = sum(applicants * (gender == "women")) /
              sum(applicants) * 100) %>%
  ggplot(aes(major_selectivity, percent_women_applicants, label = major)) +
  geom_text()

# plot number of applicants admitted and not
admissions %>%
  mutate(yes = round(admitted/100 * applicants), no = applicants - yes) %>%
  select(-applicants, -admitted) %>%
  gather(admission, number_of_students, -c("major", "gender")) %>%
  ggplot(aes(gender, number_of_students, fill = admission)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(. ~ major)

admissions %>%
  mutate(percent_admitted = admitted * applicants / sum(applicants)) %>%
  ggplot(aes(gender, y = percent_admitted, fill = major)) +
  geom_bar(stat = "identity", position = "stack")

# condition on major, then look at differences
admissions %>% ggplot(aes(major, admitted, col = gender, size = applicants)) + geom_point()

# average difference by major
admissions %>% group_by(gender) %>% summarize(avg = mean(admitted))

# SIMPSON'S PARADOX

# Simpson's Paradox happens when we see the sign of the correlation flip when comparing the entire
# dataset with specific strata.

# ASSESSMENT: CORRELATION IS NOT CAUSATION

# Question 1

# In the videos, we ran one million tests of correlation for two random variables, X and Y.

# How many of these correlations would you expect to have a significant p-value (p <= 0.05), just by
# chance?

# 5,000
# 50,000 [X]
# 100,000
# It's impossible to know

# Question 2

# Which of the following are examples of p-hacking? Select all that apply.

# Looking for associations between an outcome and several exposures and only reporting the one that
# is significant. [X]

# Trying several different models and selecting the one that yields the smallest p-value. [X]

# Repeating an experiment multiple times and only reporting the one with the smallest p-value. [X]

# Using a Monte Carlo simulation in an analysis.

# Question 3

# The Spearman correlation coefficient is robust to outliers because:

# It drops outliers before calculating correlation.
# It is the correlation of standardized values.
# It calculates correlation between ranks, not values. [X]

# Question 4

# What can you determine if you are misinterpreting results because of a confounder?

# Nothing. If the p-value says the result is significant, then it is.
# More closely examine the results by stratifying and plotting the data. [X]
# Always assume that you are misinterpreting the results.
# Use linear models to tease out a confounder.

# Question 5

# Look again at the admissions data presented in the confounders vide using ?admissions.

# What important characteristic of the table variables do you need to know to understand the calculations
# used in this video?

# The data are from 1973.
# The columns major and gender are of class character, while admitted and applicants are numeric.
# The data are from the dslabs package.
# The column admitted is the percent of students admitted, while the column applicants is the total
# number of applicants. [X]

# Question 6

# In the example in the confounders video, major selectivity confounds the relationship between UC Berkeley
# admission rates and gender because: 

# It was harder for women to be admitted to UC Berkeley.

# Major selectivity is associated with both admission rates and with gender, as women tended to apply to
# more selective majors. [X]

# Some majors are more selective than others.

# Major selectivity is not a confounder.

# Question 7

# Admission rates at UC Berkeley are an example of Simpson's Paradox because:

# It appears that men have a higher admission rate than women, however, after we stratify by major,
# we see that women have a higher admission rate than men. [X]

# It was a paradox that women were admitted at a lower rate than men.

# The relationship between admissions and and gender is confounded by major selectivity.

# ASSESSMENT: CONFOUNDING

# For this set of exercises, we examine the data from a 2014 PNAS paper that analyzed success rates from
# funding agencies in the Netherlands and concluded:

# "our results reveal gender bias favoring male applicants over female applicants in the prioritization of
# their 'quality of researcher' (but not 'quality of proposal') evaluation and success rates, as well as
# in the language used in instructional and evaluation materials."

# A response was published a few months later titled "No evidence that gender contributes to personal
# research funding success in The Netherlands: A reaction to Van der Lee and Ellemers," which included:

# "However, the overall gender effect borders on statistical significance, despite the large sample.
# Moreover, their conclusion could be a prime example of Simpson's paradox; if a higher percentage of
# women apply for grants in more competitive scientific disciplines (i.e., with low application success
# rates for both men and women), then an analysis across all disciplines could incorrectly show 'evidence'
# of gender inequality."

# Who is right here: the original paper or the response? Here, you will examine the data and come to your
# own conclusion.

# The main evidence for the conclusion of the original paper comes down to a comparison of percentages. The
# information we need was originally in Table S1 in the paper, which we included in dslabs:
library(dslabs)
data("research_funding_rates")
research_funding_rates

# Question 1

# Construct a two-by-two table of gender (men/women) by award status (awarded/not) using the total numbers
# across all disciplines.

research_funding_rates %>%
  mutate(not_men = applications_men - awards_men,
         not_women = applications_women - awards_women) %>%
  summarize(awarded_men = sum(awards_men),
            awarded_women = sum(awards_women),
            not_men = sum(not_men),
            not_women = sum(not_women))

# What is the number of men not awarded? # 1,345

# What is the number of women not awarded? # 1,011

# Also...

two_by_two <- research_funding_rates %>% 
  select(-discipline) %>% 
  summarize_all(funs(sum)) %>%
  summarize(yes_men = awards_men, 
            no_men = applications_men - awards_men, 
            yes_women = awards_women, 
            no_women = applications_women - awards_women) %>%
  gather %>%
  separate(key, c("awarded", "gender")) %>%
  spread(gender, value)
two_by_two

# Question 2

# Use the two-by-two table from Question 1 to compute percentages of men awarded versus women awarded.

two_by_two %>%
  summarize(men = round(sum(men * (awarded == "yes")) * 100 / sum(men), 1),
            women = round(sum(women * (awarded == "yes")) * 100 / sum(women), 1))

# What is the percentage of men awarded? 17.7 %

# What is the percentage of women awarded? 14.9 %

# Question 3

# Run a chi-square test on the two-by-two table to determine whether the difference in the two success
# rates is significant. You can use tidy() to turn the output of chisq.test() into a data frame as well.

# What is the p-value of the difference in funding rate?

two_by_two %>% select(-awarded) %>% do(tidy(chisq.test(.)))

# Question 4

# There may be an association between gender and funding. But can we infer causation here? Is gender bias
# causing this observed difference? The response to the original paper claims that what we see here is
# similar to the UC Berkeley admissions example. Specifically they state that this "could be a prime
# example of Simspon's paradox; if a higher percentage of women apply for grants in more competitive
# scientific disciplines, then an analysis across all disciplines could incorrectly show 'evidence' of 
# gender inequality."

# To settle this dispute, use this dataset with number of applications, awards, and success rate for
# each gender:
dat <- research_funding_rates %>%
  mutate(discipline = reorder(discipline, success_rates_total)) %>%
  rename(success_total = success_rates_total,
         success_men = success_rates_men,
         success_women = success_rates_women) %>%
  gather(key, value, -discipline) %>%
  separate(key, c("type", "gender")) %>%
  spread(type, value) %>%
  filter(gender != "total")
dat

# To check if this is a case of Simpson's paradox, plot the success rates versus disciplines, which have
# been ordered by overall success, with colors to denote the genders and size to denote the number of
# applications.

dat %>% ggplot(aes(discipline, success, color = gender, size = applications)) + geom_point()

# In which fields to men have a higher success rate than women?

# Chemical sciences [X]
# Earth/life sciences [X]
# Humanities
# Interdisciplinary
# Medical sciences [X]
# Physical sciences
# Physics [X]
# Social sciences [X]
# Technical sciences

# Which two fields have the most applications from women?

# Chemical sciences
# Earth/life sciences 
# Humanities 
# Interdisciplinary
# Medical sciences [X]
# Physical sciences
# Physics 
# Social sciences [X]
# Technical sciences

# Which two fields have the lowest overall funding rates?

# Chemical sciences
# Earth/life sciences 
# Humanities
# Interdisciplinary
# Medical sciences [X]
# Physical sciences
# Physics 
# Social sciences [X]
# Technical sciences