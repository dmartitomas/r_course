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

