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
set.seed(1983)
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

