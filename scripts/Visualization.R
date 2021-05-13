### DATA SCIENCE: VISUALIZATION

## SECTION 1. INTRODUCTION TO DATA VISUALIZATION AND DISTRIBUTIONS

# 1.1. INTRODUCTION TO DATA VISUALIZATION

# DataCamp Assessment 1.1. Data Types

# Exercise 1. Variable names

# The type of data we are working with will often influence the data visualization technique
# we use. We will be working with two types of variables: categorial and numeric. Each
# can be divided into two other groups: categorical can be ordinal or not, whereas numerical
# variables can be discrete or continuous.

# We will review data types using some of the examples provided in the dslabs package. For
# example, the heights dataset.

library(dslabs)
data(heights)

# Let's start by reviewing how to extract the variable names from a dataset function. What
# are the two variable names used in the heights dataset?

names(heights)

# Exercise 2. Variable type

# We saw that sex is the first variable. We know what values are represented by this
# variable and can confirm this by looking at the first few entries:

head(heights)

# What data type is the sex variable? Categorical

# Exercise 3. Numerical values

# Keep in mind that discrete numeric data can be considered ordinal. Although this is
# technically true, we usually reserve the term ordinal data for variables belonging
# to a small number of different groups, with each group having many members.

# The height variable could be ordinal if, for example, we report a small number of
# values as short, medium, and tall. Let's explore how many unique values are used
# by the heights variable. For this we can use the unique function:

x <- c(3, 3, 3, 3, 4, 4, 2)
unique(x)

# Use the unique and length functions to determine how many unique heights were reported.

x <- heights$height
length(unique(x))

# Exercise 4. Tables

# One of the useful outputs of data visualization is that we can learn about the
# distribution of variables. For categorical data we can construct this distribution
# by simply computing the frequency of each unique value. This can be done with 
# the function table. Her is an example:

x <- c(3, 3, 3, 3, 4, 4, 2)
table(x)

# Use the table function to compute the frequencies of each unique height value.
# Because we are using the resulting frequency table in a later exercise we want
# you to save the results into an object called tab.

x <- heights$height
tab <- table(x)

# Exercise 5. Indicator variables

# To see why treating the reported heights as an ordinal value is not useful in
# practice we note how many values are reported only once.

# In the previous exercise we computed the variable tab which reports the number of
# times each unique value appears. For values reported only once tab will be 1. Use
# logicals and the function sum to count the number of time this happens.

sum(tab == 1)

# Exercise 6. Data types - heights

# Since we are a finite number of reported heights and technically the height can
# be considered ordinal, which of the following is true?

# 1. It is more effective to consider heights to be numerical given the number of
# unique values we observe and the fact that if we keep collecting data even more
# will be observed.

# 2. It is actually preferable to consider heights ordinal since on a computer
# there are only a finite number of possibilities.

# 3. This is actually a categorical variable: tall, medium or short.

# 4. This is a numerical variable because numbers are used to represent it.

# The correct answer is [1]

# 1.2. INTRODUCTION TO DISTRIBUTIONS

# Cumulative Distribution Function

# DataCamp Assessment. Distributions

# Exercise 1. Distributions - 1

# You may have noticed that numerical data is often summarized with the average
# value. For example, the quality of a high school is sometimes summarized with
# one number: the average score on a standardized test. Occasionally, a second
# number is reported: the standard deviation. So, for example, you might read
# a report stating that scores were 680 plus or minus 50 (the standard
# deviation). The report has summarized an entire vector of scores with just
# two numbers. Is this appropriate? Is there any important piece of information
# that we are missing by only looking at this summary rather than the entire
# list? We are going to learn when these 2 numbers are enough and when we need
# more elaborate summaries and plots to describe the data.

# Our first data visualization building block is learning to summarize lists
# of factors or numeric vectors. The most basic statistical summary of a list
# of objects or numbers is its distribution. Once a vector has been summarized
# as a distribution, there are several data visualization techniques to
# effectively relay this information. In later assessments we will practice
# to write code for data visualization. Here we start with some multiple choice
# questions to test your understanding of distributions and related basic plots.

# In the murders dataset, the region is a categorical variable and on the right
# you can see its distribution. To the closest 5%, what proportion of the states
# are in the North Central region? 25%

# Exercise 2. Distributions - 2

# In the murders dataset, the region is a categorical variable and to the right
# is its distribution. Which of the following is true?

# 1. The graph is a histogram

# 2. The graph shows only four numbers with a bar plot.

# 3. Categories are not numbers so it does not make sense to graph the distribution.

# 4. The colors, not the height of the bars, describe the distribution.

# The correct answer is [2].

# Exercise 3. Empirical Cumulative Distribution Function (eCDF)

# The plot shows the eCDF for male heights.

# Based on the plot, what percentage of males are shorter than 75 inches? 95%

# Exercise 4. eCDF Male Heights

# The plot shows the eCDF for male heights.

# To the closest inch, what height m has the property that 1/2 of the male
# students are taller than m and 1/2 are shorter? 69"

# Exercise 5. eCDF of Murder Rates

# Here is an eCDF of the murder rates across states.

# Knowing that there are 51 states (counting DC) and based on this plot, how 
# many states have murder rates larger than 10 per 100,000 people? 1

# Exercise 6. eCDF of Murder Rates (cont'd)

# Here is an eCDF of the murder rates across states.

# Based on the eCDF above, which of the following statements are true?

# 1. About half the states have murder rates above 7 per 100,000 and the
# other half below.

# 2. Most states have murder rates below 2 per 100,000.

# 3. All states have murder rates above 2 per 100,000.

# 4. With the exception of 4 states, the murder rates are below 5 per 100,000

# The correct answer is [4]

# Exercise 7. Histograms

# Here is a histogram of male heights in our heights dataset.

# Based on this plot, how many males are between 62.5 and 65.5? 58

# Exercise 8. Histograms (cont'd)

# Here is a histogram of male heights in our heights dataset.

# About what percentage are shorter than 60 inches? 1%

# Exercise 9. Density Plots

# Based on this density plot, about what proportion of US states have populations
# larger than 10 million? 0.15

# Exercise 10. Density Plots (cont'd)

# Here are three density plots. Is it possible that they are from the same dataset?
# Which of the following startements is true?

# 1. It is impossible that they are from the same dataset.

# 2. They are from the same dataset, but different due to code errors.

# 3. They are the same dataset, but the first and second undersmooth and the 
# third oversmooths

# 4. They are the same dataset, but the first does not have the x-axis in the log
# scale, the second undersmooths, and the third oversmooths.

# The correct answer is [4]

# Normal Distribution

# Standard Units and Z-scores

# The normal CDF and pnorm()

# DataCamp Assessment. Normal Distribution

# Exercise 1. Proportions

# Histograms and density plots provide excellent summaries of a distribution. But
# can we summarize even further? We often see the average and standard deviation
# used as summary statistics: a two number summary! To understand what these
# summaries are and why they are so widely used, we need to understand the normal
# distribution.

# The normal distribution, also know as the bell curve and as the Gaussian distribution,
# is one of the most famous mathematical concepts in history. A reason for this is that
# approximately normal distributions occur in many situations. Examples include
# gambling winnings, heights, weights, blood pressure, standardized test scores, 
# and experimental measurement errors. Often data visualization is needed to confirm
# that our data follows a normal distribution.

# Here we focus on how the normal distribution helps us summarize data and can be 
# useful in practice.

# One way the normal distribution is useful is that it can be used to approximate
# the distribution of a list of numbers without having access to the entire list.
# We will demonstrate this with the heights dataset.

# Load the heights dataset and create a vector x with just the male heights:

library(dslabs)
data(heights)
x <- heights$height[heights$sex == "Male"]

# What proportion of the data is between 69 and 72 inches (taller than 69 but shorter
# or equal to 72)? A proportion is between 0 and 1.

# Use the mean function in your code. Remember that you can use mean to compute the
# proportion of entries in a logical vector that are true.

mean(x > 69 & x <= 72)

# Exercise 2. Averages and Standard Deviations

# Suppose all you know about the height data from the previous exercise is the average
# and the standard deviation and that its distribution is approximated by the normal
# distribution. We can compute the average and standard deviation like this:

avg <- mean(x)
stdev <- sd(x)

# Suppose you only have avg and stdev below, but no access to x. Can you approximate
# the proportion of the data that is between 69 and 72 inches?

# Given a normal distribution with a mean `mu` and standard deviation `sigma`, you
# can calculate the proportion of observations less than or equal to a certain
# value with pnorm(value, mu, sigma). Notice that this is the CDF for the normal
# distribution. We will learn much more about pnorm later in the course series, 
# but you can also learn more now with ?pnorm.

# Use the normal approximation to estimate the proportion of the data that is between
# 69 and 72 inches.

# Note that you can't use x in your code, only avg and stdev. Also note that R has
# a function that may prove very helpful here - check out the pnorm function (and
# remember that you can get help by using ?pnorm)

pnorm(72, avg, stdev) - pnorm(69, avg, stdev)

# Exercise 3. Approximations

# Notice that the approximation calculated in the second question is very close to
# the exact calculation in the first question. The normal distribution was a useful
# approximation for this case.

# However, the approximation is not always useful. An example is for the more extreme
# values, often called the "tails" of the distribution. Let's look at an example.
# We can compute the proportion of heights between 79 and 81.

mean(x > 79 & x <= 81)

# Use normal approximation to estimate the proportion of heights between 79 and 81
# inches and save it in an object called approx.

# Report how many times bigger the actual proportion is compared to the approximation

exact <- mean(x > 79 & x <= 81)
approx <- pnorm(81, mean(x), sd(x)) - pnorm(79, mean(x), sd(x))
exact / approx

# Exercise 4. Seven footers and the NBA

# Someone asks you what percent of seven footers are in the National Basketball
# Association (NBA). Can you provide an estimate? Let's try using the normal
# approximation to answer this question.

# Given a normal distribution with a mean `mu` and standard deviation `sigma`, you can
# calculate the proportion of observations less than or equal to a certain value with
# pnorm(value, mu, sigma). Notice that this is the CDF for the normal distribution. We
# will learn much more about pnorm later in the course series, but you can also learn
# more about it now with ?pnorm.

# First, we will estimate the proportion of adult men that are taller than 7 feet.

# Assume that the distribution of adult men in the world is normally distributed
# with an average of 69 inches and a standard deviation of 3 inches.

# Using the normal approximation, estimate the proportion of adult men that are 
# taller than 7 feet, referred to as seven footers. Remember that 1 foot equals 12
# inches.

# Use the pnorm function. Note that pnorm finds the proportion les than or equal to
# a given value, but you are asked to find the proportion greater than that value.

# Print out your estimate; don't store it in an object.

# Use pnorm to calculate the proportion over 7 feet (7*12 inches):

1 - pnorm(7 * 12, 69, 3)

# Exercise 5. Estimating the number of seven footers

# Now we have an approximation for the proportion, call it p, of men that are 7
# feet tall or taller.

# We know that there are about 1 billion men between the ages of 18 and 40 in
# the world, the age range for the NBA.

# Can we use the normal distribution to estimate how many of these 1 billion
# men are at least seven feet tall?

# Use your answer to the previous exercise to estimate the proportion of men
# that are seven feet tall or taller in the world and store that value as p.

# Then multiply this value by 1 billion (10^9) round the number of 18-40 year
# old men whi are seven feet tall or taller to the nearest integer with round.
# (Do not store this value in an object)

p <- 1 - pnorm(7 * 12, 69, 3)
round(10^9 * p)

# Exercise 6. How many seven footers are in the NBA?

# There are about 10 NBA players that are 7 feet tall or taller.

# Use your answer to exercise 4 to estimate the proportion of men that are
# seven feet tall or taller in the world and store that value as p.

# Use your answer to the previous exercise (5) to round the number of 18-40
# eyar old men who are seven feet tall or taller to the nearest integer and
# store that value as N.

# The calculate the proportion of the world's 18 to 40 year old seven
# footers that are in the NBA. (Do not store this value in an object).

p <- 1 - pnorm(7 * 12, 69, 3)
N <- round(10^9 * p)
10/N

# Exercise 7. LeBron James' height

# In the previous exercise we estimated the proportion of seven footers in
# the NBA using the above simple code.

# Repeat the calculations performed in the previous exercise for LeBron
# James' height: 6 feet 8 inches. There are about 150 players, instead of
# 10, that are at least that tall in the NBA.

# Report the estimated proportion of people at least LeBron's height that
# are in the NBA.

p <- 1 - pnorm(6*12+8, 69, 3)
N <- round(10^9*p)
150/N

# Exercise 8. Interpretation

# In answering the previous questions, we found that it is not at all rare for
# a seven footer to become an NBA player.

# What would be a fair critique of our calculations?

# 1. Practice and talent are what make great basketball players, not height.

# 2. The normal approximation is not appropriate for heights.

# 3. As seen in exercise 3, the normal approximation tends to underestimate 
# the extreme values. It's possible that there are more seven footers than
# we predicted.

# 4. As seen in exercise 3, the normal approximation tends to overestimate
# the extreme values. It's possible that there are less seven footers than
# we predicted.

# The correct answer is [3]

# 1.3. QUANTILES, PERCENTILES, AND BOXPLOTS

# Exercise 1. Vector lengths

# When analyzing data it's often important to know the number of measurements
# you have for each category.

# Define a variable `male` that contains the male heights.
# Define a variable `female` that contains the female heights.
# Report the length of each variable.

library(dslabs)
data(heights)

male <- heights$height[heights$sex == "Male"]
female <- heights$height[heights$sex == "Female"]

length(male)
length(female)

# Exercise 2. Percentiles

# Suppose we can't make a plot and want to compare the distributions side by
# side. If the number of data points is large, listing all the numbers is 
# inpractical. A more practical approach is to look at the percentiles. We can
# obtain percentiles using the quantile function like this:

quantile(heights$height, seq(.01, .99, .01))

# Create two five row vectors showing the 10th, 30th, 50th, 70th, and 90th
# percentiles for the heights of each sex. Call these vectors
# female_percentiles and male_percentiles.

# Then create a data frame called df with these two vectors as columns. The
# column names should be female and male and should appear in that order.
# As an example consider that if you want a data frame to have column names
# names and grades, in that order, you do it like this:

df <- data.frame(names = c("Jose", "Mary"), grades = c("B", "A"))

# Take a look at the df by printing it. This will provide some information on
# how male and female heights differ.

female_percentiles <- quantile(female, seq(.1, .9, .2))
male_percentiles <- quantile(male, seq(.1, .9, .2))

df <- data.frame(female = female_percentiles, male = male_percentiles)

print(df)

# Exercise 3. Interpreting Boxplots #1

# Study the boxplots summarizing the distributions of population sizes by
# country.

# Which continent has the country with the largest population size? Asia

# Exercise 4. Interpreting Boxplots #2

# Study the boxplots summarizing the distributions of population sizes by
# country.

# Which continent has the largest median population? Africa

# Exercise 5. Interpreting Boxplots #3

# Again, look at the boxplots summarizing the distributions of population
# sizes by country. To the nearest million, what is the median population
# size for Africa? 10 million

# Exercise 6. Low quantiles

# Examine the following boxplots and report approximately what proportion
# of countries in Europe have populations below 14 million: 0.75

# Exercise 7. Interquartile Range (IQR)

# Using the boxplot as guidance, which continent shown below has the
# largest interquantile range for log(population)? Americas

# 1.4. EXPLORATORY DATA ANALYSIS

# Exercise 1. Exploring the Galton Dataset - Average and Median

# For this chapter, we will use the height data collected by Francis
# Galton for his genetic studies. Here we just ue height of the
# children in the dataset.

install.packages("HistData")
library(HistData)
data(Galton)
x <- Galton$child

# Compute the average and median of these data. Note: do not assign
# them to a variable

mean(x)
median(x)

# Exercise 2. Exploring the Galton Dataset - SD and MAD

# Now for the same data compute the standard deviation and the median
# absolute deviation (MAD)

sd(x)
mad(x)

# Exercise 3. Error impact on average

# In the previous exercises we saw that the mean and median are very
# similar and so are the standard deviation and MAD. This is expected
# since the data is approximated by a normal distribution which has
# this property.

# Now suppose that Galton made a mistake when entering the first value,
# forgetting to use the decimal point. You can imitate this error by
# typing:

x_with_error <- x
x_with_error[1] <- x_with_error[1] * 10

# The data now has an outlier that the normal approximation does not
# account for. Let's see how this affects the average.

# Report how many inches the average grows after this mistake.
# Specifically, report the difference between the average of the data
# with the mistake x_with_error and the data without the mistake x.

mean(x_with_error) - mean(x)

# Exercise 4. Error impact on SD

# In the previous exercise we saw how a simple mistake in 1 out of
# over 900 observations can result in the average data increasing
# more than half an inch, which is a large difference in practical
# terms. Now let's explore the effect this outlier has on the
# standard deviation.

# Report how many inches the SD grows after this mistake. Specifically,
# report the difference between the SD of the data with the mistake
# x_with_error and the data without the mistake x.

sd(x_with_error) - sd(x)

# Exercise 5. Error impact on median

# In the previous exercises we saw how one mistake can have a 
# substantial effect on the average and the standard deviation.

# Now we are going to see how the median and MAD are much more
# resistant to outliers. For this reason we say that they are
# robust summaries.

# Report how many inches the median grows after the mistake.
# Specifically, report the difference between the median of the 
# data with the mistake x_with_error and the data without
# the mistake x.

median(x_with_error) - median(x)

# Exercise 6. Error impact on MAD

# We saw that the median barely changes. Now let's see how the
# MAD is affected.

# Report how many inches the MAD grows after the mistake. Specifically,
# report the difference between the MAD of the data with the mistake
# x_with_error and the data without the mistake x.

mad(x_with_error) - mad(x)

# Exercise 7. Usefulness of EDA

# How could you use exploratory data analysis to detect that an error
# was made?

# 1. Since it is only one value out of many, we will not be able to
# detect this.

# 2. We would see an obvious shift in the distribution.

# 3. A boxplot, histogram, or qq-plot would reveal a clear outlier.

# 4. A scatter plot would show high levels of measurement error.

# The correct answer is [3]

# Exercise 8. Using EDA to explore changes

# We have seen how the average can be affected by outliers. But how
# large can this effect get? This of course depends on the size
# of the outlier and the size of the dataset.

# To see how outliers can affect the average of a dataset, let's
# write a simple function that takes the size of the outlier as
# input and returns the average.

# Write a function called `error_avg` that takes a value `k` and
# returns the average of the vector x after the first entry
# changed to `k`. Show the results for k = 10000 and k = -10000.

error_avg <- function(k){
  x[1] <- k
  mean(x)
}

error_avg(10000)
error_avg(-10000)

## SECTION 2. INTRODUCTION TO GGPLOT2

# 2.1. BASICS OF GGPLOT2

library(tidyverse)

library(dslabs)
data(murders)

ggplot(data = murders)

murders %>% ggplot()

p <- ggplot(data = murders)
class(p)
p

# 2.2. CUSTOMIZING PLOTS

# Key Points

# In ggplot2, graphs are created by adding `layers` to the ggplot object.
# The geometry layer defines the plot type and takes the format geom_X
# Aesthetic mappings describe how properties of the data connect with features
# of the graph.
# aes() uses variables names from the object component, e.g., total rather
# than murders$total
# geom_point() creates a scatterplot and requires x and y aesthetic mappings.
# geom_text() and geom_label() add text to a scatterplot.

# Adding layers to a plot

murders %>% ggplot() +
  geom_point(aes(x = population/10^6, y = total))

# Add points layer to predefined ggplot object

p <- ggplot(data = murders)
p + geom_point(aes(x = population/10^6, y = total))

# Add text layer to scatterplot

p + geom_point(aes(population/10^6, total)) +
  geom_text(aes(population/10^6, total, label = abb))

# Example of aes behavior

# No error from this call

p_test <- p + geom_text(aes(population/10^6, total, label = abb))

# Error! `abb` is not a globally defined variable and cannot be found outside
# of aes()

p_test <- p + geom_text(aes(population/10^6, total), label = abb)

# Change the size of the points

p + geom_point(aes(population/10^6, total), size = 3) +
  geom_text(aes(population/10^6, total, label = abb))

# Move text labels slightly to the right

p + geom_point(aes(population/10^6, total), size = 3) +
  geom_text(aes(population/10^6, total, label = abb), nudge_x = 1)

# Simplify code by adding global aesthetic

p <- murders %>% ggplot(aes(population/10^6, total, label = abb))
p + geom_point(size = 3) +
  geom_text(nudge_x = 1.5)

# Local aesthetics override global aesthetics

p + geom_point(size = 3) +
  geom_text(aes(x = 10, y = 800, label = "Hello there!"))

# Log-scale the x- and y-axis

p <- murders %>% ggplot(aes(population/10^6, total, label = abb))

# log base 10 scale the x-axis and y-axis

p + geom_point(size = 3) +
  geom_text(nudge_x = 0.05) +
  scale_x_continuous(trans = "log10") +
  scale_y_continuous(trans = "log10")

# Efficient log scaling of the axes

p + geom_point(size = 3) +
  geom_text(nudge_x = 0.075) +
  scale_x_log10() +
  scale_y_log10()

# Add labels and title

p + geom_point(size = 3) +
  geom_text(nudge_x = 0.075) +
  scale_x_log10() +
  scale_y_log10() +
  xlab("Population in millions (log scale)") +
  ylab("Total number of murders (log scale)") +
  ggtitle("US Gun Murders in 2010")

# Change color of the points

# Redefine p to be everything except the points layer

p <- murders %>% ggplot(aes(population/10^6, total, label = abb)) +
  geom_text(nudge_x = 0.075) +
  scale_x_log10() +
  scale_y_log10() +
  xlab("Population in millions (log scale)") +
  ylab("Total number of murders (log scale)") +
  ggtitle ("US Gun Murders in 2010")

# Make all points blue

p + geom_point(size = 3, color = "blue")

# color points by region

p + geom_point(aes(col = region), size = 3)

# Add a line with average murder rate

# Define average murder rate

r <- murders %>%
  summarize(rate = sum(total) / sum(population) * 10^6) %>%
  pull(rate)

# Basic line with average murder rate for the country

p + geom_point(aes(col = region), size = 3) +
  geom_abline(intercept = log10(r)) # slope is default of 1

# Change line to dashed and dark grey, line under points

p + geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +
  geom_point(aes(col = region), size = 3)

# Change legend title

p <- p + scale_color_discrete(name = "Region") # Capitalize legend title

# Adding themes

install.packages("ggthemes")

# Theme used for graphs in the textbook and course

library(dslabs)
ds_theme_set()

# themes from ggthemes

library(ggthemes)

p + theme_economist() # style of the Economist magazine
p + theme_fivethirtyeight() # style of the FiveThirtyEight website

# Putting it all together to assemble the plot

# Define the intercept

r <- murders %>% 
  summarize(rate = sum(total) / sum(population) * 10^6) %>% .$rate

# Make the plot, combining all elements

install.packages("ggrepel")
library(ggrepel)

murders %>%
  ggplot(aes(population / 10^6, total, label = abb)) +
  geom_abline(intercept = log10(r), lty = 2, color = "darkgrey") +
  geom_point(aes(col = region), size = 3) +
  geom_text_repel() +
  scale_x_log10() +
  scale_y_log10() +
  xlab("Population in millions (log scale)") +
  ylab("Total number of murders (log scale)") +
  ggtitle("US Gun Murders in 2010") +
  scale_color_discrete(name = "Region") +
  theme_economist()

# Other examples

# geom_histogram() creates a histogram. Use the binwidth argument to 
# change the width of bins, the fill argument to change the bar fill color, 
# and the col argument to change the bar outline color.

# geom_density() creates smooth density plots. Change the fill color of the
# plot with the fill argument.

# geom_qq() creates a quantile-quantile plot. This geometry requires the
# sample argument. By default, the data are compared to a standard normal
# changed with the dparams argument, or the sample data can be scaled.

# Plots can be arranged adjacent to each other using the grid.arrange()
# function from the gridExtra package. First, create the plots and save
# them to objects (p1, p2, ...). Then pass the plot objects to
# grid.arrange().

# Histograms in ggplot2

# Load heights data

library(tidyverse)
library(dslabs)

# Define p
p <- heights %>%
  filter(sex == "Male") %>%
  ggplot(aes(x = height))

# Histogram with blue fill, black outline, labels, and title

p + geom_histogram(binwidth = 1, fill = "blue", col = "black") +
  xlab("Male heights in inches") +
  ggtitle("Histogram")

# Smooth density plots in ggplot2

p + geom_density()
p + geom_density(fill = "blue")

# Quantile-quantile plots in ggplot2

# Basic Q-Q plot

p <- heights %>% filter(sex == "Male") %>%
  ggplot(aes(sample = height))
p + geom_qq()

# QQ-plot against a normal distribution with the mean/sd as data

params <- heights %>%
  filter(sex == "Male") %>%
  summarize(mean = mean(height), sd = sd(height))
  p + geom_qq(dparams = params) + geom_abline()

# QQ-plot of scaled data against the standard normal distribution
  
heights %>%
  ggplot(aes(sample = scale(height))) +
  geom_qq() +
  geom_abline()

# Grids of plots with the gridExtra package

install.packages("gridExtra")

# Define plots p1, p2, p3

p <- heights %>% filter(sex == "Male") %>% ggplot(aes(x = height))
p1 <- p + geom_histogram(binwidth = 1, fill = "blue", col = "black")
p2 <- p + geom_histogram(binwidth = 2, fill = "red", col = "black")
p3 <- p + geom_histogram(binwidth = 3, fill = "green", col = "black")

# Arrange plots next to each other in 1 row, 3 columns

library(gridExtra)
library(dplyr)

grid.arrange(p1, p2, p3, ncol = 3)

# Assessment. Introduction to ggplot2

# Exercise 1. ggplot2 basics

# Start by loading the dplyr and ggplotw libraries as well as the murders data

library(dplyr)
library(ggplot2)
library(dslabs)
data(murders)

# Note that you can load both dplyr and ggplot2, as well as other packages, 
# by installing and loading the tidyverse package.

# With ggplot2 plots can be saved as objects. For example, we can associate
# a dataset with a plot object like this:

p <- ggplot(data = murders)

# Because data is the first argument we don't need to spell it out. So we can
# write this instead

p <- ggplot(murders)

# or, if we load dplyr, we can use this pipe:

p <- murders %>% ggplot()

# Remember the pipe sends the object on the left of %>% to be the first
# argument for the function to the right of %>%.

# Now let's get an introduciton to ggplot.

# what is the class of object p? 

class(p)

# "gg" "ggplot"

# Exercise 2. Printing

# Remember that to print an object you can use the command print or simply
# type the object. For example, instead of 

x <- 2
print(x)

# you can simply type

x <- 2
x

# Print the object p defined in exercise 1.

p <- ggplot(murders)

p

# and describe what you see.

# 1. Nothing happens
# 2. A blank slate plot
# 3. A scatterplot
# 4. A histogram

# The correct answer is [2]

# Exercise 3. Pipes

# Now we are going to review the use of pipes by seeing how they can 
# be used with ggplot.

# Using the pipe %>%, create an object p associated with the heights dataset
# instead of with the murders dataset as in previous exercises.

data(heights)

# Define a ggplot object called p like in the previous exercise but using
# a pipe

p <- heights %>% ggplot()

# Exercise 4. Layers

# Now we are going to add layers and the corresponding aesthetic mappings.
# For the murders data, we plotted total murders versus population sizes in
# the videos.

# Explore the murders data frame to remind yourself of the names for the two
# variables (total murders and population size) we want to plot and
# select the correct answer.

str(murders)

# total and population

# Exercise 5. geom_point #1

# To create a scatterplot, we add a layer with the function geom_point(). The
# aesthetic mappings require us to define the x-axis and y-axis variables
# respectively. So the code looks like this:

murders %>% ggplot(aes(x = `?`, y = `?`)) +
  geom_point()

# Except we have to fill in the blanks to define the two variables x and y.

# Fill out the sample code with the correct variable names to plot total
# murders versus population size.

murders %>% ggplot(aes(x = population, y = total)) +
  geom_point()

# Exercise 6. geom_point() #2

# Note that if we don't use argument names, we can obtain the same result
# by making sure we wnter the variable names in the desired order:

murders %>% ggplot(aes(population, total)) +
  geom_point()

# Remake the plot but flip the axes so that the total is on the x-axis
# and population is on the y-axis.

murders %>% ggplot(aes(total, population)) +
  geom_point()

# Exercise 7. geom_point text

# If instead of points we want to add text, we can use the geom_text() or
# geom_label() geometries. However, note that the following code

murders %>% ggplot(aes(population, total)) +
  geom_label()

# will give us an error message:

# Error: geom_label requires the following missing aesthetics: label

# Why is this?

# 1. We need to map a character to each point through the label argument
# in aes.

# 2. We need to let geom_label() know what character to use in the plot.

# 3. The geom_label() geometry does not require x-axis and y-axis values.

# 4. geom_label() is not a ggplot2 command.

# The correct answer is [1]

# Exercise 8. geom_point text

# You can also add labels to the points on a plot.

# Rewrite the code from the previous exercise to:

# Add a label aesthetic to aes equal to the state abbreviation

# use geom_label() instead of geom_point()

murders %>% ggplot(aes(population, total, label = abb)) +
  geom_label()

# Exercise 9. geom_point colors

# Now let's change the color of the labels to blue. How can we do this?

# 1. By adding a column called blue to murders.

# 2. By mapping the colors through aes because each label needs
# a different color.

# 3. By using the color argument in ggplot()

# 4. By using the color argument in geom_label because we want all colors
# to be blue so we do not need to map colors.

# The correct answer is [4]

murders %>% ggplot(aes(population, total, label = abb)) +
  geom_label(col = "blue")

# Exercise 10. geom_point colors #2

# Now let's go ahead and make the labels blue. We previously wrote this code to
# add labels to our plot:

murders %>% ggplot(aes(population, total, label = abb)) +
  geom_label()

# Now we will edit this code.

# Rewrite the code above to make the labels blue by adding an argument to
# geom_label()

# You do not need to put the color argument inside of an aes col.

# Note that the grader expects you to use the argument color instead of col;
# these are equivalent

murders %>% ggplot(aes(population, total, label = abb)) +
  geom_label(color = "blue")

# Exercise 11. geom_labels by region

# Now suppose we want to use color to represent the different regions.
# So the states from the West will be one color, states from the Northeast
# another, and so on. In this case, which of the following is most appropriate?

# 1. Adding a column called color to murders with the color we want to use.

# 2. Mapping the colors through the color argument of aes because each
# label needs a different color.

# 3. Using the color argument in ggplot

# 4. Using the color argument in geom_label because we want all colors to be
# blue so we do not need to map colors.

# The correct answer is #2

# Exercise 12. geom_label colors

# We previously used this code to make a plot using the state abbreviations
# as labels:

murders %>% ggplot(aes(population, total, label = abb)) +
  geom_label()

# We are now going to add color to represent each region.

# Rewrite the code above to make the label color correspond to the state's
# region. Because this is a mapping, you will have to do this through the aes
# function. Use the existing aes function inside of the ggplot function.

murders %>% ggplot(aes(population, total, label = abb, col = region)) +
  geom_label()

# Exercise 13. Log-scale

# Now we are going to change the axes to log scales to account for the fact
# that the population distribution is skewed. Let's start by defining an
# object p that holds the plot we have made up to now.

p <- murders %>% ggplot(aes(population, total, label = abb, color = region)) +
  geom_label()

# To change the x-axis to a log scale we learned about the scale_x_log10()
# function. We can change the axis by adding this layer to the object p to
# change the scale and render the plot using the following code.

p + scale_x_log10()

# Change both axes to be in the log scale on a single graph. Make sure you
# do not redefine p - just add the appropriate layers.

p + scale_x_log10() +
  scale_y_log10()

# Exercise 14. Titles

# In the previous exercises we create a plot using the following code:

library(dplyr)
library(ggplot2)
library(dslabs)
data(murders)
p <- murders %>% ggplot(aes(population, total, label = abb, color = region)) +
  geom_label()
p + scale_x_log10() + scale_y_log10()

# We are now going to add a title to this plot. We will do this by adding
# yet another layer, this time with the function ggtitle.

# Edit the code above to add the title "Gun murder data" to the plot.

p + scale_x_log10() +
  scale_y_log10() +
  ggtitle("Gun murder data")

# Exercise 15. Histograms

# We are going to shift our focus from the murders dataset to explore the
# heights dataset.

# We use the geom_histogram function to make a histogram of the heights
# data frame. When reading the documentation for this function we see
# that it requires just one mapping, the values to be used for the histogram.

# What is the variable containing the heights in inches in the heights
# data frame? height

str(heights)

# Exercise 16. A second example

# We are now going to make a histogram of the heights so we will load the
# heights dataset. The following code has been pre-run for you to load the
# heights dataset.

library(dplyr)
library(ggplot2)
library(dslabs)
data(heights)

# Create a ggplot object called p using the pipe to assign the heights
# data to a ggplot object.

# Assing height to the x values through the aes function.

p <- heights %>% ggplot(aes(height))

# Exercise 17. Histograms 2

# Now we are ready to add a layer to actually make the histogram.

# Add a layer to the object p (created in the previous exercise) using
# the geom_histogram() function to make the histogram.

p <- heights %>%
  ggplot(aes(height))

p + geom_histogram()


# Exercise 18. Histogram binwidth

# Note that when we run code from the previous exercise we get the following warning:

# stat_bin() using bins = 30. Pick better value with binwidth.

# Use the binwidth argument to change the histogram made in the previous exercise to
# use bins of size 1 inch.

p + geom_histogram(binwidth = 1)

# Exercise 19. Smooth density plot

# Now instead of a histogram we are going to make a smooth density plot. In this case, 
# we will not make an object p. Instead we will render the plot using a single line of
# code. In the previous exercise, we could have created a histogram using one line of
# code like this:

heights %>%
  ggplot(aes(height)) +
  geom_histogram()

# Now instead of geom_histogram() we will use geom_density to create a smooth density
# plot.

# Add the appropriate layer to create a smooth density plot of heights.

heights %>%
  ggplot(aes(height)) +
  geom_density()

# Exercise 20. Two smooth density plots

# Now we are going to make density plots for males and females separately. We can do
# this using the group argument within the aes mapping. Because each point will be
# assigned to a different density depending on a variable from the dataset, we
# need to map within aes.

# Create separate smooth density plots for males and females by defining group by sex.
# Use the existing aes function inside of the ggplot function.

heights %>% ggplot(aes(height, group = sex)) +
  geom_density()

# Exercise 21. Two smooth density plots 2

# In the previous exercise we made the two density plots, one for each sex, using:

heights %>% ggplot(aes(height, group = sex)) +
  geom_density()

# We can also assign groups through the color or fill argument. For example, if you
# type color = sex ggplot knows you want a different color for each sex. So two
# densities must be drawn. You can therefore skip the group = sex mapping. Using
# color has the added benefit that it uses color to distinguish the groups.

# Change the density plots from the previous exercise to add color.


heights %>%
  ggplot(aes(height, color = sex)) +
  geom_density()

# Exercise 22. Two smooth density plots 3

# We can also assign groups using the fill argument. When using the geom_density()
# geometry, color creates a colored line for the smooth density plot while fill
# colors the area under the curve.

# We can see what this looks like by running the following code:

heights %>%
  ggplot(aes(height, fill = sex)) +
  geom_density()

# However, here the second density is drawn over the other. We can change this
# by using something called alpha blending.

# Set the alpha parameter to 0.2 in the geom_density function to make this change.

heights %>%
  ggplot(aes(height, fill = sex)) +
  geom_density(alpha = 0.2)

## SECTION 3. SUMMARIZING WITH DPLYR

# summarize() from the dplyr/tidyverse package computes summary statistics from
# the data frame. It returns a data frame whose column names are defined within
# the function call.

# summarize() can compute any summary function that operates on vectors and
# returns a single value, but it cannot operate on functions that return
# multiple values.

# Like most dplyr functions, summarize() is aware of variable names within
# data frames and can use them directly.

library(tidyverse)
library(dslabs)
data(heights)

# Compute the average and standard deviation for males

s <- heights %>%
  filter(sex == "Male") %>%
  summarize(average = mean(height), standard_deviation = sd(height))

# Access the average and standard deviation from summary table

s$average
s$standard_deviation

# Compute the median, min, and max

heights %>%
  filter(sex == "Male") %>%
  summarize(median = median(height),
            minimum = min(height),
            maximum = max(height))

# Alternative way to get the min, median, max in base R

quantile(heights$height, c(0, 0.5, 1))

# This code generates and error: summarize can only take functions
# that return a single value

heights %>%
  filter(sex == "Male") %>%
  summarize(range = quantile(height, c(0, 0.5, 1)))

# The dot operator allows you to access values stored in data that
# is being piped using the %>% character. The dot is a placeholder
# for the data being passed in through in the pipe.

# The dot operator allows dplyr functions to return single vectors
# or numbers instead of only data frames.

# us_murder_rate %>% .$rate is equivalent to us_murder_rate$rate.

# Note that an equivalent way to extract a single column using the
# pipe is us_murder_rate %>% pull(rate). The pull() function will
# be used in later course material.

library(tidyverse)
library(dslabs)
data(murders)

murders <- murders %>%
  mutate(murder_rate = total / population * 100000)
summarize(murders, mean(murder_rate))

# Calculate the US murder rate, generating a data frame

us_murder_rate <- murders %>%
  summarize(rate = sum(total) / sum(population) * 100000)
us_murder_rate

# Extract the numeric US murder rate with the dot operator

us_murder_rate %>% .$rate

class(us_murder_rate) # data frame

# Calculate and extract the murder rate with one pipe

us_murder_rate <- murders %>%
  summarize(rate = sum(total) / sum(population) * 100000) %>%
  .$rate

class(us_murder_rate) # numeric

# The group_by() function from dplyr converts a data frame to a 
# grouped data frame, creating groups using one or more variables.

# summarize() and some other dplyr functions will behave differently
# on grouped data frames.

# Using summarize() on a grouped data frame computes the summary
# statistics for each of the separate groups.

# Compute separate average and standard deviation for male/female
# heights.

heights %>%
  group_by(sex) %>%
  summarize(average = mean(height), standard_deviation = sd(height))

# Compute median murder rate in 4 regions of the country

murders <- murders %>%
  mutate(murder_rate = total / population *100000)

murders %>%
  group_by(region) %>%
  summarize(median_rate = median(murder_rate))

# The arrange() function from dplyr sorts a data frame by a given
# column.

# By default, arrange() sorts in ascending order (lowest to highest).
# To instead sort in descending order, use the function desc() inside
# of arrange().

# You can arrange() by multiple levels: within equivalent values
# of the first level, observations are sorted by the second, and so on.

# The top_n() function shows the top results ranked by a given variable, 
# but the results are not ordered. You can combine top_n() with
# arrange() to return the top results in order.

# The top_n() function has been superseded in favor of slice_min() and
# slice_max().

# Set up the murders object

murders <- murders %>%
  mutate(murder_rate = total / population * 100000)

# Arrange by population column, smallest to largest

murders %>% arrange(population) %>% head()

# Arrange by murder rate, smallest to largest

murders %>% arrange(murder_rate) %>% head()

# Arrange by murder rate in descending order

murders %>% arrange(desc(murder_rate)) %>% head()

# Arrange by region alphabetically, then murder rate within each region

murders %>% arrange(region, murder_rate) %>% head()

# Show the top 10 states with the highest murder rate, not ordered by rate

murders %>% top_n(10, murder_rate)

# Show the top 10 states with the highest murder rate, ordered by rate

murders %>% arrange(desc(murder_rate)) %>% top_n(10)

# Alternatively, can use the slice_max() function

murders %>% slice_max(murder_rate, n = 10)

# Assessment. Summarizing with dplyr

# Practice Exercise. National Center for Health Statistics

# To practice our dplyr skills we will be working with data from the survey
# collected by the United States National Center for Health Statistics (NCHS).
# This center has conducted a series of health and nutrition surveys since
# the 1960s.

# Starting in 1999, about 5,000 individual of all ages have been interviewed
# every year and then they complete the health examination component of the
# survey. Part of this dataset is made available via the NHANES package
# which can be loaded this way.

install.packages("NHANES")
library(NHANES)
data(NHANES)

# The NHANES data has many missing values. Remember that the main
# summarization function in R will return NA if any of the entries of the
# input vector is an NA. Here is an example.

library(dslabs)
data(na_example)
mean(na_example)
sd(na_example)

# To ignore the NAs, we can use the na.rm argument:

mean(na_example, na.rm = TRUE)
sd(na_example, na.rm = TRUE)

# Try running this code, then let us know you are ready to proceed with
# the analysis.

# Exercise 1. Blood pressure 1

# Let's explore the NHANES data. We will be exploring blood pressure in
# this data set. 

# First let's select a group to set the standard. We will use the 20-29
# year old females. Note that the category is coded with 20-29, with a
# space in front of the 20! The AgeDecade is a categorical variable
# with these ages.

# To know if someone is female, you can look at the Gender variable.

# Filter the NHANES dataset so that only 20-29 year old females are
# included and assign this new data frame to the object tab.

# Use the pipe to apply the function filter, with the appropriate
# logicals, to NHANEs.

# Remember that this age group is coded with 20-29, which includes a
# space. You can use head() to explore the NHANES table to construct
# the correct call to filter.

library(dplyr)

head(NHANES)

tab <- NHANES %>% filter(Gender == "female" & AgeDecade == " 20-29")
head(tab)

# Exercise 2. Blood pressure #2

# Now we will compute the average and standard deviation for the 
# subgroup we defined in the previous exercise (20-29 year old
# females), which we will use as reference for what is typical.

# You will determine the average and standard deviation of
# systolic blood pressure, which are stored in the BPSysAve variable
# in the NHANES dataset.

# Complete the line of code to save the average and standard
# deviation of systolic blood pressure as average and standard_deviation
# to a variable called ref.

# Use the summarize function after filtering for 20-29 year old females
# and connect the results using the pipe %>%. When doing this remember
# there are NAs in the data!

ref <- NHANES %>% filter(AgeDecade == " 20-29" & Gender == "female") %>%
  summarize(average = mean(BPSysAve, na.rm = TRUE),
            standard_deviation = sd(BPSysAve, na.rm = TRUE))

# Exercise 3. Summarizing averages

# Now we will repeat the exercise and generate only the average blood
# pressure for 20-29 year old females. For this exercise, you should
# review how to use the place holder . in dplyr or the pull function.

# Modify the line of sample code to assign the average to a numeric
# variable called ref_avg using . or pull.

ref_avg <- NHANES %>%
  filter(AgeDecade == " 20-29" & Gender == "female") %>%
  summarize(average = mean(BPSysAve, na.rm = TRUE),
            standard_deviation = sd(BPSysAve, na.rm = TRUE)) %>%
  .$average

# Exercise 4. Min and max

# Let's continue practicing by calculating two other data summaries:
# the minimum and the maximum.

# Again we will do it for the BPSysAve variable and the group of
# 20-29 year old females.

# Report the min and max values for the same group as in the previous
# exercises.

# Use filter and summarize connected by the pipe %>% again. The
# functions min and max can be used to get the values you want.

# Within summarize, save the min and max of systolic blood pressure as
# minbp and maxbp.

NHANES %>%
  filter(AgeDecade == " 20-29" & Gender == "female") %>%
  summarize(minbp = min(BPSysAve, na.rm = TRUE),
            maxbp = max(BPSysAve, na.rm = TRUE))

# Exercise 5. group_by

# Now let's practice using the group_by function.

# What we are about to do is a very common operation in data science:
# you will split a data table into groups and then compute summary
# statistics for each group.

# We will compute the average and standard deviation of systolic
# blood pressure for females for each age group separately.
# Remember that the age groups are contained in AgeDecade.

# Use the functions filter, group_by, summarize, and the pipe %>%
# to compute the average and standard deviation of systolic blood
# pressure for females for each age group separately.

# Within summarize, save the average and standard deviation of
# systolic blood pressure (BPSysAve) as average and standard_deviation.

# Note: ignore warnings about implicit NAs. This warning will not
# prevent your code from running or being graded correctly.

NHANES %>%
  filter(Gender == "female") %>%
  group_by(AgeDecade) %>%
  summarize(average = mean(BPSysAve, na.rm = TRUE),
            standard_deviation = sd(BPSysAve, na.rm = TRUE))

# Exercise 6. group_by example 2

# Now let's practice using group_by some more. We are going to
# repeat the previous exercise of calculating the average and
# standard deviation of systolic blood pressure, but for males
# instead of females.

# This time we will not provide much sample code. You are on
# your own!

# Calculate the average and standard deviation of systolic
# blood pressure for males for each age group separately
# using the same methods as in the previous exercise.

# Note: ignore warnings about implicit NAs. This warning will
# not prevent your code from running or being graded correctly.

NHANES %>%
  filter(Gender == "male") %>%
  group_by(AgeDecade) %>%
  summarize(average = mean(BPSysAve, na.rm = TRUE),
            standard_deviation = sd(BPSysAve, na.rm = TRUE))

# Exercise 7. group+by example 3

# We can actually combine both of these summaries into a single
# line of code. This is because group_by permits us to group by
# more than one variable.

# We can use group_by(AgeDecade, Gender) to group by both age
# decades and gender.

# Create a single summary table for the average and standard
# deviation of systolic blood pressure using
# group_by(AgeDecade, Gender).

# Note that we no longer have to filter!

# Your code within summarize should remain the same as in the 
# previous exercises.

# Note: ignore warnings about implicit NAs. This warning will
# not prevent your code from running or being graded correctly.

NHANES %>%
  group_by(AgeDecade, Gender) %>%
  summarize(average = mean(BPSysAve, na.rm = TRUE),
            standard_deviation = sd(BPSysAve, na.rm = TRUE))

# Exercise 8. Arrange

# Now we are going to explore differences in systolic blood
# pressure across races, as reported in the Race1 variable.

# We will learn to use the arrange function to order the
# outcome according to one variable.

# Note that this function can be used to order any table by
# a given outcome. Here is an example that arranges by
# systolic blood pressure

NHANES %>% arrange(BPSysAve)

# If we want it in descending order we can use the desc
# functionlike this:

NHANES %>% arrange(desc(BPSysAve))

# In this example, we will compare systolic blood pressure
# across values of the Race1 variable for males between
# the ages of 40-49.

# Compute the average and standard deviation for each
# value of Race1 for males in the age decade 40-49.

# Order the resulting table from lowest to highest
# average systolic blood pressure.

# Use the functions filter, group_by, summarize, arrange,
# and the pipe %>% to do this in one line of code.

# Within summarize, save the average and standard deviation
# of systolic blood pressure as average and standard_deviation.

NHANES %>%
  filter(AgeDecade == " 40-49" & Gender == "male") %>%
  group_by(Race1) %>%
  summarize(average = mean(BPSysAve, na.rm = TRUE),
            standard_deviation = sd(BPSysAve, na.rm = TRUE)) %>%
  arrange(average)

## SECTION 4. GAPMINDER

# 4.1. INTRODUCTION TO GAPMINDER

# Load and inspect the gapminder data

library(dslabs)
library(dplyr)
data("gapminder")
head(gapminder)

# Compare infant mortality in Sri Lanka and Turkey

gapminder %>%
  filter(year == 2015 & country %in% c("Sri Lanka", "Turkey")) %>%
  select(country, infant_mortality)

# Basic scatterplot of life expectancy versus fertility

library(ggplot2)
library(ggthemes)

ds_theme_set() # set plot theme
filter(gapminder, year == 1962) %>%
  ggplot(aes(fertility, life_expectancy)) +
  geom_point()

# Add color as continent

filter(gapminder, year == 1962) %>%
  ggplot(aes(fertility, life_expectancy, color = continent)) +
  geom_point()

# 4.2. USING THE GAPMINDER DATASET

# Faceting makes multiple side-by-side plots stratified by some variables.
# This is a way to ease comparisons.

# The facet_grid() function allows faceting by up to two variables, with
# rows faceted by one variable and columns faceted by the other variable.
# To facet by only one variable, use the dot operator as the other
# variable.

# The facet_wrap() function facets by one variable and automatically wraps
# the series of plots so they have readable dimensions.

# Faceting keeps axes fixed across all plots, easing comparisons between
# plots.

library(dplyr)
library(ggplot2)
library(dslabs)
data("gapminder")

# Facet by continent and year

filter(gapminder, year %in% c(1962, 2012)) %>%
  ggplot(aes(fertility, life_expectancy, color = continent)) +
  geom_point() +
  facet_grid(continent ~ year)

# Facet by year only

filter(gapminder, year %in% c(1962, 2012)) %>%
  ggplot(aes(fertility, life_expectancy, color = continent)) +
  geom_point() +
  facet_grid(. ~ year)

# Facet by year, plots wrapped into multiple rows

filter(gapminder, year %in% c(1962, 1980, 1990, 2000, 2012)) %>%
  ggplot(aes(fertility, life_expectancy, color = continent)) +
  geom_point() +
  facet_wrap(~year)

# Time series plots have time on the x-axis and a variable of interest
# on th y-axis.

# The geom_line() geometry connects adjacent data points to form
# continuous lines. A line plot is appropriate when points are regularly
# spaced, densely packed and from a single data series.

# You can plot multiple lines on the same graph. Remember to group or
# color by a variable so that the lines are plotted independently.

# Labelling is usually preferred over legends. However, legends are
# easier to make and appear by default. Add a label with geom_text(), 
# specifying the coordinates where the label should appear on the graph.

# Scatterplot of U.S. fertility by year

gapminder %>%
  filter(country == "United States") %>%
  ggplot(aes(year, fertility)) +
  geom_point()

# Line plot of U.S. fertility by year

gapminder %>%
  filter(country == "United States") %>%
  ggplot(aes(year, fertility)) +
  geom_line()

# Line plot fertility time series of two countries - only one line
# INCORRECT!

countries <- c("South Korea", "Germany")
gapminder %>% filter(country %in% countries) %>%
  ggplot(aes(year, fertility)) +
  geom_line()

# Line plot fertility time series for two countries - one line per
# country

gapminder %>% filter(country %in% countries) %>%
  ggplot(aes(year, fertility, group = country)) +
  geom_line()

# Fertility time series for two countries - lines colored by country

gapminder %>% filter(country %in% countries) %>%
  ggplot(aes(year, fertility, col = country)) +
  geom_line()

# Life expectancy time series - lines colored by country and
# labeled, no legend.

library(ggthemes)

labels <- data.frame(country = countries, x = c(1975, 1965), y = c(60, 72))
gapminder %>% filter(country %in% countries) %>%
  ggplot(aes(year, life_expectancy, col = country)) +
  geom_line() +
  geom_text(data = labels, aes(x, y, label = country), size = 5) +
  theme(legend.position = "none")

# Transformations

# Log transformations convert multiplicative changes into additive changes.

# Common transformations are the log base 2 transformation and the log
# base 10 transformation. The choice of base depends on the range of the
# data. The natural log is not recommended for visualization because it
# is difficult to interpret.

# The mode of a distribution is the value with the highest frequency. The
# mode of a normal distribution is the average. A distribution can have
# multiple local modes.

# There are two ways to use log transformations in plots: transform the
# data before plotting or transform the axes of the plot. Log scales
# have the advantage of showing the original values as axis labels, 
# while log transformed values ease interpretation of intermediate
# values between labels.

# Scale the x-axis using scale_x_continuous() or scale_x_log10() layers
# in ggplot2. Similar functions exist for the y-axis.

# In 1970, income distribution is bimodal, consistent with the
# dichotomous Western vs. developing worldview.

# Add dolars per day variable

gapminder <- gapminder %>% mutate(dollars_per_day = gdp/population/365)

# Histogram of dollars per day

past_year <- 1970

gapminder %>%
  filter(year == past_year & !is.na(gdp)) %>%
  ggplot(aes(dollars_per_day)) +
  geom_histogram(binwidth = 1, color = "black")

# Repeat histogram with log2 scaled data

gapminder %>%
  filter(year == past_year & !is.na(gdp)) %>%
  ggplot(aes(log2(dollars_per_day))) +
  geom_histogram(binwidth = 1, color = "black")

# Repeat histogram with log scaled x-axis

gapminder %>%
  filter(year == past_year & !is.na(gdp)) %>%
  ggplot(aes(dollars_per_day)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(trans = "log2")

# Stratify and Boxplot

# Make boxplots stratified by a categorical variable using the
# geom_boxplot() geometry.

# Rotate axis labels by changing the theme through element_text().
# You can change the angle and justification of the text labels.

# Consider ordering your factors by a meaningful value with the
# reorder() function, which changes the order of factor levels
# based on a related numeric vector. This is a way to ease
# comparisons.

# Show the data by adding data points to the boxplot with a
# geom_point() layer. This adds information beyond the five-number
# summary to your plot, but too many data points can obfuscate
# your message.

# Boxplot of GDP by region

# Add dollars per day variable

gapminder <- gapminder %>%
  mutate(dollars_per_day = gdp/population/365)

# number of regions

length(levels(gapminder$region))

# Boxplot of GDP by region in 1970

past_year <-  1970

p <- gapminder %>%
  filter(year == past_year & !is.na(gdp)) %>%
  ggplot(aes(region, dollars_per_day))
p + geom_boxplot()

# Rotate names on the x-axis

p + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# The reorder() function

# By default, factor order is alphabetical

fac <- factor(c("Asia", "Asia", "West", "West", "West"))
levels(fac)

# Reorder factor by the category means

value <- c(10, 11, 12, 6, 4)
fac <- reorder(fac, value, FUN = mean)
levels(fac)

# Enhanced boxplot ordered by median income, scaled, and showing data

# Reorder by median income and color by continent

p <- gapminder %>%
  filter(year == past_year & !is.na(gdp)) %>%
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>% # reorder
  ggplot(aes(region, dollars_per_day, fill = continent)) + # color by continent
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab(" ")
p

# log2 scale y-axis

p + scale_y_continuous(trans = "log2")

# Add data points

p + scale_y_continuous(trans = "log2") + geom_point(show.legend = FALSE)

# Comparing Distributions

# Use intersect() to find the overlap between two vectors

# To make boxplots where grouped variables are adjacent, color the boxplot by
# a factor instead of faceting by that factor. This is a way to ease
# comparisons.

# The data suggest that the income gap between rich and poor countries has
# narrowed, not expanded.

# Histogram of income in West vs. developing world, 1970 - 2010

library(dplyr)
library(ggplot2)
library(ggthemes)
library(dslabs)
data(gapminder)

# Add dollars per day variable and define past year

gapminder <- gapminder %>%
  mutate(dollars_per_day = gdp/population/365)
past_year <- 1970

# Define Western countries

west <- c("Western Europe", "Northern Europe", "Southern Europe", "North America",
          "Australia and New Zealand")

# Facet by West vs. developing

gapminder %>%
  filter(year == past_year & !is.na(gdp)) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(trans = "log2") +
  facet_grid(. ~ group)

# Facet by West/developing and year

present_year <- 2010

gapminder %>%
  filter(year %in% c(past_year, present_year) & !is.na(gdp)) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(trans = "log2") +
  facet_grid(year ~ group)

# Income distribution of West vs. developing world, only countries
# with data

# Define countries that have data available in both years

country_list_1 <- gapminder %>%
  filter(year == past_year & !is.na(dollars_per_day)) %>% .$country
country_list_2 <- gapminder %>%
  filter(year == present_year & !is.na(dollars_per_day)) %>% .$country
country_list <- intersect(country_list_1, country_list_2)

# Make histogram including only countries with data available in both
# years

gapminder %>%
  filter(year %in% c(past_year, present_year) & country %in% country_list) %>% 
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous(trans = "log2") +
  facet_grid(year ~ group)

# Boxplots of income in West vs. developing world, 1970 - 2010

p <- gapminder %>%
  filter(year %in% c(past_year, present_year) & country %in% country_list) %>%
  mutate(region = reorder(region, dollars_per_day, FUN = median)) %>%
  ggplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab(" ") + scale_y_continuous(trans = "log2")

p + geom_boxplot(aes(region, dollars_per_day, fill = continent)) +
  facet_grid(year ~ .)

# Arrange matching boxplots next to each other, colored by year

p + geom_boxplot(aes(region, dollars_per_day, fill = factor(year)))

# Density plots

# Change the y-axis of density plots to variable counts using ..count..
# as the y argument

# The case_when() function defines a factor whose levels are defined
# by a variety of logical operations to group data.

# Plot stacked density plots using position = "stack".

# Define a weight aesthetic mapping to change the relative weights of
# density plots - for example, this allows weighting plots by 
# population rather than number of countries.

# Faceted smooth density plots

# Smooth density plots - area under each curve adds to 1

gapminder %>%
  filter(year == past_year & country %in% country_list) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>% group_by(group) %>%
  summarize(n = n()) %>% knitr::kable()

# Smooth density plots - variable counts on y-axis

p <- gapminder %>%
  filter(year == past_year & country %in% country_list) %>%
  mutate(group = ifelse(region %in% west, "West", "Developing")) %>%
  ggplot(aes(dollars_per_day, y = ..count.., fill = group)) +
  scale_x_continuous(trans = "log2")
p + geom_density(alpha = 0.2, bw = 0.75) + facet_grid(year ~ .)

# Add new region groups with case_when()

# Add group as a factor, grouping regions

gapminder <- gapminder %>%
  mutate(group = case_when(
    .$region %in% west ~ "West",
    .$region %in% c("Eastern Asia", "South-Eastern Asia") ~ "East Asia",
    .$region %in% c("Caribbean", "Central America", "South America") ~ "Latin America",
    .$continent == "Africa" & .$region != "North Africa" ~ "Sub-Saharan Africa",
    TRUE ~ "Others"))

# Reorder factor levels

gapminder <- gapminder %>%
  mutate(group = factor(group, levels = c("Other", "Latin America", "East Asia", "Sub-Saharan
                                          Africa", "West")))

# Stacked density plot

# Note you must redefine p with the new gapminder object first

p <- gapminder %>%
  filter(year %in% c(past_year, present_year) & country %in% country_list) %>%
  ggplot(aes(dollars_per_day, fill = group)) +
  scale_x_continuous(trans = "log2")

# Stacked density plot

p + geom_density(alpha = 0.2, bw = 0.75, position = "stack") +
  facet_grid(year ~ .)

# Weighted stacked density plot

gapminder %>%
  filter(year %in% c(past_year, present_year) & country %in% country_list) %>%
  group_by(year) %>%
  mutate(weight = population/sum(population * 2)) %>%
  ungroup() %>%
  ggplot(aes(dollars_per_day, fill = group, weight = weight)) +
  scale_x_continuous(trans = "log2") +
  geom_density(alpha = 0.2, bw = 0.75, position = "stack") + facet_grid(year ~ .)

# Ecological fallacy

# The breaks argument allows us to set the location of the axis labels and tick
# marks.

# The logistic or logit transformation is defined as f(p) = log (p / (1 - p)),
# or the log of odds. This scale is useful for highlighting differences near
# 0 or near 1 and converts fold changes into constant increases.

# The ecological fallacy is assuming that conclusions made from the average
# of a group apply to all members of that group.

# Define gapminder

library(tidyverse)
library(dslabs)
data(gapminder)

# Add additional cases

gapminder <- gapminder %>%
  mutate(group = case_when(
    .$region %in% west ~ "The West",
    .$region == "Northern Africa" ~ "Northern Africa",
    .$region %in% c("Eastern Asia", "South-Easter Asia") ~ "East Asia",
    .$region == "Southern Asia" ~ "Southern Asia",
    .$region %in% c("Central America", "South America", "Caribbean") ~ "Latin America",
    .$continent == "Africa" & .$region != "Northern Africa" ~ "Sub-Saharan Africa",
    .$region %in% c("Melanesia", "Micronesia", "Polynesia") ~ "Pacific Islands"))

# Define a data frame with group average income and average infant survival

surv_income <- gapminder %>%
  filter(year %in% present_year & !is.na(gdp) & !is.na(infant_mortality) & !is.na(group)) %>%
  group_by(group) %>%
  summarize(income = sum(gdp) / sum(population) / 365,
            infant_survival_rate = 1 - sum(infant_mortality/1000*population)/sum(population))
surv_income %>% arrange(income)

# Plot infant survival versus income, with transformed axes

surv_income %>% ggplot(aes(income, infant_survival_rate, label = group, color = group)) +
  scale_x_continuous(trans = "log2", limit = c(0.25, 150)) +
  scale_y_continuous(trans = "logit", limit = c(0.875, .9981),
                     breaks = c(0.85, 0.90, 0.95, 0.99, 0.995, 0.998)) +
  geom_label(size = 3, show.legend = FALSE)

# Assessment. Exploring the gapminder dataset

# Exercise 1. Life expectancy vs. fertility - part 1

# The Gapminder Foundation (www.gapminder.org) is a non-profit organization based
# in Sweden that promotes global development through the use of statistics that
# can help reduce misconceptions about lobal development.

# Using ggplot and the points layer, create a scatter plot of life expectancy
# versus fertility for the African continent in 2012.

# Remember that you can use the R console to explore the gapminder dataset to
# figure out the names of the columns in the dataframe.

# In this exercise we provide parts of code to get you going.  You need to fill
# out what is missing. But note that going forward, in the next exercises, you
# will be required to write most of the code.

library(dplyr)
library(ggplot2)
library(dslabs)
data(gapminder)

gapminder %>% filter(year == 2012 & continent == "Africa") %>%
  ggplot(aes(fertility, life_expectancy)) +
  geom_point()

# Exercise 2. Life expectancy vs. fertility - part 2

# Note that there is quite a bit of variability in life expectancy and fertility
# with some African countries having very high life expectancies. There also
# appear to be three clusters in the plot.

# Remake the plot from the previous exercises but this time use color to distinguish
# the different regions of Africa to see if this explains the clusters. Remember
# that you can explore the gapminder data to see how the regions of Africa are
# labelled in the data frame!

# Use color rather than col inside your ggplot call - while these two forms are
# equivalent in R, the grader specifically looks for `color`

gapminder %>% filter(year == 2012 & continent == "Africa") %>%
  ggplot(aes(fertility, life_expectancy, color = region)) +
  geom_point()

# Exercise 3. Life expectancy vs. fertility - part 3 - selecting country and region

# While many of the countries in the high life expectancy/low fertility cluster are
# from Northern Africa, three countries are not.

# Create a table showing the country and region for the African countries (use select)
# that in 2012 had fertility rates of 3 or less and life expectancies of at least 70.

# Assign your result to a data frame called df.

df <- gapminder %>%
  filter(year == 2012 & continent == "Africa" & fertility <= 3 & life_expectancy >= 70) %>%
  select(country, region)

# Exercise 4. Life expectancy and the Vietnam War - part 1

# The Vietnam War lasted from 1955 to 1975. Do the data support war having a negative effect
# on life expectancy? We will create a time series plot that covers the period from 1960 to
# 2010 of life expectancy for Vietnam and the United States, using color to distinguish the
# two countries. In this start we start the analysis by generating a table.

# Use filter to create a table with data for the years from 1960 to 2010 in Vietnam and the
# United States.

# Save the table in an object called tab.

tab <- gapminder %>% filter(year %in% seq(1960, 2010) & country %in% c("Vietnam", "United States"))

# Exercise 5. Life expectancy and the Vietnam War - part 2

# Now that you have created the data table in Exercise 4, it is time to plot the data for
# the two countries.

# Use geom_line to plot life expectancy vs year for Vietnam and the United States and save the
# plot as p. The data table is stored in tab.

# Use color to distinguish the two countries.

# Print the object p.

p <- tab %>%
  ggplot(aes(year, life_expectancy, color = country)) +
  geom_line()
p

# Exercise 6. Life expectancy in Cambodia

# Cambodia was also involved in this conflict and, after the war, Pol Pot and his communist
# Khmer Rouge took control and ruled Cambodia from 1975 to 1979. He is considered one of the
# most brutal dictators in history. Do the data support this claim?

# Use a single line of code to create a time series plot from 1960 to 2010 of life expectancy
# vs year for Cambodia.

gapminder %>% filter(year %in% seq(1960, 2010) & country == "Cambodia") %>%
  ggplot(aes(year, life_expectancy)) +
  geom_line()

# Exercise 7. Dollars per day - part 1

# Now we are going to calculate and plot dollars per day for African countries in 2010
# using GDP data.

# In the first part of this analysis, we will create the dollars per day variable.

# Use mutate() to create a dollars_per_day variable, which is defined as
# gdp / population / 365.

# Create a dollars_per_day variable for African countries for the year 2010.

# Remove any NA values.

# Save the mutated dataset as daydollars.

daydollars <- gapminder %>%
  filter(year == 2010 & continent == "Africa") %>%
  mutate(dollars_per_day = gdp / population / 365) %>%
  filter(!is.na(dollars_per_day))

# Exercise 8. Dollars per day - part 2

# Now we are going to calculate and plot dollars per day for African countries in 2010
# using GDP data.

# In the second part of this analysis, we will plot the smooth density plot using a log
# (base 2) x axis.

# The dataset including the dollars_per_day variable is preloaded as daydollars.

# Create a smooth density plot of dollars per day from daydollars

# Use scale_x_continuous to change the x-axis to a log (base 2) scale.

daydollars %>%
  ggplot(aes(dollars_per_day)) +
  geom_density() +
  scale_x_continuous(trans = "log2")

# Exercise 9. Dollars per day - part 3 - multiple density plots

# Now we are going to combine the plotting tools we have used in the past two exercises
# to create density plots for multiple years.

# Create the dollars per day variable as in Exercise 7, but for African countries in years
# 1970 and 2010 this time.

# Make sure you remove any NA values.

# Create a smooth density plot of dollars per day for 1970 and 2010 using a log (base 2)
# scale for the x axis.

# Use facet_grid to show a different density plot for 1970 and 2010.

gapminder %>%
  filter(year %in% c(1970, 2010) & continent == "Africa") %>%
  mutate(dollars_per_day = gdp / population / 365) %>%
  filter(!is.na(dollars_per_day)) %>%
  ggplot(aes(dollars_per_day)) +
  geom_density() +
  scale_x_continuous(trans = "log2") +
  facet_grid(year ~ .)

# Exercise 10. Dollars per day - part 4 - stacked density plot

# Now we are going to edit the code from Exercise 9 to show a stacked density plot of each
# region in Africa. 

# Much of the code will be the same as in Exercise 9:

# Create the dollars_per_day variable as in Exercise 7, but for African countries in the
# years 1970 and 2010 this time.

# Make sure you remove any NA values.

# Create a smooth density plot of dollars per day for 1970 and 2010 using a log (base 2)
# scale for the x axis.

# Use facet_grid to show a different density plot for 1970 and 2010.

# Make sure the densities are smooth by using bw = 0.5.

# Use fill and position arguments where appropriate to create the stacked density plot of
# each region.

gapminder %>%
  filter(year %in% c(1970, 2010) & continent == "Africa") %>%
  mutate(dollars_per_day = gdp / population / 365) %>%
  filter(!is.na(dollars_per_day)) %>%
  ggplot(aes(dollars_per_day, fill = region)) +
  geom_density(bw = 0.5, position = "stack") +
  scale_x_continuous(trans = "log2") +
  facet_grid(year ~ .)

# Exercise 11. Infant mortality scatter plot - part 1

# We are going to continue looking at patterns in the gapminder dataset by plotting
# infant mortality rates versus dollars per day for African countries

# Generate dollars_per_day using mutate() and filter for the year 2010 for African countries.
# Remember to remove NA values.

# Store the mutated dataset in gapminder_Africa_2010.

# Make a scatterplot of infant_mortality versus dollars_per_day for countries in the African
# continent.

# Use color to denote the different regions of Africa.

gapminder_Africa_2010 <- gapminder %>%
  filter(year == 2010 & continent == "Africa") %>%
  mutate(dollars_per_day = gdp / population / 365) %>%
  filter(!is.na(dollars_per_day))

gapminder_Africa_2010 %>%
  ggplot(aes(dollars_per_day, infant_mortality, color = region)) +
  geom_point()

# Exercise 12. Infant mortality scatter plot - part 2 - logarithmic axis

# Now we are going to transform the x axis of the plot from the previous exercise.

# The mutated dataset is preloaded as gapminder_Africa_2010.

# As in the previous exercise, make a scatterplot of infant_mortality versus
# dollars_per_day for countries in the African continent.

# As in the previous exercise, use color to denote the different regions of Africa.

# Transform the x-axis to be in the log (base 2) scale.

gapminder_Africa_2010 %>%
  ggplot(aes(dollars_per_day, infant_mortality, color = region)) +
  geom_point() +
  scale_x_continuous(trans = "log2")

# Exercise 13. Infant mortality scatter plot - part 3 - adding labels

# Note that there is a large variation in infant mortality and dollars per day among
# African countries.

# As an example, one country has infant mortality rates of less than 20 per 1000 and dollars
# per day of 16, while another country has infant mortality rates over 10% and dollars per
# day of about 1.

# In this exercise, we will remake the plot from Exercise 12 with country names instead of points
# so we can identify which countries are which.

# The mutated dataset is preloaded as gapminder_Africa_2010.

# As in the previous exercise, make a scatterplot of infant_mortality versus dollars_per_day for
# countries in the African continent.

# As in the previous exercise, use color to denote the different regions of Africa.

# As in the previous exercise, transform the x axis to be in the log (base 2) scale.

# Add a geom_text layer to display country names in addition to of points.

gapminder_Africa_2010 %>%
  ggplot(aes(dollars_per_day, infant_mortality, color = region, label = country)) +
  geom_point() +
  scale_x_continuous(trans = "log2") +
  geom_text()

# Exercise 14. Infant mortality scatter plot - part 4 - comparison of scatter plots

# Now we are going to look at changes in the infant mortality and dollars per day patterns in
# African countries between 1970 and 2010.

# Generate dollars_per_day using mutate and filter for the years 1970 and 2010 for African countries.
# Remember to remove any NA values.

# As in the previous exercise, make a scatter plot of infant_mortality versus dollars per day for
# countries in the African continent.

# As in the previous exercise, use color to denote the different regions of Africa.

# As in the previous exercise, transform the x-axis to be in the log (base 2) scale.

# As in the previous exercise, add a layer to display country names instead of points.

# Use facet_grid to show different plots for 1970 and 2010. Align the plots vertically.

gapminder %>%
  filter(year %in% c(1970, 2010) & continent == "Africa") %>%
  mutate(dollars_per_day = gdp / population / 365) %>%
  filter(!is.na(dollars_per_day) & !is.na(infant_mortality)) %>%
  ggplot(aes(dollars_per_day, infant_mortality, color = region, label = country)) +
  geom_point() +
  scale_x_continuous(trans = "log2") +
  geom_text() +
  facet_grid(year ~ .)

## SECTION 5. DATA VISUALIZATION PRINCIPLES

# 5.1. Data Visualization Principles, Part 1

# Exercise 1. Customizing plots - pie charts

# Pie charts are appropriate:

# When we want to display percentages.

# When ggplot is not available.

# When I am in a bakery.

# NEVER. BARPLOTS AND TABLES ARE ALWAYS BETTER # This is the correct answer

# Exercise 2. Customizing plots - what's wrong?

# What is the problem with this plot?

# The values are wrong. The final vote was 306 to 232.

# THE AXIS DOES NOT START AT ZERO. JUDGING BY THE LENGTH, IT APPEARS THAT
# TRUMP RECEIVED THREE TIMES AS MANY VOTES WHEN IN FACT IT WAS ABOUT
# 30% MORE. # This is the correct answer.

# The colors should be the same.

# Percentages should be shown as a pie chart.

# Exercise 3. Customizing plots - what's wrong 2?

# Take a look at the following two plots. They show the same information: rates of
# measles by state in the United States for 1928.

# Both plots provide the same information, so they are equally good.

# The plot on the left is better because it orders the states alphabetically.

# THE PLOT ON THE RIGHT IS BETTER BECAUSE IT ORDERS THE STATES BY DESEASE RATE
# SO WE CAN QUICKLY SEE THE STATES WITH THE HIGHEST AND LOWEST RATES. # This is correct.

# Both pots should be pie charts instead.

# 5.2 Data Visualization Principles, Part 2

# Show the data

# A dynamite plot - a bar graph of group averages with error bars denoting standard
# errors - provides almost no information about a distribution.

# By showing the data, you provide viewers extra information about distributions.

# Jitter is adding a small random shift to each point in order to minimize the number
# of overlapping points. To add jitter, use the geom_jitter() grometry instead of
# geom_point().

# Alpha blending is making points somewhat transparent, helping visualize the density
# of overlapping points. Add an alpha argument to the geometry.

data(heights)

# dot plot showing the data

heights %>% ggplot(aes(sex, height)) + geom_point()

# Jittered, alpha blended point plot

heights %>% ggplot(aes(sex, height)) + geom_jitter(width = 0.1, alpha = 0.2)

#  Assessment. Data Visualization Principles, Part 2

# Exercise 1. Customizing plots - watch and learn

# To make the plot on the right in the exercise from the last set of assessments, we
# had to reorder the levels of the states variables.

# Redefine the state object so that the levels are re-ordered by rate.

# Print the new object state and its levels (using levels) so you can see that the vector
# is now re-ordered by the levels.

library(dplyr)
library(ggplot2)
library(dslabs)
dat <- us_contagious_diseases %>%
  filter(year == 1967 & disease == "Measles" & !is.na(population)) %>%
  mutate(rate = count / population * 10000 * 52 / weeks_reporting)
state <- dat$state
rate <- dat$count / (dat$population / 10000) * (52 / dat$weeks_reporting)

state <- reorder(state, rate)
levels(state)

# Exercise 2. Customizing plots - redefining

# Now we are going to customize this plot a little more by creating a rate variable
# and reordering by that variable instead.

# Add a single line of code to the definition of the dat table that uses mutate()
# to reorder the states by the rate variable.

# The sample code provided will then create a bar plot using the newly defined dat.

library(dplyr)
library(ggplot2)
library(dslabs)
data(us_contagious_diseases)
dat <- us_contagious_diseases %>%
  filter(year == 1967 & disease == "Measles" & !is.na(population)) %>%
  mutate(rate = count / population * 10000 * 52 / weeks_reporting) %>%
  mutate(state = reorder(state, rate))

dat %>% ggplot(aes(state, rate)) +
  geom_bar(stat = "identity") +
  coord_flip()

# Exercise 3. Showing the data and customizing plots

# Say we are interested in comparing gun homicide rates across regions of the U.S.
# We see this bar chart:

library(dplyr)
library(ggplot2)
library(dslabs)
data(murders)
murders %>% mutate(rate = total / population * 100000) %>%
  group_by(region) %>%
  summarize(avg = mean(rate)) %>%
  mutate(region = factor(region)) %>%
  ggplot(aes(region, avg)) +
  geom_bar(stat = "identity") +
  ylab("Murder Rate Average")

# and decide to move to a state in the western region. What is the main problem with
# this interpretation?

# The categories are ordered alphabetically.

# The graph does not show standard errors.

# IT DOES NOT SHOW ALL THE DATA. WE DO NOT SEE THE VARIABILITY WITHIN A REGION AND
# IT'S POSSIBLE THAT THE SAFEST STATES ARE NOT IN THE WEST. # This is the correct one

# The Northeast has the lowest average

# Exercise 4. Making a boxplot

# To further investigate whether moving to a western region is a wise decision, let's
# make a box plot of murder rates by region, showing all points.

# Order the regions by their median murder rate by using mutate and reorder.

# Make a box plot of the murder rates by region.

# Show all of the points on the box plot.

library(dplyr)
library(ggplot2)
library(dslabs)
data(murders)

murders %>% mutate(rate = total / population * 100000) %>%
  mutate(region = reorder(region, rate, FUN = median)) %>%
  ggplot(aes(region, rate)) +
  geom_boxplot() +
  geom_point()

# 5.3. Data Visualization Principles, Part 3

# Slope Charts

# Consider using a slope chart or Bland-Altman plot when comparing one variable at two
# different time points, especially for a small number of observations.

# Slope charts use angle to encode change. Use geom_line() to create slope charts. It is
# useful when comparing a small number of observations.

# The Bland-Altman plot (Tukey mean difference plot, MA plot) graphs the difference 
# between conditions on the y-axis and the mean between conditions on the x-axis. It is 
# more appropriate for large numbers of observations than slope charts.

library(tidyverse)
library(dslabs)
data(gapminder)

west <- c("Western Europe", "Northern Europe", "Southern Europe", "Northern America",
          "Australia and New Zealand")

dat <- gapminder %>%
  filter(year %in% c(2010, 2015) & region %in% west & !is.na(life_expectancy) & population > 10^7)

dat %>%
  mutate(location = ifelse(year == 2010, 1, 2),
         location = ifelse(year == 2015 & country %in% c("United Kingdom", "Portugal"),
                           location + 0.22, 0),
         hjust = ifelse(year == 2010, 1, 0)) %>%
  mutate(year = as.factor(year)) %>%
  ggplot(aes(year, life_expectancy, group = country)) +
  geom_line(aes(color = country), show.legend = FALSE) +
  geom_text(aes(x = location, label = country, hjust = hjust), show.legend = FALSE) +
  xlab(" ") +
  ylab("Life Expectancy")

# Bland-Altman plot

library(ggrepel)

dat %>%
  mutate(year = paste0("life_expectancy_", year)) %>%
  select(country, year, life_expectancy) %>% spread(year, life_expectancy) %>%
  mutate(average = (life_expectancy_2015 + life_expectancy_2010) / 2,
         difference = life_expectancy_2015 - life_expectancy_2010) %>%
  ggplot(aes(average, difference, label = country)) +
  geom_point() +
  geom_text_repel() +
  geom_abline(lty = 2) +
  xlab("Average of 2010 and 2015") +
  ylab("Difference between 2015 and 2010")

# Case study: Vaccines

# Vaccines save millions of lives, but misinformation has led some to question the safety of
# vaccines. The data support vaccines are safe and effective. We visualize data about measles
# incidence in order to demonstrate the impact of vaccination programs on disease rate.

# The RColorBrewer package offers several color palettes. Sequential color palettes are best
# suited to data that span from high to low. Diverging color palettes are best suited for data
# that are centered and diverge towards high or low values.

# The geom_tile() geometry creates a grid of colored tiles.

# Position and length are stronger cues than color for numeric values, but color can be
# appropriate sometimes.

# Tile plot of measles rate by year and state

# Import data and inspect

library(tidyverse)
library(dslabs)
data("us_contagious_diseases")
str(us_contagious_diseases)

# Assign dat to the per 10,000 rate of measles, removing Alaska and Hawaii and adjusting for
# weeks reporting

the_disease <- "Measles"
dat <- us_contagious_diseases %>%
  filter(!state %in% c("Hawaii", "Alaska") & disease == the_disease) %>%
  mutate(rate = count / population * 10000 * 52 / weeks_reporting) %>%
  mutate(state = reorder(state, rate))

# Plot disease rates per year in California

dat %>% filter(state == "California" & !is.na(rate)) %>%
  ggplot(aes(year, rate)) +
  geom_line() +
  ylab("Cases per 10,000") +
  geom_vline(xintercept = 1963, col = "blue")

# Tile plot of disease rate by state and year

dat %>% ggplot(aes(year, state, fill = rate)) +
  geom_tile(color = "grey50") +
  scale_x_continuous(expand = c(0,0)) +
  scale_fill_gradientn(colors = RColorBrewer::brewer.pal(9, "Reds"), trans = "sqrt") +
  geom_vline(xintercept = 1963, col = "blue") +
  theme_minimal() + theme(panel.grid = element_blank()) +
  ylab(" ") +
  xlab(" ")

# Line plot of measles rate by year and state

# Compute US average measles rate by year

avg <- us_contagious_diseases %>%
  filter(disease == the_disease) %>% group_by(year) %>%
  summarize(us_rate = sum(count, na.rm = TRUE) / sum(population, na.rm = TRUE) * 10000)

# Make line plot of measles rate by year by state

dat %>%
  filter(!is.na(rate)) %>%
  ggplot() +
  geom_line(aes(year, rate, group = state), color = "grey50",
            show.legend = FALSE, alpha = 0.2, size = 1) +
  geom_line(mapping = aes(year, us_rate), data = avg, size = 1, col = "black") +
  scale_y_continuous(trans = "sqrt", breaks = c(5, 25, 125, 300)) +
  ggtitle("Cases per 10,000 by state") +
  xlab(" ") +
  ylab(" ") +
  geom_text(data = data.frame(x = 1955, y = 50),
            mapping = aes(x, y, label = "US average"), color = "black") +
  geom_vline(xintercept = 1963, col = "blue")

# Assessment. Data Visualization Principles, Part 3

# Exercise 1. Tile plot - measles and smallpox

# The sample code given creates a tile plot showing the rate of measles cases per
# population. We are going to modify the tile plot to look at smallpox cases instead.

# Modify the tile plot to show the rate of smallpox cases instead of measles cases.

# Exclude years in which cases were reported in fewer than 10 weeks from the plot.

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(dslabs)
data(us_contagious_diseases)

the_disease = "Smallpox"
dat <- us_contagious_diseases %>% 
  filter(!state%in%c("Hawaii","Alaska") & disease == the_disease & weeks_reporting >= 10) %>% 
  mutate(rate = count / population * 10000) %>% 
  mutate(state = reorder(state, rate))

dat %>% ggplot(aes(year, state, fill = rate)) + 
  geom_tile(color = "grey50") + 
  scale_x_continuous(expand=c(0,0)) + 
  scale_fill_gradientn(colors = RColorBrewer::brewer.pal(9, "Reds"), trans = "sqrt") + 
  theme_minimal() + 
  theme(panel.grid = element_blank()) + 
  ggtitle(the_disease) + 
  ylab("") + 
  xlab("")

# Exercise 2. Time series plot - measles and smallpox

# The sample code given creates a time series plot showing the rate of measles cases per
# population by state. We are going to again modify this plot to look at smallpox
# cases instead.

# Modify the sample code for the time series plot to plot data for smallpox instead of
# measles.

# Once again, restrict the plot to years in which cases were reported in at least 10
# weeks.

library(dplyr)
library(ggplot2)
library(dslabs)
library(RColorBrewer)
data(us_contagious_diseases)

the_disease = "Smallpox"
dat <- us_contagious_diseases %>%
  filter(!state%in%c("Hawaii","Alaska") & disease == the_disease & weeks_reporting >= 10) %>%
  mutate(rate = count / population * 10000) %>%
  mutate(state = reorder(state, rate))

avg <- us_contagious_diseases %>%
  filter(disease==the_disease) %>% group_by(year) %>%
  summarize(us_rate = sum(count, na.rm=TRUE)/sum(population, na.rm=TRUE)*10000)

dat %>% ggplot() +
  geom_line(aes(year, rate, group = state),  color = "grey50", 
            show.legend = FALSE, alpha = 0.2, size = 1) +
  geom_line(mapping = aes(year, us_rate),  data = avg, size = 1, color = "black") +
  scale_y_continuous(trans = "sqrt", breaks = c(5,25,125,300)) + 
  ggtitle("Cases per 10,000 by state") + 
  xlab("") + 
  ylab("") +
  geom_text(data = data.frame(x=1955, y=50), mapping = aes(x, y, label="US average"), color="black") + 
  geom_vline(xintercept=1963, col = "blue")

# Exercise 3. Time series plot - all diseases in California

# Now we are going to look at the rates of all diseases in one state. Again, you will be 
# modifying the sample code to produce the desired plot.

# For the state of California, make a time series plot showing rates for all diseases.

# Include only years with 10 or more weeks reporting.

# Use a different color for each disease.

# Include your aes function inside of ggplot rather than inside your geom layer.

library(dplyr)
library(ggplot2)
library(dslabs)
library(RColorBrewer)
data(us_contagious_diseases)

us_contagious_diseases %>% filter(state=="California" & weeks_reporting >= 10) %>% 
  group_by(year, disease) %>%
  summarize(rate = sum(count)/sum(population)*10000) %>%
  ggplot(aes(year, rate, color = disease)) + 
  geom_line()

# Exercise 4. Time series plot - all diseases in the United States

# Now we are going to make a time series plot for the rates of all diseases in the
# United States. For this exercise, we have provided less sample code - you can
# take a look at the previous exercise to get you started.

# Compute the U.S. rate by using summarize to sum over states. Call the variable rate.
# The U.S. rate for each disease will be the total number of cases divided by the total 
# population.
# Remember to convert to cases per 10,000.

# You will need to filter for !is.na(population) to get all the data.

# Plot each disease in a different color.

us_contagious_diseases %>% filter(!is.na(population)) %>%
  group_by(year, disease) %>%
  summarize(rate = sum(count) / sum(population) * 10000) %>%
  ggplot(aes(year, rate, color = disease)) +
  geom_line()

# Section 5. Assessment: Titanic Survival

# Put all your new skills together to perform exploratory data analysis on a classic
# machine learning dataset.

# Background

# The Titanic was a British ocean liner that struck an iceberg and sunk on its maiden
# voyage in 1912 from the UK to New York. More than 1,500 of the estimated 2,224 passengers
# and crew died in the accident, making this one of the largest maritime disasters ever
# outside of war. The ship carried a wide range of passengers of all ages and both genders,
# from luxury travelers in first class to immigrants in the lower classes. However, not all
# passengers were equally likely to survive the accident. We use real data about a selection
# of 891 passengers to learn who was on the Titanic and which passengers were more likely
# to survive.

# Libraries, Options, and Data

# Be sure that you have installed the titanic package before proceding.

install.packages("titanic")

# Define the titanic dataset starting from the titanic library with the following code:

options(digits = 3) # report 3 significant digits
library(tidyverse)
library(titanic)

titanic <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(Survived = factor(Survived),
         Pclass = factor(Pclass),
         Sex = factor(Sex))

# Question 1. Variable Types

# Inspect the data and also use ?titanic_train to learn more about the variables in the dataset.
# Match this variables from the dataset to their variable type. There is at least one variable
# in each type (ordinal categorical, non-ordinal categorical, continuous, discrete).

# Survived # non-ordinal categorical

class(titanic_train$Survived) # integer
head(titanic_train$Survived) # 0 1 1 1 0 0

# Pclass # ordinal categorical

class(titanic_train$Pclass) # integer
head(titanic_train$Pclass) # 3 1 3 1 3 3
range(titanic_train$Pclass) # 1 3

# Sex # non-ordinal categorical

class(titanic_train$Sex) # character
head(titanic_train$Sex)

# SibSp # Discrete

class(titanic_train$SibSp) # integer
head(titanic_train$SibSp) # 1 1 0 1 0 0

# Parch # Discrete

class(titanic_train$Parch) # integer
head(titanic_train$Parch)

# Fare # Continuous

class(titanic_train$Fare) # numeric
head(titanic_train$Fare) # 7.25 71.28 ...

# Question 2. Demographics of Titanic Passengers

# Make density plots of age grouped by sex. Try experimenting with combinations of faceting, alpha
# blending, stacking, and using variable counts on the y-axis to answer the following questions.
# Some questions may be easier to answer with different versions of the density plot.

# Which of the following are true?

# Females and males had the same general shape of age distribution # Yes

titanic %>%
  ggplot(aes(Age, group = Sex, col = Sex)) +
  geom_density()

# The age distribution was bimodal with one mode around 25 years of age and a second smaller
# mode around 5 years of age # Yes

titanic %>%
  ggplot(aes(Age)) +
  geom_histogram(binwidth = 10)

# There were more females than males # No

titanic %>%
  ggplot(aes(Age, y = ..count.., group = Sex, col = Sex)) +
  geom_density()

# The count of males of age 40 was higher than the count of females of age 40 # Yes

# The proportion of males age 18-35 was higher than the proportion of females aged
# 18-35 # Yes

titanic %>%
  ggplot(aes(Age, group = Sex, col = Sex, fill = Sex)) +
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 18, col = "black") +
  geom_vline(xintercept = 35, col = "black")

# The proportion of females under age 17 was higher than the proportion of males
# aged 17 # Yes

titanic %>%
  ggplot(aes(Age, group = Sex, col = Sex, fill = Sex)) +
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 17, col = "black")

# The oldest passengers were female # No

# Question 3. QQ-plot of Age Distribution

# Use geom_qq() to make a QQ-plot of passanger age and add an identity line with
# geom_abline(). Filter out any individuals with an age of NA first. Use the following
# object as the dparams argument in geom_qq():

params <- titanic %>%
  filter(!is.na(Age)) %>%
  summarize(mean = mean(Age), sd = sd(Age))

titanic %>%
  filter(!is.na(Age)) %>%
  ggplot(aes(sample = Age)) +
  geom_qq(dparams = params) + geom_abline()

# Which of the following is the correct plot according to the instructions above? # plot 3

# Question 4. Survival by Sex

# To answer the following questions, make barplots of the Survided and Sex variables using
# geom_bar(). Try plotting one variable and filling by the other variable. You may want to try the
# default plot, then try adding position = position_dodge() to geom_bar to make separate bars
# for each group.

# Which of the following are true?

# Less than half of passengers survived. # Yes

titanic %>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(Survived)) +
  geom_bar()

# Most of survivors were female # Yes

titanic %>%
  filter(!is.na(Survived) & Survived == 1) %>%
  ggplot(aes(Survived, group = Sex, fill = Sex)) +
  geom_bar()

# Most of the males survived # No

titanic %>%
  filter(!is.na(Survived) & Sex == "male") %>%
  ggplot(aes(Survived)) +
  geom_bar()

# Most of the females survived # Yes

titanic %>%
  filter(!is.na(Survived) & Sex == "female") %>%
  ggplot(aes(Survived)) +
  geom_bar()

# Question 5. Survival by Age

# Make a density plot of age filled by survival status. Change the y-axis to count and set
# alpha = 0.2

# Which age group is the only group more likely to survive than die? # 0-8

titanic %>%
  group_by(Survived) %>%
  ggplot(aes(Age, y = ..count.., color = factor(Survived))) +
  geom_density()

# Which group had the most deaths? # 18-30

# Which age group had the highest proportion of deaths? 70-80

# Survival by Fare

# Filter the data to remove individuals who paid a fare of 0. Make a boxplot of fare grouped
# by survival status. Try log 2 transformation fares. Add the data points with jitter and
# alpha blending.

# Which of the following are true?

# Passengers who survived generally payed higher fares than those who did not survive.# Yes

titanic %>% 
  filter(Fare > 0 & !is.na(Survived)) %>%
  ggplot(aes(Survived, Fare, group = Survived)) +
  geom_boxplot() +
  scale_y_continuous(trans = "log2") +
  geom_jitter(alpha = 0.2)

# The interquartile range for fares was smaller for passengers who survived. # No

# The median fare was lower for passengers who did not survive. # Yes

# Only one individual paid around $500. That individual survived. # No

# Most individuals who paid a fare around $8 did not survive. # Yes.

# Question 7. Survival by Passenger Class

# The Pclass variable corresponds to the passenger class. Make three barplots. For the
# first, make a basic barplot of passenger class filled by survival. For the second, make
# the same barplot but use the argument position = position_fill() to show relative
# proportions in each group instead of counts. For the third, make a barplot of
# survival filled by passenger class using position = position_fill().

# Which of the following are true?

titanic %>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(Pclass, fill = factor(Survived))) +
  geom_bar()

titanic %>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(Pclass, fill = factor(Survived))) +
  geom_bar(position = position_fill())
  
titanic %>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(Survived, fill = Pclass)) +
  geom_bar(position = position_fill())

# There were more third class passengers than passengers in the first two classes combined. # Yes

# There were the fewest passengers in first class, second-most passengers in second class, and
# most passengers in third class. # No

# Survival proportion was highest for first class passengers, followed by second class. Third-
# class had the lowest survival proportion. # Yes

# Most passengers in first class survived. Most passengers in other classes did not. # Yes

# The majority of survivors were from first class (50%+) # No

# The majority of those who did not survive were from third class. # Yes

# Question 8. Survival by Age, Sex and Passenger Class

# Create a grid of density plots for age, filled by survival status, with count on the y-axis,
# faceted by sex and passenger class.

# Which of the following are true?

titanic %>%
  ggplot(aes(Age, y = ..count.., fill = factor(Survived), position = "stack")) +
  geom_density(alpha = 0.5) +
  facet_grid(Sex ~ Pclass)

# The largest group of passengers was third-class males # Yes

# The age distribution is the same across passenger classes # No

# The gender distribution is the same across passenger classes # No

# Most first-class and second-class females survived. # Yes

# Almost all second-class males did not survive, with the exceptions of children. # Yes

## FINAL ASSESSMENT PART 1: Properties of Stars

# Astronomy is one of the oldest data-driven sciences. In the late 1800s,the director of
# the Harvard College Observatory hired women to analyze astronomical data, which at
# the time was done using photographic glass plates. These women became known as the
# "Harvard Computers". They computed the position and luminosity of various astronomical
# objects such as stars and galaxies. Today, astronomy is even more of a data-driven
# science, with an inordinate amomunt of data being produced by modern instruments
# every day.

# In the following exercises we will analyze some actual astronomical data to inspect
# properties of stars, their absolute magnitude (which relates to a star's luminosity,
# or brightness), temperature and type (spectral class).

# Libraries and options

library(tidyverse)
library(dslabs)
data(stars)
options(digits = 3)

# Question 1

# Load the stars data frame from dslabs. This contains the name, absolute magnitude,
# temperature in degrees Kelvin, and spectral class of selected stars. Absolute
# magnitude (shortened in these problems to simply "magnitude") is a function of 
# star luminosity, where negative values of magnitude have higher luminosity.

# What is the mean magnitude?

str(stars)
mean(stars$magnitude) # 4.26

# What is the standard deviation of magnitude?

sd(stars$magnitude) # 7.35

# Question 2

# Make a density plot of the magnitude. How many peaks are there in the data? # 2

ggplot(data = stars, aes(magnitude)) +
  geom_density()

# Question 3

# Examine the distribution of star temperature. Which of these statements best characterizes
# the temperature distribution?

ggplot(data = stars, aes(temp)) +
  geom_density()

# The majority of stars have a high temperature.

# The majority of stars have a low temperature. # This is the correct one

# The temperature distribution is normal

# There are equal numbers of stars across the temperature range.

# Question 4

# Make a scatter plot of the data with temperature on the x-axis and magnitude on the y-axis
# and examine the relationship between the variables. Recall that lower magnitude means a
# more luminous (brighter) star.

# When considering the plot of magnitude vs. temperature, most stars follow a ***** trend. This
# are called main sequence stars. Fill in the blank:

# decreasing linear

# increasing linear

# decreasing exponential # This is the right answer

# increasing exponential

ggplot(data = stars,aes(temp, magnitude)) +
  geom_point()

# Question 5

# For various reasons, scientists do not always follow straight conventions when making plots,
# and astronomers usually transform values of star luminosity and temperature before plotting.
# Flip the y-axis so that lower values of magnitude are at the top of the axis (recall that
# more luminous stars have lower magnitude) using scale_y_reverse(). Take the log base 10 and
# then also flip the x-axis.

# Fill in the blanks in the statements below to describe the resulting plot:

ggplot(data = stars, aes(log10(temp), magnitude)) +
  geom_point() +
  scale_y_reverse() +
  scale_x_reverse()

# The brightest, highest temperature stars are in the ***** corner of the plot.

# lower left

# lower right

# upper left # This is the correct answer

# upper right

# For main sequence stars, hotter stars have ***** luminosity.

# higher # This is the correct one

# lower

# Question 6

# The trends you see allow scientists to learn about the evolution and lifetime of stars.
# The primary group of stars to which most stars belong we will call the main sequence
# stars (discussed in question 4). Most stars belong to this main sequence, however some
# of the more rare stars are classified as "old" and "evolved" stars. These stars tend to
# be hotter stars, but also have low luminosity, and are known as white dwarfs.

# How many white dwarfs are there in our sample? # 4

# Question 7

# Consider stars which are not part of the Main Group but are not old/evolved (white dwarf)
# stars. These stars must also be unique in certain ways and are known as giants. Use the
# plot from question 5 to estimate the average temperature of a giant.

# Whic of these temperatures is closest to the average temperature of a giant?

# 5000K # This is the correct answer; points in the top right corner

# 10000K

# 15000K

# 20000K

# Question 8

# We can now identify whether specific stars are main sequence stars, red giants or white
# dwarfs. Add text labels to the plot to answer these questions. You may wish to plot only
# a selection of labels, repel the labels, or zoom in on the plot in RStudio so you can locate
# specific stars.

# Fill in the blanks in the statements below:

library(ggrepel)

ggplot(data = stars, aes(log10(temp), magnitude, label = star)) +
  geom_point() +
  scale_y_reverse() +
  scale_x_reverse() +
  geom_text_repel()

# The least luminous star in the sample with a surface temperature over 5000K is...

# Antares

# Castor

# Mirfak

# Polaris

# van Maanen's Star # This one

# The two stars with the lowest temperature and highest luminosity are know as supergiants. The two
# supergiants in this dataset are...

# Rigel and Deneb

# *SiriusB and van Maanen's Star

# Alnitak and Alnilam

# Betelgeuse and Antares # Correct answer

# Wolf359 and G51-I5

# The Sun is a...

# Main sequence star # Yes

# Giant

# White dwarf

# Question 9

# Remove the text labels and color the points by star type. This classification describes the
# properties of the star's spectrum, the amount of light produced at various wavelengths.

ggplot(data = stars, aes(log10(temp), magnitude, color = type)) +
  geom_point() +
  scale_y_reverse() +
  scale_x_reverse()

# Which star type has the lowest temperature? #  M

# Which star type has the highest temperature? # O

# The sun is classified as a G-type star. Is the most luminous G-type star in this dataset
# also the hottest? # No

## FINAL ASSESSMENT PART 2: Climate Change

# The planet's surface temperature is increasing as greenhouse gas emissions increase, and
# this global warming and carbon cycle disruption is wreaking havoc on natural systems. Living
# systems that depend on current temperature, weather, currents and carbon balance are
# jeopardized, and human society will be forced to contend with widespread economic, social and
# political and environmental damage as the temperature continues to rise. In these exercises,
# we examine human carbon emissions using time series of actual atmospheric and ice core
# measurements from the National Oceanic and Atmospheric Administration (NOAA) and Carbon
# Dioxide Information Analysis Center (CDIAC).

# Libraries and data import

library(tidyverse)
library(dslabs)
data(temp_carbon)
data(greenhouse_gases)
data(historic_co2)

# Question 1

# Load the temp_carbon dataset from dslabs, which contains annual global temperature anomalies
# (difference from 20th century mean temperature in degrees Celsius), temperature anomalies over
# the land and ocean, and global carbon emissions (in metric tons). Note that the ranges differ
# for temperature and carbon emissions.

# Which of these code blocks returns the latest year for which carbon emissions are reported?

# 1

temp_carbon %>% .$year %>% max() # 2018

# 2

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% pull(year) %>% max() # 2014

# 3

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% max(year) # error, object year not found

# 4

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% .$year %>% max() # 2014

# 5

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% select(year) %>% max() # 2014

# 6

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% max(.$year) # NA

# Question 2

# Inspect the difference in carbon emissions in temp_carbon from the first available year
# to the last available year.

# What is the first year for which carbon emissions data are available?

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% .$year %>% min() #1751

# What is the last year for which carbon emissions data are available?

temp_carbon %>% filter(!is.na(carbon_emissions)) %>% pull(year) %>% max() # 2014

# How many times larger were carbon emissions in the last year relative to the first year?

temp_carbon %>% filter(!is.na(carbon_emissions) & year %in% c(1751, 2014)) %>%
  summarize(rate = max(carbon_emissions) / min(carbon_emissions)) %>% .$rate #3285

# Question 3

# Inspect the difference in temperature in temp_carbon from the first available year to the
# last available year.

# What is the first year for which global temperaature anomaly (temp_anomaly) data are
# available?

temp_carbon %>% filter(!is.na(temp_anomaly)) %>% .$year %>% min() # 1880

# What is the last year for which global temperature anomaly data are available?

temp_carbon %>% filter(!is.na(temp_anomaly)) %>% pull(year) %>% max() # 2018

# How many degrees Celsius has temperature increased over the range? Compare the
# temperatures in the most recent year versus the oldest year.

temp_carbon %>% filter(!is.na(temp_anomaly) & year %in% c(1880, 2018)) %>%
  summarize(increase = temp_anomaly[2] - temp_anomaly[1]) # .93

# Question 4

# Create a time series line plot of the temperature anomaly. Only include years where
# temperatures are reported. Save this plot to the object p.

p <- temp_carbon %>% filter(!is.na(temp_anomaly)) %>%
  ggplot(aes(year, temp_anomaly)) +
  geom_line()

# Which command adds a blue horizontal line indicating the 20th century mean temperature?

# 1

p + geom_vline(aes(xintercept = 0), col = "blue")

# 2

p + geom_hline(aes(y = 0), col = "blue")

# 3

p + geom_hline(aes(yintercept = 0, col = blue))

# 4

p <- p + geom_hline(aes(yintercept = 0), col = "blue") # Correct

# Question 5

# Continue working with p, the plot created in the previous question.

# Change the y-axis label to be "Temperature anomaly (degrees C)". Add a title, "Temperature
# anomaly relative to 20th century mean, 1880-2018". Also add a layer to the plot: the x-coord
# should be 2000, the y-coord should be 0.05, the text should be "2oth century mean", and the text
# color should be blue.

# Which of the following code blocks is correct?

# 1

p + ylab("Temperature anomaly (degrees C)") +
  title("Temperature anomaly relative to 20th century mean, 1880-2018") +
  geom_text(aes(x = 2000, y = 0.05, label = "20th century mean", col = "blue")) # error in title

# 2

p + ylim("Temperature anomaly (degrees C)") +
  ggtitle("Temperature anomaly relative to 20th century mean, 1880-2018") +
  geom_text(aes(x = 2000, y = 0.05, label = "20th century mean"), col = "blue") # wrong, ylim, not label

# 3

p + ylab("Temperature anomaly (degrees C)") +
  ggtitle("Temperature anomaly relative to 20th century mean, 1880-2018") +
  geom_text(aes(x = 2000, y = 0.05, label = "20th century mean"), col = "blue")) # wrong, extra )

# 4

p <- p + ylab("Temperature anomaly (degrees C)") +
  ggtitle("Temperature anomaly relative to 20th century mean, 1880-2018") +
  geom_text(aes(x = 2000, y = 0.05, label = "20th century mean"), col = "blue") # Correct

# 5

p + ylab("Temperature anomaly (degrees C)") +
  title("Temperature anomaly relative to 20th century mean, 1880-2018") +
  geom_text(aes(x = 2000, y = 0.05, label = "20th century mean"), col = "blue") # error in title

# Question 6

# Use the plot created in the last two exercises to answer the following questions. Answers within
# five years of the correct answer will be accepted.

# When was the earliest year with a temperature above the 20th century mean? # ~1939

# When was the last year with an average temperature below the 20th century mean? # ~1976

# In what year did the temperature anomaly exceed 0.5 degrees Celsius for the first time? # ~1999

# Question 7

# Add layers to the previous plot to include line graphs of the temperature anomaly in the ocean
# and on land. Assign different colors to the lines. Compare the global temperature anomaly to
# the land temperature anomaly and ocean temperature anomaly.

p + geom_line(aes(y = land_anomaly), col = "green") +
  geom_line(aes(y = ocean_anomaly), col = "red")

# Which region has the largest 2018 temperature anomaly relative to 20th century mean? # land

# Which region has the largest change in temperature since 1880? # land

# Which region has a temperature anomaly pattern that more closely matches the global pattern? # ocean

# CLIMATE CHANGE EXERCISES (CONT'D)

# Libraries and Data Import

library(tidyverse)
library(dslabs)
data("temp_carbon")
data("greenhouse_gases")
data("historic_co2")

# Question 8

# A major determinant of Earth's temperature is the greenhouse effect. Many gases trap heat and
# reflect it towards the surface, preventing heat from escaping the atmosphere. The greenhouse
# effect is vital in keeping Earth at a warm enough temperature to sustain liquid water and life;
# however, changes in greenhouse gas levels can alter the temperature balance of the planet.

# The greenhouse_gases data frame from dslabs contains concentrations of the three most significant
# greenhouse gases: carbon dioxide (abbreviated in the data as co2), methane (ch4 in the data),
# and nitrous oxide (n2o in the data). Measurements are provided every 20 years for the past 2000
# years.

# Complete the code outline below to make a line plot of concentration on the y-axis by year on
# the x-axis. Facet by gas, aligning the plots vertically so as to ease comparisons along the
# year axis. Add a vertical line with an x-intercept at the year 1850, noting the unoffcial start
# of the industrial revolution and widespread fossil fuel consumption. Note that the units for
# ch4 and n2o are ppb while the units for co2 are ppm.

greenhouse_gases %>%
  ggplot(aes(year, concentration)) + # aes() is first blank
  geom_line() +
  facet_grid(gas ~ ., scales = "free") + # second blank is the first argument in facet_grid()
  geom_vline(xintercept = 1850) + # third blank
  ylab("Concentration (ch4/n2o ppb, co2 ppm)") +
  ggtitle("Atmospheric greenhouse gas concentration by year, 0-2000")

# Question 9

# Interpret the plot of greenhouse gases over time from the previous question. You will use each answer
# exactly once: ch4, co2, n2o, all, none.

# Which gas was stable at approximately 275 ppm/ppb until around 1850? # co2

# Which gas more than doubled in concentration since 1850? # ch4

# Which gas decreased in contentration since 1850? # none

# Which gas had the smallest magnitude change since 1850? # n2o

# Which gas increased exponentially in concentration after 1850? # all

# Question 10

# While many aspects of climate are independent of human influence, and co2 levels can change without
# human intervention, climate models cannot reconstruct current conditions without incorporating
# the effect of manmade carbon emissions. These emissions consist of greenhouse gases and are mainly
# the result of burning fossil fuels such as oil, coal and natural gas.

# Make a time series line plot of carbon emissions (carbon_emissions) from the temp_carbon dataset.
# The y-axis is metric tons of carbon emitted per year.

temp_carbon %>%
  ggplot(aes(year, carbon_emissions)) +
  geom_line()

# Which of the following are true about the trend of carbon emissions?

# Carbon emissions were essentially zero before 1850 and have increased exponentially since then. # True
# Carbon emissions are reaching a stable level.
# Carbon emissions have increased every year on record.
# Carbon emissions in 2014 were about 4 times as large as 1960 emissions. # True
# Carbon emissions have doubled since the late 1970s. # True
# Carbon emissions change with the same trend as atmospheric greenhouse gas levels. # True

# Question 11

# We saw how greenhouse gases have changed over the course of human history, but how has co2 varied
# over a longer time scale? The historic_co2 data frame in dslabs contains direct measurements of
# atmospheric co2 from Mauna Loa since 1959 as well as indirect measurements of atmospheric co2 from
# ice cores dating back to 800,000 years.

# Make a line plot of co2 concentration over time (year), coloring by the measurement source (source).
# SAve this plot as co2_time for later use.

co2_time <- historic_co2 %>%
  ggplot(aes(year, co2, group = source, color = source)) +
  geom_line()

# Which of the following are tre about co2_time, the time series of co2 over the last 800,000 years?

# Modern co2 levels are higher than at any point in the last 800,000 years # True
# There are natural cycles of co2 increase and decrease lasting 50k-100k years per cycle. # True
# In most cases, it appears to take longer for co2 levels to decrease than to increase. # True
# co2 concentration has been at least 200 ppm for the last 800,000 years. 

# Question 12

# One way to differentiate natural co2 oscillations from today's manmade co2 spike is by examining
# the rate of change of co2. The planet is affected not only by the absolute concentration of co2
# but also by its rate of change. When the rate of change is slow, living and nonliving systems
# have time to adapt to new temperature and gas levels, but when the rate of change is fast, abrupt
# differences can overwhelm natural systems. How does the pace of natural co2 change differ from
# the current rate of change?

# Use the co2_time plot saved above. Change the limits as directed to investigate the rate of change
# in co2 over varios periods with spikes in co2 concentration.

# Change the x-axis limits to -800,000 and -775,000. About how many years did it take for co2 to
# rise from 200 ppmv to its peak near 275ppmv?

co2_time +
  xlim(-800000, -775000)

# 100
# 3,000
# 6,000
# 10,000 # This is the correct answer

# Change the x-axis limits to -375,000 and -330,000. About how many years did it take for co2
# to rise from the minimum of 180 ppm to its peak of 300 ppm?

co2_time +
  xlim(-375000, -330000)

# 3,000
# 6,000
# 12,000
# 25,000 # This is right

# Change the x-axis limits to -140,000 and -120,000. About how many years did it take for co2
# to rise from 200 ppmv to its peak near 280 ppmv?

co2_time +
  xlim(-140000, -120000)

# 200
# 1,500
# 5,000
# 9,000 # This one

# Change the x-axis limits to -3,000 and 2018 to investigate modern changes in co2. About how
# many years did it take for co2 to rise from its stable level around 275 ppmv to the current
# level of over 400 ppmv?

co2_time +
  xlim(-3000, 2018)

# 250 # This one
# 1,000
# 2,000
# 5,000

