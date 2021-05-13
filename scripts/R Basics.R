### DATA SCIENCE: R BASICS

## SECTION 1. R BASICS, FUNCTIONS, AND DATA TYPES

# 1.1. MOTIVATION AND GETTING STARTED

# Running commands while editing scripts

library(tidyverse)
library(dslabs)
data("murders")
murders %>%
  ggplot(aes(population, total, label=abb, color=region)) +
  geom_label()

# 1.2. R BASICS

# DataCamp Assessment 1

# Using Variables 1

# What is the sum of the first 'n' positive integers?
# We can use the formula n(n + 1) / 2 to quickly compute this quantity.

# Define n <- 100
# Then use R to compute the sum of 1 through 100 using the formula
# n(n + 1) / 2. What is the sum?

# Here is how you compute the sum for the first 20 integers.
20*(20+1)/2

# However, we can define a variable to use the formula for other values of n
n <- 20
n*(n+1)/2

n <- 25
n*(n+1)/2

# Below, write code to calculate the sum of the first 100 integers
n <- 100
n*(n+1)/2

# Using variables 2

# What is the sum of the first 1000 positive integers?

# We can use the formula n(n + 1) / 2 to quickly compute this quantity

# Use the same formula as the last exercise but change the value of n. 

# Below, write code to calculate the sum of the first 1000 integers
n <- 1000
n*(n+1)/2

# Functions

# Run the following code
n <- 1000
x <- seq(1, n)
sum(x)

# Based on the result, what do you think the functions 'seq' and 'sum' do?

# Answer: 'seq' creates a list of numbers and 'sum' adds them up

# Nested function calls 1

# In math and programming we say we evaluate a function when we replace
# arguments with specific values. So if we type log2(16) we evaluate
# the log2 function to get the log base 2 of 16 which is 4.

# In R it is often useful to evaluate a function inside another function.
# For example, sqrt(log2(16)) will calculate the log to the base 2 of 16
# and then compute the square root of that value. So the first evaluation
# gives a 4 and this gets evaluated by sqrt to give the final answer of 2.

# Use one line of code to compute the log, to the base 10, of the square
# root of 100. Make sure your code includes the log10 and sqrt functions.

# log to the base 2
log2(16)

# sqrt of the log to the base 2 of 16
sqrt(log2(16))

# Compute log to the base 10 (log10) of the sqrt of 100.
sqrt(log10(100))

# Nested functions call 2

# Which of the following will always return the numeric value stored in x?

# Answer: log(exp(x))

# 1.3. DATA TYPES

# DataCamp Assessment 2

# str

# Use th function str to examine the structure of the murders object.
# We can see that this object is a data frame with 51 rows and five columns.
# Which of the following best describes the variables represented in
# this data frame?
str(murders)

# Answer: The state name, the abbreviation, the state's region, the state's
# population and the total number of murders.

# Variable names

# The function names() is specifically designed to extract the column names
# from a data frame.

# What are the column names used by the murders data frame?
names(murders)

# Examining variables

# In this module we have learned that every variable has a class. For example,
# the class can be a `character`, `numeric` or `logical`. The function class()
# can be used to determine the class of an object.

# Here we are going to determine the class of one of the variables in the murders
# data frame. To extract variables from a data frame we use `$`, referred to as
# the accessor.

# To access the population variable from the murders dataset we use this code:
p <- murders$population

# To determine the class of object `p` we use this code:
class(p)

# Use the accessor to extract state abbreviations and assign it to `a`:
a <- murders$abb

# Determine the class of `a`:
class(a)

# Multiple ways to access variables

# There are multiple ways to access variables in a data frame. For example,
# we can use the square brackets [[ instead of the accessor.
# If you instead try to access a column with just one bracket,
murders["population"]
# R returns a subset of the original data frame containing just this column.
# This new object will be of class data.frame rather than a vector. To access
# the column itself you need to use either the accessor $ or the double square
# brackets [[.

# Use the square brackets [[ to extract the state abbreviations and assign
# them to the object b. Then use the identical function to determine if a,
# as defined in the previous exercises, and b are the same.

# We extract the population like this:
p <- murders$population

# This is how we do the same with the square brackets:
o <- murders[["population"]]

# We can confirm these two are the same
identical(p, o)

# Use square brackets to extract `abb` from `murders` and assign it to b
b <- murders[["abb"]]

# Check if `a` and `b` are identical
identical(a, b)

# Factors

# Using the str() command, we saw that the region column stores a factor.
# You can corroborate this by using the class() command on the region column.

# The function levels shows us the categories for the factor.

# For this question we will combine two functions, in a nested way, to
# determine the number of unique categories. Use the functions levels and
# length to determine the number of regions defined by this dataset and
# contained in murders$region.

# We can see the class of the region variable using class
class(murders$region)

# Determine the number of regions included in this variable
length(levels(murders$region))

# Tables

# The function table takes a vector as input and returns the frequency
# of each unique element in the vector.

# We will use the table function to answer this question. Use the table
# function in one line of code to create a table showing the number of
# states per region.

# The `c` in `c()` is short for `concatenate`, which is the action of
# connecting items into a chain. The function `c()` connects all of the
# strings within it into a single vector, which we can assign to `x`
x <- c("a", "a", "b", "b", "c")

# Here is an example of what the table function does
table(x)

# Write one line of code to show the number of states per region
table(murders$region)

# SECTION 1 ASSESSMENT

# Question 1

# To find the solutions to an equation of the format ax^2 + bx + c, use
# the quadratic equation x = -b +|- sqrt(b^2 - 4ac) / 2a.
# What are the two solutions to 2x^2 - x - 4 = 0?
a <- 2
b <- -1
c <- -4
(-b+sqrt(b^2-4*a*c))/(2*a) # 1.686
(-b-sqrt(b^2-4*a*c))/(2*a) # -1.186

# Question 2

# Use R to compute log base 4 of 1024.
log(1024, 4)

# Question 3

# Load the movielens dataset
data("movielens")

# Question 3a

# Begin your exploration of this dataset by looking at the structure
str(movielens)

# How many rows are in the dataset: 100004

# Question 3b

# How many different variables are in the dataset: 7

# Question 3c

# What is the variable type of title?
class(movielens$title) # character

# Question 3d

# What is the variable type of genres?
class(movielens$genres) # factor

# Question 4

# We already know we can use the levels() function to determine the levels
# of a factor. A different function, nlevels(), may be used to determine
# the number of levels of a factor.

# Use this function to determine how many levels are in the factor genres
# in the movielens data frame.
nlevels(movielens$genres) # 901

## SECTION 2. VECTORS, SORTING

# 2.1. VECTORS

# DataCamp Assessment 2.1. Vectors

# Numeric vectors

# A vector is a series of vales, all of the same type. They are the most
# basic data type in R and can hold numeric data, character data, or logical data.
# In R, you can create a vector with the concatenate (or combine) function c().
# You place the vector elements separated by a comma between the parentheses.
# For example a numeric vector would look something like this:

cost <- c(50, 75, 90, 100, 150)

# Use the function c() to create a numberic vector with the average temperatures
# in January for Beijing, Lagos, Paris, Rio de Janeiro, San Juan, and Toronto. The
# average high temperatures are 35, 88, 42, 84, 81, and 30 degrees Fahrenheit. Call
# the object temp.

temp <- c(35, 88, 42, 84, 81, 30)

# Character vectors

# As in the previous question, we are going to create a vector. Only this time, 
# we learn to create character vectors. The main difference is that these have to
# be written as strings and so the names are enclosed within double quotes.

# A character vector would look something like this:

food <- c("pizza", "burgers", "salads", "cheese", "pasta")

# The temperatures we stored in temp are from the cities Beijing, Lagos, Paris, 
# Rio de Janeiro, San Juan, and Toronto.
# Create a vector with these city names and call the object city. Make sure to use
# the correct order, spelling and capitalization.

city <- c("Beijing", "Lagos", "Paris", "Rio de Janeiro", "San Juan", "Toronto")

# Connecting numeric and character vectors

# We have successfully assigned the temperatures as numeric values to temp and
# the city names as character values to city. But can we associate the temperature
# to its related city? Yes! We can do so using a codee we already know - names.
# We assign names to the numeric values.

# It would look like this:

cost <- c(50, 75, 90, 100, 150)
food <- c("pizza", "burgers", "salads", "cheese", "pasta")
names(cost) <- food

# Use the names function and the objects defined in the previous exercises to
# associate the temperature data with its corresponding city. Note: to see what
# happened, after assigning the city names to the temp vector, try printing the temp
# vector to understand how the names are associated with elements of temp.

names(temp) <- city
temp

# Subsetting vectors

# If we want to display only selected values from the object, R can help us
# do that easily. For example, if we want to see the cost of the last 3 items
# in our food list, we would type:

cost[3:5]

# Note here that we could also type cost[c(3,4,5)] and get the same result.
# The : operator helps us condense the code and get consecutive values.

# We will learn to subset using several special operators. Use the [ and :
# operators to access the temperature of the first three cities in the list, 
# which are already stored in temp.

temp[1:3]

# Subsetting vectors (cont'd)

# In the previous question, we accessed the temperature for consecutive cities
# (first three). But what if we want to access the temperatures for any 2 specific
# cities?

# An example: To access the cost of pizza (1st) and pasta (5th) in our food list, 
# the code would be:

cost[c(1,5)]

# This time we will access our object using just the [ operator. Use the [ operator
# to access the temperature of Paris and San Juan already stored in the temp object.

temp[c("Paris", "San Juan")]

# Sequences

# The : operator helps us create sequences of numbers. For example, 32:99 would
# create a list of numbers from 32 to 99. Then, if we want to know the length of
# this sequence, all we need to do is use the length command.

# This time we will use just the : operator. Use the : operator to create a 
# sequence of consecutive integers starting at 12 and ending at 73 and save it in 
# an object x, then determine the length of object x.

x <- 12:73
length(x)

# Sequences (cont'd)

# We can also create different types of sequences in R. For example, in 
# seq(7, 49, 7), the first argument defines the start, and the second the end. The
# default is to go up in increments of 1, but a third argument lets us tell it by
# what interval.

# We will learn how to use the seq() function in this question. Create a vector
# containing all the positive odd numbers smaller than 100. The numbers should be
# in ascending order.

seq(1,100,2)

# Sequences and length

# The second argument of the function seq is actually a maximum, not necessarily the end.
# So, if we type

seq(7, 50, 7)

# we actually get the same vector of integers as if we type

seq(7, 49, 7)

# This can be useful because sometimes all we want are sequential numbers that are 
# smaller than some value. Let's look at an example.

# Create a vector of numbers that starts at 6, does not go beyond 55, and adds numbers
# in increments of 4/7. So the first three numbers will be 6, 6+4/7, and 6+8/7. How 
# many numbers does the list have? Use only one line of code to answer both questions.

length(seq(6,55,4/7))

# Sequences of certain length

# The seq() function has another useful argument. The argument length.out. This 
# argument lets us generate sequences that are increasing by the same amount but
# are of the prespecified length. For example, this line of code:

x <- seq(0, 100, length.out = 5)

# produces the numbers 0, 25, 50, 75, 100. Let's create a vector and see what is
# the class of the object produced.

# Determine the class of a vector generated with seq() uusing the length.out
# argument. Specifically, what is the class of the following object?

a <- seq(1, 10, length.out = 100)
class(a)

# Integers

# We have discussed the numeric class. We just saw that the seq function can
# generate objects of this class. For another example, type

class(seq(1, 10, 0.5))

# and note that the class is numeric. R has another type of vector we have not
# described, the integer class. You can create an integer by adding the letter L
# after a whole number. If you type

class(3L)

# you see this is an integer and not a numeric. For most practical purposes, integers
# and numerics are indistinguishable. For example 3, the integer, minus 3 the numeric is
# 0. To see this type this:

3L - 3

# The main difference is that integers occupy less space in the computer memory, so for
# big computations using integers can have a substantial impact.

# Does this change depending on what we store in an object? What is the class of the
# following object?

a <- seq(1, 10)
class(a)

# Integers and numerics

# Let's confirm that 1L is an integer not a numeric.

# Confirm that the class of 1 is numeric and the class of 1L is integer.

class(1)
class(1L)

# Coercion

# The concept of coercion is a very important one. Watching the video, we learned
# that when an entry does not match what an R function is expecting, R tries to 
# guess what we meant before throwing an error. This might get confusing at times.

# As we've discussed in earlier questions, there are numeric and character vectors.
# The character vectors are placed in quotes and the numerics are not.

# We can avoid issues with coercion in R by changing characters to numerics and 
# vice-versa. This is know as typecasting. The code is as.numeric(x) and it helps
# us convert character strings to numbers. There is an equivalent function that
# converts its argument to a string as.character(x).

# Define the following vector:

x <- c(1, 3, 5, "a")

# Notice that x has been coerced into a character string.

class(x)

# If we assign something new to a vector that we've already created, the new
# definition replaces whatever was previously assigned to the vector. Convert x
# to a vector of numbers using the as.numeric() function. Note the warning message.

x <- as.numeric(x)

# 2.2. SORTING

# DataCamp Assessment 2.2. Sorting

# Sort

library(dslabs)
data(murders)

# When looking at a dataset, we may want to sort the data in order that makes more
# sense for analysis. Let's learn to do this using the murders dataset as an example.

# Use the $ operator to access the population size data and store it in the object pop.

pop <- murders$population

# Then use the sort function to redefine pop so that it is sorted

pop <- sort(pop)

# Finally use the [ operator to report the smallest population size

pop[1]

# Access the state variable and store it in an object

states <- murders$state

# Sort the object alphabetically and redefine the object

states <- sort(states)

# Report the first alphabetical value

states[1]

# Order

# The function order() returns the index vector needed to sort the vector. This
# implies that sort(x) and x[order(x)] give the same result.

# This can be useful for finding row numbers with certain properties such as
# "the row for the state with the smallest population". Remember that when we extract
# a variable from a data frame the order of the resulting vector is the same as the
# order of the rows of the data frame. So for example, the entries of the vector
# murders$state are ordered in the same way as the states if you go down the rows of
# murders.

# Now instead of the smallest population size, let's find out the row number, in the data
# frame murders, of the state with the smallest population size.

# This time we need to use order() instead of sort(). Remember that the entries in the
# vector murders$population follow the order of the rows of murders.

# Access population from the dataset and store it in pop

pop <- murders$population

# Use the command order to find the vector of indexes that order pop and store it
# in an object ord

ord <- order(pop)

# Find the index number of the entry with the smallest population size

ord[1]

# New codes

# We can actually perform the same operation as in the previous exercise using the
# function which.min. It basically tells us which is the minimum value.

# Write one line of code that gives the index of the lowest population entry.

which.min(murders$population)

# Using the output of order

# Now we know how small the smallest state is and we know which row represents it.
# However, which state is it?

# Find the index of the smallest state using 

i <- which.min(murders$population)

# Define a variable states to hold the states names from the murders data frame

states <- murders$state

# Use the index you just defined to find the state with the smallest population

states[i]

# Ranks

# You can create a data frame using the data.frame function. Here's a quick example

temp <- c(35, 88, 42, 84, 81, 30)
city <- c("Beijing", "Lagos", "Paris", "Rio de Janeiro", "San Juan", "Toronto")
city_temps <- data.frame(name = city, temperature = temp)

# Define a variable states to be the state names from murders

states <- murders$state

# Use rank(murders$population) to determine the population size rank (from smallest
# to biggest) of each state. Save these ranks in an object called ranks.

ranks <- rank(murders$population)

# Create a data frame with the state names and their respective ranks. Call the 
# data frame my_df

my_df <- data.frame(state = states, rank = ranks)

# Data frames, ranks and orders

# This exercise is somewhat more challenging. We are going to repeat the previous exercise
# but this time order my_df so that the states are ordered from least populous to most.

# Create variables states and ranks to store the state names and ranks by population size
# respectively.

states <- murders$state
ranks <- rank(murders$population)

# Create an object ind that stores the indexes needed to order the population values,
# using the order command. For example, we could define o <- order(murders$population)

ind <- order(murders$population)

# Create a data frame with both variables following the correct order. Use the bracket [ 
# operator to re-order each column in the data frame. For example, states[o] orders the 
# abbreviations based by population size. The columns of the data frame must be in the 
# specific order and have the specific names: states, ranks

my_df <- data.frame(states = states[ind], ranks = ranks[ind])

# NAs

# The na_example dataset represents a series of counts. It is included in the dslabs
# package. You can quickly examine the object using

library(dslabs)
data(na_example)
str(na_example)

# However, when we compute the average we obtain an NA. You can see this by typing:

mean(na_example)

# The is.na returns a logical vector that tells us which entries are NA. Assign the logical
# vector that is returned by is.na(na_example) to an object called ind.

ind <- is.na(na_example)

# Determine how many NAs na_example has, using the sum command.

sum(ind)

# Removing NAs

# We previously computed the average of na_example using mean(na_example) and get NA.
# This is because the function mean returns NA if it encounters at least one NA. A common
# operation is therefore removing the entries that are NA and after that perform operations
# on the rest.

# Write one line of code to compute the average, but only for the entries that are not NA
# making use of the ! operator before ind. 

# Note what we can do with the ! operator

x <- c(1, 2, 3)
ind <- c(FALSE, FALSE, FALSE)
x[!ind]

# Create the ind vector

ind <- is.na(na_example)

# We saw that this gives an NA

mean(na_example)

# Compute the average, for entries of na_example that are not NA

mean(na_example[!ind])

# DataCamp Assessment 2.3. Vector Arithmetic

# Vectorized operations

# Previously we created this data frame: 

temp <- c(35, 88, 42, 84, 81, 30)
city <- c("Beijing", "Lagos", "Paris", "Rio de Janeiro", "San Juan", "Toronto")
city_temps <- data.frame(name = city, temperature = temp)

# We are interested in specifying Celsius instead of Fahrenheit. Remember that to
# convert from Fahrenheit to Celsius we use C = 5/9 * (F - 32)

# Use vector arithmetic to convert temp to Celsius.

temp <- 5/9 * (temp - 32)

# Create a data frame called city_temps with the city names and temperatures
# in Celsius.

city_temps <- data.frame(name = city, temperatures = temp)

# Vectorized operations (cont'd)

# We can use some of what we have learned to perform calculations that would 
# otherwise be quite complicated. Let's see an example.

# What is the sum of the following equation: 1 + 1/2^2 + ... + 1/100^2? Thanks to
# Euler we know it should be close to pi^2/6.

# Define an object x that contains the numbers 1 through 100.

x <- 1:100

# Compute the sum 1 + 1/2^2 + 1/3^2 + ... + 1/100^2.

sum(1/x^2)

# Vectorized operations (cont'd)

# compute the per 100,000 murder rate for each state and store it in the object
# murder_rate.

murder_rate <- murders$total / murders$population * 100000

# Then copute the average of the state murder rates for the US using the funtion
# mean. What is this average?

mean(murder_rate)

# SECTION 2 ASSESSMENT

# Question 1

# Consider the vector x:

x <- c(2, 43, 27, 96, 18)

# Match the following outputs to the function which produces that output. Options
# include sort(x), order(x), rank(x), and none of these.

# a. 1, 2, 3, 4, 5 # none of these

# b. 1, 5, 3, 2, 4
order(x)
# c. 1, 4, 3, 5, 2
rank(x)
# d. 2, 18, 27, 43, 96
sort(x)

# Question 2

# Match the following functions to their output. Options include integers 1 through 5
# and none of these.

# a. min(x) = 2
min(x)
# b. which.min(x) = 1
which.min(x)
# c. max(x) = 96, i.e., none of these
max(x)
# d. which.max(x) = 4
which.max(x)

# Question 3

# Mandi, Amy, Nicole, and Olivia all ran different distances in different time
# intervals. Their distances (in miles) and times (in minutes) are as follows:

name <- c("Mandi", "Amy", "Nicole", "Olivia")
distance <- c(0.8, 3.1, 2.8, 4.0)
time <- c(10, 30, 40, 50)

# Write a line of code to convert time to hours. Remember there are 60 minutes in
# an hour. Then write a line of code to calculate the speed of each runner in miles
# per hour. Speed is distance divided by time.

time <- time / 60
speed <- distance / time

# How many hours did Olivia run?

time[4] # = 0.833

# What was Mandi's speed in miles per hour?

speed[1] # = 4.8

# Which runner had the fastest speed?

rank(speed) # = Amy; index = 2

## SECTION 3. INDEXING, DATA WRANGLING, PLOTS

# 3.1. INDEXING

# Logical vectors

# Here we will be using logical operators to create a logical vector

# Compute the per 100,000 murder rate for each state and store it in an object
# called murder_rate. Then use the logical operators to create a logical vector,
# name it low, that tells us which entries of murder_rate are lower than 1, and 
# which are not, in one line of code.

library(dslabs)
data(murders)

# Store the murder rate per 100,000 for each state, in `murder_rate`

murder_rate <- murders$total / murders$population * 100000

# Store the murder_rate < 1 in `low`

low <- murder_rate < 1

# which

# The function which() helps us know directly, which values are low or high, etc. 
# Let's use it in this question.

# Use the results from the previous exercise and the function which to determine
# the indices of murder_rate associated with values lower than 1.

# Get the indices of entries that are below 1

which(low)

# Ordering vectors

# Note that if we want to know which entries of a vector are lower than a particular
# value we can use code like this

small <- murders$population < 1000000
murders$state[small]

# Use the results from the previous exercise to report the names of the states
# with murder rates lower than 1, using the square brackets to retrieve the
# names of the states from the dataset.

# Names of states with murder rates lower than 1.

murders$state[low]

# Filtering

# Now we will extend the code from the previous exercises to report the states in
# the Northeast with a murder rate lower than 1.

# Define low as before. Use the & operator to create a new object ind that is true
# when low is true and the state is in the Northeast. Use the brackets [ and ind to
# show the state names that satisfy this condition.

# Store the murder rate per 100,000 for each state in `murder_rate`

murder_rate <- murders$total / murders$population * 100000

# Store the `murder_rate < 1` in `low`

low <- murder_rate < 1

# Create a vector ind for states in the Northeast and with murder rates lower than 1

ind <- low & murders$region == "Northeast"

# Names of states in `ind`

murders$state[ind]

# Filtering (cont'd)

# In a previous exercise we computed the murder rate for each state and the average of
# these numbers. How many states are below the average?

# Store the murder rate per 100,000 for each state in murder_rate

murder_rate <- murders$total / murders$population * 100000

# Compute the average murder rate using `mean` and store it in an object named `avg`

avg <- mean(murder_rate)

# How many states have murder rates below avg? Check using sum.

sum(murder_rate < avg)

# Match

# In this exercise we use the match function to identify the states with abbreviations
# AK, MI, and IA.

# Define a character vector with the abbreviations.
# Start by defining an index of the entries of murders$abb that match the three
# abbreviations.
# Use the [ operator to extract the states.

# Store the three abbreviations in a vector called `abbs` (remember that they
# are character vectors and need quotes)

abbs <- c("AK", "MI", "IA")

# Match the abbs to the murders$abb and store in ind

ind <- match(abbs, murders$abb)

# Print the state names from ind

murders$state[ind]

# %in%

# If rather than an index we want a logical that tells us whether or not each element
# of a first vector is in a second, we can use the function %in%. For example:

x <- c(2, 3, 5)
y <- c(1, 2, 3, 4)
x %in% y

# Gives us two TRUE followed by a FALSE because 2 and 3 are in y, but 5 is not.

# Which of the following are actual abbreviations: MA, ME, MI, MO, MU?

# Define a character vector with the abbreviations MA, ME, MI, MO, MU.

# Use the %in% operator to create a logical vector that is TRUE when the abbreviation
# is in murders$abb.

# Store the 5 abbreviations in `abbs`

abbs <- c("MA", "ME", "MI", "MO", "MU")

# Use the %in% command to check if the entries of abbs are abbreviations in the
# murders data frame

abbs %in% murders$abb

# Logical operator

# We are again working with the characters

abbs <- c("MA", "ME", "MI", "MO", "MU")

# In a previous exercise we computed the index abbs %in% murders$abb. Based on that, 
# and using the which function and the ! operator, get the index of the entries of abbs
# that are NOT abbreviations.

# Show the entries of abbs that are not actual abbreviations.

# Use the which command and ! operator to find out which index abbreviations are not
# actually part of the dataset and store it in `ind`

ind <- which(!abbs %in% murders$abb)

# Names of the abbreviations in `ind`

abbs[ind]

# 3.2. BASIC DATA WRANGLING

# dplyr

# Load the dplyr package and the murders dataset

library(dplyr)
library(dslabs)
data(murders)

# You can add columns using the dplyr function mutate. This function is aware
# of the column names and inside the function you can call them unquoted. Like this:

murders <- mutate(murders, population_in_millions = population / 10^6)

# Note that we can write population rather than murders$population. The function
# mutate knows we are grabbing columns from murders.

# Use the function mutate to add a murders column named rate with the 100,000 murder rate.
# Make sure you redefine murders as done in the example code above.

# Remember the murder rate is defined as the total murders divided by the population
# size times 100,000.

# Redefine murders so that it includes a column named rate with the per 100,000 murder
# rates.

murders <- mutate(murders, rate = total / population * 100000)

# mutate

# Note than if rank(x) gives you the ranks of x from lowest to highest, rank(-x)
# gives you the ranks from highest to lowest.

# Use the function mutate to add a column rank containing the rank, from highest
# to lowest murder rate. Make sure you redefine murders.

# Note that if you want the ranks from highest to lowest you can take the 
# negative and then compute the ranks.

x <- c(88, 100, 83, 92, 94)
rank(-x)

# Defining rate

rate <- murders$total / murders$population * 100000

# Redefine murders to include a column named rank with the ranks of rate
# from highest to lowest.

murders <- mutate(murders, rank = rank(-rate))

# select

# With dplyr we can use select to show only certain columns. For example with
# this code we would only show the states and population sizes.

select(murders, state, population)

# Use select to show the state names and abbreviations in murders. Just show it,
# do not define a new object.

# Use select to only show state names and abbreviations from murders.

select(murders, state, abb)

# filter

# The dplyr function filter is used to choose specific rows of the data frame to keep. 
# Unlike select which is for columns, filter is for rows. For example you can
# show just the New York row like this.

filter(murders, state == "New York")

# You can use other logical vectors to filter rows.

# Use filter to show the top 5 states with the highest murder rates. After we add
# murder rate and rank, do not change the murders dataset, just show the result.
# Note that you can filter based on the rank column.

# Filter to show the top 5 states with the highest murder rates.

filter(murders, rank <= 5)

# filter with !=

# We can remove rows using the != operator. For example to remove Florida we would
# do this:

no_florida <- filter(murders, state != "Florida")

# Create a new data frame called no_south that removes states from the South region.
# How many states are in this category? You can use the function nrow for this.

# Use filter to create a new data frame no_south.

no_south <- filter(murders, region != "South")

# Use nrow() to calculate the number of rows.

nrow(no_south)

# filter with %in%

# We can also use the %in% to filter with dplyr. For example you can see the data
# from New York and Texas like this:

filter(murders, state %in% c("New York", "Texas"))

# Create a new data frame called murders_nw with only the states from the Northeast
# and the West.
# How many states are in this category?

# Create a new data frame called murders_nw with only the states from the Northeast
# and the West.

murders_nw <- filter(murders, region %in% c("Northeast", "West"))

# Number of states (rows) in this category.

nrow(murders_nw)

# filtering by two conditions

# Suppose you want to live in the Northeast or West and want the murder rate to be less
# than 1. We want to see the data for the states satisfying these options. Note that
# you can use logical operators with filter.

filter(murders, population < 5000000 & region == "Northeast")

# Add a murder rate column and a rank column as done before.
# Create a table, call it my_states, that satisfies both conditions: it is in the 
# Northeast or West and the murder rate is less than 1.

# Use the select function to show only the state name, the rate and the rank.

# Create a table, call it my_states, that satisfies both conditions.

my_states <- filter(murders, region %in% c("Northeast", "West") & rate < 1)

# Use select to show only the state name, the murder rate, and the rank.

select(my_states, state, rate, rank)

# using the pipe %>%

# The pipe %>% can be used to perform operations sequentially without having to
# define intermediate objects. After redefining murder to include rate and rank
# in the solution to the previous exercise we did the following:

# Created a table
my_states <- filter(murders, region %in% c("Northeast", "West") & rate < 1)

# Used select to show only the state name, the murder rate, and the rank.
select(my_states, state, rate, rank)

# The pipe %>% permits us to perform both operations sequentially and without
# having to define an intermediate variable my_states

# For example we could have mutated and selected in the same line like this:

mutate(murders, rate = total / population * 100000, rank = rank(-rate)) %>%
  select(state, rate, rank)

# Note that select no longer has a data frame as the first argument. The first
# argument is assumed to be the result of the operation conducted right before
# the %>%

# Repeat the previous exercise, but now instead of creating a new object, show
# the result and only include the state, rate, and rank columns in that order.

# Use a pipe %>% to do this in just one line.

# Show the result and only include the state, rate, and rank columns, all in
# one line, in that order

filter(murders, region %in% c("Northeast", "West") & rate < 1) %>% select(state, rate, rank)

# mutate, filter, and select

# Now we will reset murders to the original table by using:

data(murders)

# Use one line of code to create a new data frame, called my_states, that has 
# murder rate and rank columns (with the rank ordered from highest to lowest),
# considers only states in the Northeast or West which have a murder rate lower
# than 1, and contain only the state, rate, and rank columns. The line should have
# four components separated by three %>% operators:

# The original dataset murders
# A call to mutate to add the murder rate and the rank
# A call to filter to keep only the states from the Northeast or West and that
# have a murder rate below 1.
# A call to select that keeps only the columns with the state name, the murder
# rate, and the rank.

# The line should look something like this my_states <- murders %>% mutate something
# %>% filter something %>% select something. Columns in the final data frame MUST
# be in the order: state, rate, rank.

# Create a new data frame called my_states.

my_states <- murders %>% mutate(rate = total / population * 100000, rank = rank(-rate)) %>%
  filter(region %in% c("Northeast", "West") & rate < 1) %>%
  select(state, rate, rank)

# 3.3. BASIC PLOTS

# Scatterplots

# We made a plot of total murders versus population and noted a strong relationship:
# not surprisingly, states with larger populations had more murders. You can run
# the code in console to get the plot.

library(dslabs)
data(murders)

population_in_millions <- murders$population / 10 ^ 6
total_gun_murders <- murders$total

plot(population_in_millions, total_gun_murders)

# Note that many states have populations below 5 million and are bunched up in the plot.
# We may gain further insights from making this plot in the log scale.

# Transform the variables using the log base 10 transformation. Plot the log transformed
# total murders versus population.

# Transform population (not population_in_millions) using the log10 transformation and
# save the object to log10_population

log10_population <- log10(murders$population)

# Transform total gun murders using log10 transformation and save the object as
# log10_total_gun_murders

log10_total_gun_murders <- log10(murders$total)

# Create a scatterplot with the long scale transformed population and murders

plot(log10_population, log10_total_gun_murders)

# Histograms

# Now we are going to make a histogram.

# Compute the population in millions and save it to the object population_in_millions
# Create a histogram of the state populations using the function hist

# Store the population in millions and save it to population_in_millions.

population_in_millions <- murders$population/10^6

# Create a histogram of this variable

hist(population_in_millions)

# Boxplots

# Now we are going to make boxplots. Boxplots are useful when we want a summary of
# several variables or several data strata of the same variables. Making too many
# histograms can become too cumbersome.

# In one line of code...

# Stratify the state populations by region.
# Generate boxplots for the strata.

# Note that you can achieve this using this population~region inside boxplot to 
# generate the strata and specify the dataset with the data argument.

# Create a boxplot of state populations by region for the murders dataset.

boxplot(population~region, murders)

# SECTION 3 ASSESSMENT

# For questions 1-8, load the dslabs dataset heights:

library(dslabs)
data(heights)
options(digits = 3) # report 3 significant digits for all answers

# Question 1

# First, determine the average height in this dataset. Then create a logical vector ind
# with the indices for those individuals who are above average height.

# How many individuals are above average height? 532

ind <- heights$height > mean(heights$height)
sum(ind)

# Question 2

# How many individuals in the dataset are above average height and are female? 31

sum(heights$height > mean(heights$height) & heights$sex == "Female")

# Question 3

# If you use mean() on a logical vector, it returns the proportion of observations
# that are TRUE. 

# What proportion of individuals in the dataset are female? 0.227

mean(heights$sex == "Female")

# Question 4

# This question takes you through three steps to determine the sex of the individual
# with the minimum height.

# Question 4a

# Determine the minimum height in the heights dataset. 50

min(heights$height)

# Question 4b

# Use the match() function to determine the index of the first individual with the
# minimum height. 1032

match(min(heights$height), heights$height)

# Question 4c

# Subset the sex column of the dataset by the index in 4b to determine the individual's
# sex. Male

heights$sex[match(min(heights$height), heights$height)]

# Question 5

# This question takes you three steps to determine how many of the integer height values
# between the minimum and maximum heights are not actual heights of individuals in the
# heights dataset.

# Question 5a

# Determine the maximum height. 82.7

max(heights$height)

# Question 5b

# Which integer values are between the maximum and minimum heights? For example, if the
# minimum height is 10.2 and the maximum height is 20.8, your answer should be x <- 11:20
# to capture the integers in between those values. (If either the maximum or minimum height
# are integers, include those values too.)

# Write code to create a vector x that includes the integers between the minimum and 
# maximum heights (as numbers.) There are multiple ways to solve this problem, but the grader
# expects you to use the format in the problem description. Your answer should have numbers
# and a colon ( : ), and it should not use other functions.

x <- 50:82

# Question 5c

# How many of the integers in x are NOT heights in the dataset? 3
# Use the sum() and %in% functions in addition to the ! operator.

sum(!x %in% heights$height)

# Question 6

# Using the heights dataset, create a new column of heights in centimeters named ht_cm.
# Recall that 1 inch = 2.54 cm. Save the resulting dataset as heights2.

library(dplyr)
heights2 <- mutate(heights, ht_cm = heights$height * 2.54)

# Question 6a

# What is the height in centimeters of the 18th individual (index = 18)? 163

heights2$ht_cm[18]

# Question 6b

# What is the mean height in centimeters? 174

mean(heights2$ht_cm)

# Question 7

# Create a data frame females by filtering the heights2 data to contain only female
# individuals.

females <- filter(heights2, sex == "Female")

# Question 7a

# How many females are in the heights2 dataset? 238

sum(heights2$sex == "Female")

# Question 7b

# What is the mean height of the females in centimeters? 165

mean(females$ht_cm)

# Question 8

# The olive dataset in dslabs contains composition in percentage of eight fatty acids
# found in the lipid fraction of 572 Italian olive oils:

library(dslabs)
data(olive)
head(olive)

# Plot the percent palmitic acid versus palmitoleic acid in a scatterplot.
# What relationship do you see?

plot(olive$palmitic, olive$palmitoleic)

# There is a positive linear relationship between palmitic and palmitoleic

# Question 9

# Create a histogram of the percentage of eicosenoic acid in olive. Which is true?

hist(olive$eicosenoic)

# The most common value of eicosenoic is below 0.05%

# Question 10

# Make a boxplot of palmitic acid percentage in olive with separate distributions
# for each region.

boxplot(palmitic~region, olive)

# Which region has the highest median palmitic acid percentage? Southern Italy

# Which region has the most variable palmitic acid percentage? Southern Italy

## SECTION 4. PROGRAMMING BASICS

# 4.1. INTRODUCTION TO PROGRAMMING IN R

# 4.2. BASIC CONDITIONALS

# 4.3. FUNCTIONS

# 4.4. FOR LOOPS

# DataCamp Assessment 4

# Conditionals

# What will this conditional expression return? Run it from the console.

x <- c(1, 2, -3, 4)
if(all(x>0)){
  print("All Positives")
} else{
  print("Not All Positives")
}

# "Not All Positives"

# Conditional continued

# Which of the following expressions is always FALSE when at least one
# entry of a logical vector x is TRUE? You can try examples in the R console.

all(x)
any(x)
any(!x)
all(!x) # This one

# ifelse

# The function nchar tells you how many characters long a character vector is.
# For example:

library(dslabs)
data(murders)

char_len <- nchar(murders$state)
head(char_len)

# The function ifelse is useful because you convert a vector of logicals into
# something else. For example, some datasets use the number -999 to denote NA.
# A bad practice! You can convert the -999 in a vector to NA using the following
# ifelse call:

x <- c(2, 3, -999, 1, 4, 5, -999, 3, 2, 9)
ifelse(x == -999, NA, x)

# If the entry is -999 it returns NA, otherwise it returns the entry.

# We will combine a number of functions for this exercise. Use the ifelse function
# to write one line of code that assigns to the object new_names the state
# abbreviation when the state name is longer than 8 characters and assigns the 
# state name when the name is not longer than 8 characters.

# For example, where the original vector has `Massachusetts` (13 characters), the
# new vector should have MA. But where the original vector has `New York` (8 
# characters), the new vector should have New York as well.

# Assign the state abbreviation when the state name is longer than 8 characters

new_names <- ifelse(nchar(murders$state) > 8, murders$abb, murders$state)

# Defining functions

# You will encounter situations in which the function you need does not already
# exist. R permits you to write your own. Let's practice one such situation, 
# in which you first need to define the function to be used. The functions you
# define can have multiple arguments as well as default values.

# To define functions we use function(). For example the following function
# adds 1 to the number it receives as an argument.

my_func <- function(x){
  y <- x + 1
  y
}

# The last value in the function, in this case that stored in y, gets returned.

# If you run the code above, R does not show anything. This means you defined
# the function. You can test it out like this:

my_func(5)

# We will define a function sum_n for this exercise. Create a function sum_n
# that for any given value, say n, creates the vector 1:n, and then computes the
# sum of the integers from 1 to n.

# Use the function you just defined to determine the sum of integers from 1 to
# 5,000.

# Create the function `sum_n`

sum_n <- function(n){
  sum(1:n)
}

# Use the function to determine the sum of integers from 1 to 5,000

sum_n(5000)

# Defining functions continued

# We will make another function for this exercise. We will define a function
# altman_plot that takes two arguments x and y and plots the difference y - x
# in the y-axis and the sum x + y in the x-axis.

# You can define functions with as many variables as you want. For example, 
# here we need at least two, x and y. The following function plots log
# transformed values:

log_plot <- function(x, y){
  plot(log10(x), log10(y))
}

# This function does not return anything. It just makes a plot.

# We will make another function in this exercise. Create a function altman_plot
# that takes two arguments x any y and plots y - x (on the y-axis) against
# x + y (on the x-axis)

# Note: don't use parentheses around the arguments in the plot function because
# you will confuse R.

# Create `altman_plot`

altman_plot <- function(x, y){
  plot(x + y, y - x)
}

# Lexical scope

# Lexical scoping is a convention used by many languages that determine when an
# object is available by its name. When you run the code below you will see which
# x is available at different points in the code.

x <- 8
my_func <- function(y){
  x <- 9
  print(x)
  y + x
}
my_func(x)
print(x)

# Note that when we define x as 9, this is inside the function, but it is 8 after you
# run the function. The x changed inside the function but not outside.

# After running the code below, what is the value of x?

x <- 3
my_func <- function(y){
  x <- 5
  y
  print(x)
}
my_func(x)

# Run this code

x <- 3
my_func <- function(y){
  x <- 5
  y + 5
}

# Print the value of x

print(x)

# For loops

# In the next exercise we are going to write a for-loop. In that for-loop we are going
# to call a function. We define that function here:

# Write a function compute_s_n that for any given n computes the sum s_n = 1^2 + 2^2 +
# + #^2 + ... + n^2. Report the value for n = 10.

# Here is an example of a function that adds numbers from 1 to n

example_func <- function(n){
  x <- 1:n
  sum(x)
}

# Here is the sum of the first 100 numbers

example_func(100)

# Write a function compute_s_n with argument n that for any given n computes the sum
# of 1 +2^2 + ... + n^2

compute_s_n <- function(n){
  x <- 1:n
  sum(x^2)
}

# Report the value of the sum when n = 10

compute_s_n(10)

# For loops continued

# Now we are going to compute the sum of the squares for several values of n. We will
# use a for-loop for this. Here is an example of a for-loop.

results <- vector("numeric", 10)
n <- 10
for(i in 1:n){
  x <- 1:i
  results[i] <- sum(x)
}

# Note that we start with a call to vector which constructs an empty vector that we
# will fill while the loop runs.

# Define an empty numeric vector s_n of size 25 using

s_n <- vector("numeric", 25)

# Compute the sum when n is equal to each integer from 1 to 25 using the function we
# defined in the previous exercise compute_s_n

# Save the results in s_n

# Define a function and store it in `compute_s_n`

compute_s_n <- function(n){
  x <- 1:n
  sum(x^2)
}

# Create a vector for storing results

s_n <- vector("numeric", 25)

# Write a for-loop to store the results in s_n
n <- 25
for(i in 1:n){
  s_n[i] <- compute_s_n(i)
}

# Checking our math

# If we do the math, we can show that 
# s_n = 1^2 + 2^2 + 3^2 + ... + n^2 = n * (n + 1) * (2*n +1) / 6

# We have already computed the values of s_n from 1 to 25 using a for-loop. If the
# formula is correct then a plot of s_n versus n should look cubic. Let's make it.

# Define n <- 1:25. Note that with this we can use for(i in n)

# Use a for-loop to save the sums into a vector s_n <- vector("numeric", 25)

# Plot s_n (on the y-axis) against n (on the x-axis)

# Define the function

compute_s_n <- function(n){
  x <- 1:n
  sum(x^2)
}

# Define the vector of n

n <- 1:25

# Define the vector to store data

s_n <- vector("numeric", 25)
for(i in n){
  s_n[i] <- compute_s_n(i)
}

# Create the plot

plot(n, s_n)

# Checking our math (cont'd)

# Now let's actually check if we get the exact same answer.

# Confirm that s_n and n(n+1)(2n+1)/6 are the same using the identical command.

# Define the function

compute_s_n <- function(n){
  x <- 1:n
  sum(x^2)
}

# Define the vector of n

n <- 1:25

# Define the vector to store the data

s_n <- vector("numeric", 25)
for(i in n){
  s_n[i] <- compute_s_n(i)
}

# Check that s_n is identical to the formula given in the instructions

identical(s_n, n*(n+1)*(2*n+1)/6)

# SECTION 4 ASSESSMENT

# Question 1

# Load the heights dataset from dslabs

library(dslabs)
data(heights)

# Write an ifelse() statement that returns 1 if the sex is Female and 2 if
# sex is Male

ifelse(heights$sex == "Female", 1, 2)

# What is the sum of the resulting vector?

sum(ifelse(heights$sex == "Female", 1, 2)) # 1862

# Question 2

# Write an ifelse statement that takes the height column and returns the height if
# it is greated than 72 inches and returns 0 otherwise.

# What is the mean of the resulting vector?

mean(ifelse(heights$height > 72, heights$height, 0))

# Question 3

# Write a function inches_to_ft that takes the number of inches x and returns the 
# number of feet. One foot equals 12 inches

inches_to_ft <- function(x){
  x / 12
}

# What is inches_to_ft(144)? 12

inches_to_ft(144)

# How many individuals in the heights dataset have a height less than 5 feet? 20

sum(inches_to_ft(heights$height) < 5)

# Question 4

# Which of the following are TRUE?

any(TRUE, TRUE, TRUE) #
any(TRUE, TRUE, FALSE) #
any(TRUE, FALSE, FALSE) #
any(FALSE, FALSE, FALSE)
all(TRUE, TRUE, TRUE) #
all(TRUE, TRUE, FALSE)
all(TRUE, FALSE, FALSE)
all(FALSE, FALSE, FALSE)

# Question 5

# Given an integer x, the factorial of x is called x! and is the product of all integers
# up to and including x. The factorial() function computes factorials in R. For example, 
# factorial(4) returns 4! = 4 * 3 * 2 * 1 = 24.

# Define a vector of length m
m <- 10
f_n <- vector(length = m)

# Make a vector of factorials
for(n in 1:m){ # This is the missing line of code
  f_n[n] <- factorial(n)
}

# inspect f_n
f_n

# Complete the code above to generate a vector of length m where the first entry is 1!,
# the second entry is 2!, and so up to m!