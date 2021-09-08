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

