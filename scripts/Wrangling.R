### DATA SCIENCE: WRANGLING!

# In this course you will learn:

# How to import data into R from different file formats.
# How to scrape data from the web.
# How to tidy data using the tidyverse to better facilitate analysis.
# How to process strings with regular expressions (regex).
# How to wrangle data using dplyr.
# How to work with dates and times as file formats.
# How to mine text.

# INTRODUCTION TO WRANGLING

# The first step in data analysis is importing, tidying and cleaning data. This is the process
# of data wrangling.

# In this course, we cover several common steps of the data wrangling process: tidying data, 
# string processing, html parsing, working with dates and times, and text mining.

## SECTION 1: DATA IMPORT

# In the Data Import section, you will learn how to import data into R.

# Import data from spreadsheets.
# Identify and set your working directory and specify the path to a file.
# Use the readr and readxl packages to import spreadsheets.
# Use R-base functions to import spreadsheets.
# Download files from the internet using R.

# 1.1. DATA IMPORT

# IMPORTING SPREADSHEETS

# Many datasets are stored in spreadsheets. A spreadsheet is essentially a file version of a
# data frame with rows and columns.

# Spreadsheets have rows separated by returns and columns separated by a delimiter. The most
# common delimiters are comma, semicolon, white space and tab.

# Many spreadsheets are raw text files and can be read with any basic text editor. However,
# some formats are proprietary and cannot be read with a text editor, such as Microsoft Excel
# files (.xls).

# Most important functions assume that the first row of a spreadsheet file is a header with
# column names. To know if the file has a header, it helps to look at the file with a text
# editor before trying to import it.

# PATHS AND THE WORKING DIRECTORY

# The working directory is where R looks for files and saves files by default.

# See your working directory with getwd(). Change your working directory with setwd().

# We suggest you create a directory for each project and keep your raw data inside that
# directory.

# Use the file.path() function to generate a full path from a relative path and a file name.
# Use file.path() instead of paste() because file.path() is aware of your operating system
# and will use the correct slashes to navigate your machine.

# The file.copy() function copies a file to a new path.

# See the working directory
getwd()

# Change your working directory
setwd()

# Set path to the location for raw data files in the dslabs package and list files
path <- system.file("extdata", package = "dslabs")
list.files(path)

# Generate a full path to a file
filename <- "murders.csv"
fullpath <- file.path(path, filename)
fullpath

# copy file from dslabs package to your working directory
file.copy(fullpath, getwd())

# check if the file exists
file.exists(filename)

# THE READR AND READXL PACKAGES

# readr is the tidyverse library that includes functions for reading data stored in text
# file spreadsheets into R. Functions in the package include read_csv(), read_tsv(), read_delim(),
# and more. These differ by the delimiter they use to split columns. 

# The readxl package provides functions to read Microsoft Excel formatted files.

# The excel_sheets() function gives the names of the sheets in the Excel file. These names
# are passed to the sheet argument for the readxl functions read_excel(), read_xls() and
# read_xlsx().

# The read_lines() function shows the first few lines of a file in R.

# Code

library(dslabs)
library(tidyverse) # includes readr
library(readxl)

# Inspect the first 3 lines
read_lines("murders.csv", n_max = 3)

# Read file in CSV format
dat <- read_csv(filename)

# Read using fullpath
dat <- read_csv(fullpath)
head(dat)

# Ex:
path <- system.file("extdata", package = "dslabs")
files <- list.files(path)
files

filename <- "murders.csv"
filename1 <- "life-expectancy-and-fertility-two-countries-example.csv"
filename2 <- "fertility-two-countries-example.csv"
dat = read.csv(file.path(path, filename))
dat1 = read.csv(file.path(path, filename1))
dat2 = read.csv(file.path(path, filename2))

# IMPORTING DATA USING R-BASE FUNCTIONS

# R-base import functions (read.csv(), read.table(), read.delim()) generate data frames
# rather than tibbles and character variables are converted to factors. This can be avoided
# by setting the argument stringsAsFactors = FALSE.

# Code

# Filename is defined as above
# read.csv converts strings to factors
dat2 <- read.csv(filename)
class(dat2$abb)
class(dat2$region)

# UPDATE: The function read.table() has now changed its default from stringsAsFactors = TRUE
# to stringsAsFactors = FALSE. In this way, when we are using read.table(), read.csv(), or
# read.delim(), the character features in the data file won't be automatically read in as
# "factor", but instead will remain as "character".

# DOWNLOADING FILES FROM THE INTERNET

# The read_csv() function and other import functions can read a URL directly.
# If you want to have a local copy of the file, you can use download.file().
# tempdir() creates a directory with a name that is very unlikely not to be unique.
# tempfile() creates a character string that is likely to be a unique file name.

# Code
url <- "https://raw.githubusercontent.com/rafalab/dslabs/master/inst/extdata/murders.csv"
dat <- read_csv(url)
download.file(url, "murders.csv")
tempfile()
tmp_filename <- tempfile()
download.file(url, tmp_filename)
dat <- read_csv(tmp_filename)
file.remove(tmp_filename)

# ASSESSMENT PART 1. DATA IMPORT

# Question 1

# Which of the following is NOT part of the data wrangling process?

# Importing data into R
# Formatting dates/times
# Checking correlations between your variables # Correct
# Tidying data

# Question 2

# Which files could be opened in a basic text editor? Select all that apply.

# data.txt # Correct
# data.csv # Correct
# data.xlsx
# data.tsv # Correct

# Question 3

# You want to analyze a file containing race finish times for a recent marathon. You
# open the file in a basic text editor and see lines that look like the following:

# initials,state,age,time
# vib,MA,61,6:01
# adc,TX,45,5:45
# kme,CT,50,4:19

# What type of file is this?

# A comma-delimited file without a header
# A tab-delimited file with a header
# A white space-delimited file without a header
# A comma-delimited file with a header # Correct

# Question 4

# ASsume the following is the full path to the directory that a student wants to use as
# their working directory in R: "/Users/student/Documents/projects/"

# Which of the following lines of code CANNOT set the working directory to the desired
# "projects" directory?

# setwd("~/Documents/projects/")
# setwd("/Users/student/Documents/projects/")
# setwd(/Users/student/Documents/projects/) # Correct
# dir <- "Users/student/Documents/projects/"
#     setwd(dir)

# Question 5

# We want to copy the "murders.csv" file from the dslabs package into an existing folder
# "data", which is located in our HarvardX-Wrangling projects folder. We first enter the code
# below into our RStudio console.

# getwd()
# [1] "C:/Users/UNIVERSITY/Documents/Analyses/HarvardX-Wrangling"
# filename <- "murders.csv"
# path <- system.file("exdata", package = "dslabs")

# Which of the following commands would NOT successfully copy "murders.csv" into the
# folder "data"?

# file.copy(file.path(path, "murders.csv"), getwd()) # Correct

# setwd("data")
#    file.copy(file.path(path, filename), getwd())

# file.copy(file.path(path, "murders.csv"), file.path(getwd(), "data"))

# file.location <- file.path(system.file("extdata", package = "dslabs"), "murders.csv")
#    file.destination <- file.path(getwd(), "data")
#    file.copy(file.location, file.destination)

# Question 6

# You are not sure whether the murders.csv file has a header row. How could you check this?
# Select all that apply.

# Open the file in a basic text editor. # Correct

# In the RStudio "Files" pane, click on your file, then select "View File".

# Use the command read_lines (remembering to specify the number of rows with the n_max
# argument).

# Question 7

# What is one difference between read_excel() and read_xlsx()?

# read_excel() also reads meta-data from the excel file, such as sheet names, while
# read_xlsx() only reads the first sheet in a file.

# read_excel() reads both .xls and .xlsx files by detecting the file format from its extension,
# while read_xlsx() only reads .xlsx files. # Correct

# read_excel() is part of the readr package, while read_xlsx() is part of the readxl package
# and has more options.

# read_xlsx() has been replaced by read_excel() in a recent readxl package update.

# Question 8

# You have a file called "times.txt" that contains race finish times for a marathon. The
# first four lines of the file look like this:

# initials,state,age,time
# vib,MA,61,6:01
# adc,TX,45,5:45
# kme,CT,50,4:19

# Which line of code will NOT produce a tibble with column names "initials", "state", "age",
# and "time"?

# race_times <- read_csv("times.txt")
# race_times <- read.csv("times.txt") # Correct, creates a data frame rather than a tibble
# race_times <- read_csv("times.txt", col_names = TRUE)
# race_times <- read_delim("times.txt", delim = ",")

# Question 9

# You also have access to marathon finish times in the form of an Excel document named
# times.xlsx. In the Excel document, different sheets contain race information for different
# years. The first sheet is named "2015", the second is named "2016", and the third is named
# "2017".

# Which line of code will NOT import the data contained in the "2016" tab of this Excel sheet?

# times_2016 <- read_excel("times.xlsx", sheet = 2)
# times_2016 <- read_xlsx("times.xlsx", sheet = "2") # Correct
# times_2016 <- read_excel("times.xlsx", sheet = "2016")
# times_2016 <- read_xlsx("times.xlsx", sheet = 2)

# Question 10

# You have a comma-separated values file that contains the initials, home states, ages, and
# race finish times for marathon runners. The runners' initials contain three characters for
# the runners' first, middle, and last names (for example, "KME").

# You read in the file using the following code.

# race_times <- read.csv("times.csv")

# What is the data type of the initials in the object race_times?

# integers
# characters # Correct
# factors
# logical

# Question 11

# Which of the following is NOT a real difference between the readr import functions and the
# base R import functions?

# The import functions in the readr package all start as read_, while the import functions for
# base R all start with read.

# Base R import functions automatically convert character columns to factors. 

# The Base R import functions can read .csv files, but cannot read files with other delimiters,
# such as .tsv files, or fixed-width files. # Correct

# Base R functions import data as a data frame, while readr functions import data as a tibble.

# Question 12

# You read in a file containing runner information and marathon finish times using the
# following code.

# race_times <- read.csv("times.csv", stringAsFactors = F)

# What is the class of the object race_times?

# data frame # Correct
# tibble
# matrix
# vector

# Question 13

# Select the answer choice that summarizes all of the actions that the following lines of
# code can perform. Please note that the url below is an example and does not lead to data.

# url <- "https://raw.githubusercontent.com/MyUserName/MyProject/master/MyData.csv"
# dat <- read_csv(url)
# download.file(url, "MyData.csv")

# Create a tibble in R called dat that contains the information contained in the csv file
# stored on GitHub and save that tibble to the working directory.

# Create a matrix in R called dat that contains the information contained in the csv file
# stored on GitHub. Download the csv file to the working directory and name the downloaded
# file "MyData.csv".

# Create a tibble in R called dat that contains the information contained in the csv file
# stored on GitHub. Download the csv file to the working directory and randomly assign it
# a temporary name that is very likely to be unique.

# Create a tibble in R called dat that contains the information contained in the csv file
# stored on GitHub. Download the csv file to the working directory and name the downloaded
# file "MyData.csv". # Correct

# ASSESSMENT PART 2: DATA IMPORT

# Question 14

# Inspect the file at the following URL:

# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

# Which readr function should be used to import this file?

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data <- read_table(url) # Doesn't work, not separating columns
data <- read_csv(url) # Correct
data <- read_csv2(url) # Doesn't seem to work, not separating columns
data <- read_tsv(url) # Ibid
# None of the above 

# Question 15

# Check the documentation for the readr function you chose in the previous question to learn about
# its arguments. Determine which arguments you need to the file from the previous question:

url

# Does this file have a header row? Does the reader function you chose need any additional
# arguments to import the data correctly?

?read_csv

# Yes, there is a header. No arguments are needed.
# Yes, there is a header. The header = TRUE argument is necessary.
# Yes, there is a header. The col_names = TRUE argument is necessary.
# No, there is no header. No arguments are needed.
# No, there is no header. The header = FALSE argument is necessary.
# No, there is no header. The col_names = FALSE argument is necessary. # Correct

data <- read_csv(url, col_names = FALSE)

# Question 16

# Inspect the imported data from the previous question.

# How many rows are in the dataset?

nrow(data) # 569

# How many columns are in the dataset?

ncol(data) # 32

## SECTION 2. TIDY DATA

# In this section, you will learn how to convert data from a raw to a tidy format.

# This section is divided into three parts: Reshaping Data, Combining Tables, and
# Web Scrapping.

# You will be able to:

# Reshape data using functions from the tidyr package, including gather(), spread(),
# separate() and unite().

# Combine information from different tables using join functions from the dplyr package.

# Combine information from different tables using binding functions from the dplyr package.

# Use set operators to combine data frames.

# Gather data from a website through web scrapping and use of CSS selectors.

# 2.1. RESHAPING DATA

# TIDY DATA

# In tidy data, each row represents an observation and each column represents a different
# variable.

# In wide data, each row includes several observations and one of the variables is stored
# in the header.

library(tidyverse)
library(dslabs)
data(gapminder)

# Create and inspect a tidy data frame
tidy_data <- gapminder %>%
  filter(country %in% c("South Korea", "Germany")) %>%
  select(country, year, fertility)
head(tidy_data)

# Plotting tidy data is simple
tidy_data %>%
  ggplot(aes(year, fertility, color = country)) +
  geom_point()

# Import and inspect example of original Gapminder data in wide format
path <- system.file("extdata", package = "dslabs")
filename <- file.path(path, "fertility-two-countries-example.csv")
wide_data <- read_csv(filename)
select(wide_data, country, `1960`:`1967`)

# RESHAPING DATA

# The tidyr package includes several functions that are useful for tidying data.

# The gather() function converts wide data into tidy data.

# The spread() function converts tidy data to wide data.

# Original wide data
library(tidyverse)
path <- system.file("extdata", package = "dslabs")
filename <- file.path(path, "fertility-two-countries-example.csv")
wide_data <- read_csv(filename)

# Tidy data from dslabs
library(dslabs)
data(gapminder)
tidy_data <- gapminder %>%
  filter(country %in% c("South Korea", "Germany")) %>%
  select(country, year, fertility)

# Gather wide data to make new tidy data
new_tidy_data <- wide_data %>%
  gather(year, fertility, -country)

# Gather treats column name as characters by default
class(tidy_data$year)
class(new_tidy_data$year)

# Convert gathered column names to numberic
new_tidy_data <- wide_data %>%
  gather(year, fertility, -country, convert = TRUE)
class(new_tidy_data$year)

# ggplot works on new tidy data
new_tidy_data %>%
  ggplot(aes(year, fertility, color = country)) +
  geom_point()

# Spread tidy data to generate wide data
new_wide_data <- new_tidy_data %>% spread(year, fertility)
select(new_wide_data, country, `1960`:`1967`)

# SEPARATE AND UNITE

# The separate() function splits one column into two or more columns at a specified
# character that separates the variables.

# When there is an extra separation in some of the entries, use fill = "right" to pad
# missing values with NAs, or use extra = "merge" to keep extra elements together.

# The unite() function combines two columns and adds a separating character.

# Import data
path <- system.file("extdata", package = "dslabs")
filename <- file.path(path, "life-expectancy-and-fertility-two-countries-example.csv")
raw_dat <- read_csv(filename)
select(raw_dat, 1:5)

# Gather all columns except country
dat <- raw_dat %>% gather(key, value, -country)
head(dat)
dat$key[1:5]

# Separate on underscores
dat %>% separate(key, c("year", "variable_name"), "_")
dat %>% separate(key, c("year", "variable_name"))

# Split on all underscores, pad empty cells with NA
dat %>% separate(key, c("year", "first_variable_name", "second_variable_name"),
                 fill = "right")

# Split on first underscore but keep life_expectancy merged
dat %>% separate(key, c("year", "variable_name"), sep = "_", extra = "merge")

# Separate then spread
dat %>%
  separate(key, c("year", "variable_name"), sep = "_", extra = "merge") %>%
  spread(variable_name, value)

# Separate then unite
dat %>%
  separate(key, c("year", "first_variable_name", "second_variable_name"), fill = "right") %>%
  unite(variable_name, first_variable_name, second_variable_name, sep = "_")

# Full code for tidying data
dat %>%
  separate(key, c("year", "first_variable_name", "second_variable_name"), fill = "right") %>%
  unite(variable_name, first_variable_name, second_variable_name, sep = "_") %>%
  spread(variable_name, value) %>%
  rename(fertility = fertility_NA)

# ASSESSMENT PART 1: RESHAPING DATA

# Question 1

# A collaborator sends you a file containing data of three years of average race finish
# times.

# age_group,2015,2016,2017
# 20,3:46,3:22,3:50
# 30,3:50,3:43,4:43
# 40,4:39,3:49,4:51
# 50,4:48,4:59,5:01

# Are these data considered "tidy" in R? Why or why not?

# Yes. These data are considered "tidy" because each row contains unique observations.
# Yes. These data are considered "tidy" because there are no missing data in the data frame.
# No. These data are not considered "tidy" because the variable "year" is stored in the header. [X] 
# No. These data are not considered "tidy" because there are not an equal number of columns
# and rows.

# Question 2

# Below are four versions of the same dataset. Which one is in a tidy format?

# 1

# state,abb,region,population,total
# Alabama,AL,South,4779736,135
# Alaska,AK,West,710231,19
# ...

# Question 3

# Your file called "times.csv" has age groups and average race times for three years of
# marathons.

# age_group,2015,2016,2017
# 20,3:46,3:22,3:50
# 30,3:50,3:43,4:43
# 40,4:39,3:49,4:51
# 50,4:48,4:59,5:01

# You read in the data file using the following command:

d <- read_csv("times.csv")

# Which commands will help you "tidy" the data?

# tidy_data <- d %>% gather(year, time, `2015`:`2017`) [X]
# tidy_data <- d %>% spread(year, time, `2015`:`2017`)
# tidy_data <- d %>% gather(age_group, year, time, `2015`:`2017`)
# tidy_data <- d %>% gather(time, `2015`:`2017`)

# Question 4

# You have a dataset on U.S. contagious diseases, but it is in the following wide format:

head(dat_wide)

# state,year,population,HepatitisA,Mumps,Polio,Rubella
# Alabama,1990,4040587,86,19,76,1
# Alabama,1991,4066003,39,14,65,0
# ...

# You want to transform this into a tidy dataset, with each row representing an observation
# of the incidence of each specific disease (as show below):

head(dat_tidy)

# state,year,population,disease,count
# Alabama,1990,4040587,HepatitisA,86
# Alabama,1991,4066003,HepatitisA,39
# ...

# Which of the following commands would achieve this transformation to tidy data? (Pay
# attention to the column names).

# dat_tidy <- dat_wide %>% gather(key = count, value = disease, HepatitisA, Rubella)
# dat_tidy <- dat_wide %>% gather(key = count, value = disease, -state, -year, -population)
# dat_tidy <- dat_wide %>% gather(key = disease, value = count, -state)
# dat_tidy <- dat_wide %>% gather(key = disease, value = count, HepatitisA:Rubella) [X]

# Question 5

# You have successfully formatted marathon finish times into a tidy object called tidy_data.
# The first few lines are shown below

# age_group,year,time
# 20,2015,03:46
# 30,2015,03:50
# 40,2015,04:39
# 50,2015,04:48
# 20,2016,03:22

# Select the code that converts these data back to the wide format, where each year has a 
# separate column

# tidy_data %>% spread(time, year)
# tidy_data %>% spread(year, time) [X]
# tidy_data %>% spread(year, age_group)
# tidy_data %>% spread(time, year, `2015`:`2017`)

# Question 6

# You have the following dataset:

head(dat)

# state,abb,region,var,people
# Alabama,AL,South,population,4779736
# Alabama,AL,South,total,135
# Alaska,AK,West,population,710231
# ...

# You would like to transform it into a dataset where population and total are each their
# own column (shown below):

# state,abb,region,population,total
# Alabama,AL,South,4779736,135
# Alaska,AK,West,710231,19
# ...

# Which code would best accomplish this?

# dat_tidy <- dat %>% spread(key = var, value = people) [X]
# dat_tidy <- dat %>% spread(key = state:region, value = people)
# dat_tidy <- dat %>% spread(key = people, value = var)
# dat_tidy <- dat %>% spread(key = region, value = people)

# Question 7

# A collaborator sends you a file containing data for two years of average race finish times,
# "times.csv".

# age_group,2015_time,2015_participants,2016_time,2016_participants
# 20,3:46,54,3:22,62
# 30,3:50,60,3:43,58
# ...

# You read in the data file:
d <- read_csv("times.csv")

# Which of the answers below best makes the data tidy?

# 2 is correct

tidy_data <- d %>%
  gather(key = "key", value = "value", -age_group) %>%
  separate(col = "key", into = c("year", "variable_name"), sep = "_") %>%
  spread(key = variable_name, value = value)

# Question 8

# You are in the process of tidying some data on heights, hand length, and wingspan for
# basketball players in the draft. Currently, you have the following:

head(stats)

# key,value
# allen_height,75
# allen_hand_length,8.25
# allen_wingspan,79.25
# bamba_height,83.25
# ...

# Select all of the correct commands below that would turn this data into a "tidy" format
# with columns "height", "hand_length" and "wingspan".

key <- c("allen_height", "allen_hand_length", "allen_wingspan", "bamba_height",
         "bamba_hand_length", "bamba_wingspan")
value <- c(75,8.25,79.25,83.25,9.75,94)

stats <- data.frame(key, value)

# 1 is correct

tidy_data <- stats %>%
  separate(col = key, into = c("player", "variable_name"), sep = "_", extra = "merge") %>%
  spread(key = variable_name, value = value)

# 2 doesn't work because col names include "_NA"

tidy_data <- stats %>%
  separate(col = key, into = c("player", "variable_name1", "variable_name2"), sep = "_",
           fill = "right") %>%
  unite(col = variable_name, variable_name1, variable_name2, sep = "_") %>%
  spread(key = variable_name, value = value)

# 3 doesn't work because hand_length is only spelled as "hand"

tidy_data <- stats %>%
  separate(col = key, into = c("player", "variable_name"), sep = "_") %>%
  spread(key = variable_name, value = value)

# ASSESSMENT PART 2: RESHAPING DATA

library(tidyverse)
library(dslabs)

# Question 9

# Examine the built-in dataset co2. This dataset comes with base R, not dslabs. Just type
# co2 to access the dataset.

# Is co2 tidy? Why or why not?

# co2 is tidy data: it has one year for each row.
# co2 is tidy data: each column is a different month.
# co2 is not tidy: there are multiple observations per column.
# co2 is not tidy: to be tidy we would have to wrangle it to have three columns (year, month
# and value), and then each co2 observation would have a row. [X]

head(co2)

# Question 10

# Run the following command to define the co2_wide object:

co2_wide <- data.frame(matrix(co2, ncol = 12, byrow = TRUE)) %>%
  setNames(1:12) %>%
  mutate(year = as.character(1959:1997))

# Use the gather function to make this dataset tidy. Call the column with the co2 measurements
# co2 and call the month column month. Name the resulting object co2_tidy.

# Which code would return the correct tidy format?

# co2_tidy <- gather(co2_wide,month,co2,year)
# co2_tidy <- gather(co2_wide,co2,month,-year)
# co2_tidy <- gather(co2_wide,co2,month,year)
# co2_tidy <- gather(co2_wide,month,co2,-year) [X]

co2_tidy <- gather(co2_wide, month, co2, -year)

# Question 11

# Use co2_tidy to plot co2 versus month with a different curve for each year:

co2_tidy %>% ggplot(aes(as.numeric(month), co2, color = year)) + geom_line()

# What can be concluded from this plot?

# co2 concentrations increased monotonically (never decreased) from 1959 to 1997.
# co2 concentrations are highest around May and the yearly average increased from 1959 to 1997. [X]
# co2 concentrations are highest around October and the yearly average increased from 1959
# to 1997.
# Yearly average co2 concentrations have remained constant over time.
# co2 concentrations do not have a seasonal trend.

# Question 12

# Load the admissions dataset from dslabs, which contains college admission information for men
# and women across six majors, and remove the applicants percentage column.

library(dslabs)
data(admissions)
dat <- admissions %>% select(-applicants)

# Your goal is to get the data in the shape that has one row for each major, like this:

# major,men,women
# A,62,82,
# B,63,68
# ...

# Which command could help you to wrangle the data into the desired format?

# dat_tidy <- spread(dat, major, admitted)
# dat_tidy <- spread(dat, gender, major)
# dat_tidy <- spread(dat, gender, admitted) [X]
# dat_tidy <- spread(dat, admitted, gender)

dat_tidy <- spread(dat, gender, admitted)

# Question 13

# Now use the admissions dataset to create the object tmp, which has columns major, gender, key,
# and value:

tmp <- gather(admissions, key, value, admitted:applicants)
tmp

# Combine the key and gender and create a new column called column_name to get a variable with
# the following values: admitted_men, admitted_women, applicants_men and applicants_women. Save
# the dataset as tmp2.

# Which command could help you to wrangle the data into the desired format?

# tmp2 <- spread(tmp, column_name, key, gender)
# tmp2 <- gather(tmp, column_name, c(gender, key))
# tmp2 <- unite(tmp, column_name, c(gender, key))
# tmp2 <- spread(tmp, column_name, c(key, gender))
# tmp2 <- unite(tmp, column_name, c(key, gender)) [X]

tmp2 <- unite(tmp, column_name, c(key, gender))

# Question 14

# Which function can reshape tmp2 to a table with six rows and five columns named major,
# admitted_men, admitted_women, applicants_men, and applicants_women?

# gather()
# spread() [X]
# separate()
# unite()

tmp3 <- spread(tmp2, column_name, value)

# 2.2. COMBINING TABLES

# COMBINING TABLES

# The join functions in the dplyr package combine two tables such that matching rows are together.
# left_join() only keeps rows that have information in the first table.
# right_join() only keeps rows that have information in the second table.
# inner_join() only keeps rows that have information in both tables.
# full_join() keeps all rows from both tables.
# semi_join() keeps the part of first table for which we have information in the second.
# anti_join() keeps the elements of the first table for which there is no information in the second.

# import US murders data
library(tidyverse)
library(ggrepel)
library(dslabs)
ds_theme_set()
data(murders)
head(murders)

# import US election results data
data(polls_us_election_2016)
head(results_us_election_2016)
identical(results_us_election_2016$state, murders$state)

# join the murders table and US election results table
tab <- left_join(murders, results_us_election_2016, by = "state")
head(tab)

# plot electoral votes versus population
tab %>% ggplot(aes(population / 10^6, electoral_votes, label = abb)) +
  geom_point() +
  geom_text_repel() +
  scale_x_continuous(trans = "log2") +
  scale_y_continuous(trans = "log2") +
  geom_smooth(method = "lm", se = FALSE)

# make two smaller tables to demonstrate joins
tab1 <- slice(murders, 1:6) %>% select(state, population)
tab1
tab2 <- slice(results_us_election_2016, c(1:3, 5, 7:8)) %>% select(state, electoral_votes)
tab2

# experiment with different joins
left_join(tab1, tab2)
tab1 %>% left_join(tab2)
tab1 %>% right_join(tab2)
inner_join(tab1, tab2)
semi_join(tab1, tab2)
anti_join(tab1, tab2)

# BINDING

# Unlike the join functions, the binding functions do not try to match by a variable, but rather just
# combine datasets.

# bind_cols() binds two objects by making them columns in a tibble. The R-base function cbind() binds
# columns but makes a data frame or a matrix instead.

# The bind_rows() function is similar but binds rows instead of columns. The R-base function rbind()
# binds rows but makes a data frame or matrix instead.

bind_cols(a = 1:3, b = 4:6)

tab1 <- tab[, 1:3]
tab2 <- tab[, 4:6]
tab3 <- tab[, 7:9]
new_tab <- bind_cols(tab1, tab2, tab3)
head(new_tab)

tab1 <- tab[1:2,]
tab2 <- tab[3:4,]
bind_rows(tab1, tab2)

# SET OPERATORS

# By default, the set operators in R-base work on vectors. If tidyverse/dplyr are loaded, they also
# work on data frames.
# You can take intersections of vectors using intersect(). This returns the elements common to both
# sets. 
# You can take the union of vectors using union(). This returns the elements that are in either set.
# The set difference between a first and second argument can be obtained with setdiff(). Note that
# this function is not symmetric.
# The function set_equal() tells us if two sets are the same, regardless of the order of elements.

# intersect vectors or data frames
intersect(1:10, 6:15)
intersect(c("a", "b", "c"), c("b", "c", "d"))
tab1 <- tab[1:5,]
tab2 <- tab[3:7,]
intersect(tab1, tab2)

# perform a union of vectors or data frames
union(1:10, 6:15)
union(c("a", "b", "c"), c("b", "c", "d"))
tab1 <- tab[1:5,]
tab2 <- tab[3:7,]
union(tab1, tab2)

# set difference of vectors or data frames
setdiff(1:10, 6:15)
setdiff(6:15, 1:10)
tab1 <- tab[1:5,]
tab2 <- tab[3:7,]
setdiff(tab1, tab2)

# setequal determines whether sets have the same elements, regardless of order
setequal(1:5, 1:6)
setequal(1:5, 5:1)
setequal(tab1, tab2)

# ASSESSMENT: COMBINING TABLES

# Question 1

# You have created data frames tab1 and tab2 of state population and election data.
tab$state
tab1 <- slice(tab, c(1:3, 8:9)) %>% select(state, population)
tab1
tab2 <- slice(tab, c(1:3, 5:7)) %>% select(state, electoral_votes)
tab2
dim(tab1)
dim(tab2)

# What are the dimensions of the table dat, created by the following command?
dat <- left_join(tab1, tab2, by = "state")
dim(dat)

# 3 rows by 3 columns
# 5 rows by 2 columns
# 5 rows by 3 columns [X]
# 6 rows by 3 columns

dat

# Question 2

# We are still using the tab1 and tab2 tables shown in question 1. What join command would create
# a new table "dat" with three rows and two columns?

# dat <- right_join(tab1, tab2, by = "state")
# dat <- full_join(tab1, tab2, by = "state")
# dat <- inner_join(tab1, tab2, by = "state")
# dat <- semi_join(tab1, tab2, by = "state")

semi_join(tab1, tab2)

# Question 3

# Which of the following are real difference between the join and bind functions?

# Binding functions combine by position, while join functions match by variables. [X]
# Joining functions can join datasets of different dimensions, but the bind functions must match
## on the appropriate dimension (either same row or column numbers). [X]
# Bind functions can combine both vectors and dataframes, while join functions work only for dataframes. [X]
# The join functions are a part of the dplyr package and have been optimized for speed, while the bind
## functions are inefficient base functions.

# Question 4

# We have two simple tables, shown below, with columns x and y:
df1 <- data.frame(x = c("a", "b"), y = c("a", "a"))
df2 <- data.frame(x = c("a", "a"), y = c("a", "b"))

# Which command would result in the following table?
# x y
# b a

# final <- union(df1, df2)
# final <- setdiff(df1, df2) [X]
# final <- setdiff(df2, df1)
# final <- intersect(df1, df2)

setdiff(df1, df2)

# Introduction to Questions 5-7

# Install and load the Lahman library. This library contains a variety of datasets related to US
# professional baseball. We will use this library for the next few questions and will discuss it
# more extensively in the Regression course. For now, focus on wrangling the data rather than
# understanding the statistics.

# The Batting data frame contains the offensive statistics for all baseball players over several
# seasons. Filter this data frame to define top as the top 10 home run (HR) hitters in 2016:
library(Lahman)
top <- Batting %>%
  filter(yearID == 2016) %>%
  arrange(desc(HR)) %>% # arrange by descending HR count
  slice(1:10) # take entries 1-10
top %>% as_tibble()

# Also inspect the Master data frame, which has demographic information for all players:

Master %>% as_tibble()

# Question 5

# Use the correct join or bind function to create a combined table of the names and statistics
# of the top 10 home run (HR) hitters for 2016. This table should have the player ID, first name,
# last name, and number of HR for the top 10 players. Name this data frame top_names.

# Identify the join or bind that fills the blank in this code to create the correct table.

# top_names <- top %>% ## blank ## %>%
# select(playerID, nameFirst, nameLast, HR)

# rbind(Master)
# cbind(Master)
# left_join(Master) [X]
# right_join(Master)
# full_join(Master)
# anti_join(Master)

top_names <- top %>% left_join(Master) %>%
  select(playerID, nameFirst, nameLast, HR)

# Question 6

# Inspect the salaries data frame. Filter this data frame to the 2016 salaries, then use the
# correct bind join function to add a salary column to the top_names data frame from the previous
# question. Name the new data frame top_salary. Use this code framework

# top_salary <- Salaries %>% filter(yearID == 2016) %>%
# ## Blank ## %>%
# select(nameFirst, nameLast, teamID, HR, salary)

# Which bind or join function fills the blank to generate the correct table?

# rbind(top_names)
# cbind(top_names)
# left_join(top_names)
# right_join(top_names) [X]
# full_join(top_names)
# anti_join(top_names)

head(Salaries)

top_salary <- Salaries %>% filter(yearID == 2016) %>%
  right_join(top_names) %>%
  select(nameFirst, nameLast, teamID, HR, salary)

# Question 7

# Inspect the AwardsPlayers table. Filter awards to include only the year 2016.

# How many players from the top 10 hom run hitters won at least one award in 2016? Use a set operator.

head(AwardsPlayers)

awards_2016 <- AwardsPlayers %>% filter(yearID == 2016)
head(awards_2016)
intersect(awards_2016$playerID, top$playerID) # 3

# How many players won an award in 2016 but were not one of the top 10 home run hitters in 2016? Use
# a set operator.

length(setdiff(awards_2016$playerID, top$playerID)) # 44

# 2.3. WEB SCRAPING

# WEB SCRAPING

# Web scraping is extracting data from a website.
# The rvest web harvesting package includes functions to extract nodes of an HTML document:
# html_nodes() extracts all nodes of different types, and html_node() extracts the first node.
# html_table() converts an HTML table to a data frame.

# import a webpage into R
library(rvest)
url <- "https://en.wikipedia.org/wiki/Murder_in_the_United_States_by_state"
h <- read_html(url)
class(h)

library(tidyverse)
tab <- h %>% html_nodes("table")
tab <- tab[[2]]

tab <- tab %>% html_table
class(tab)

tab <- tab %>% setNames(c("state", "population", "total", "murders", 
                          "gun_murders", "gun_ownership", "total_rate", "murder_rate", "gun_murder_rate"))
head(tab)

# CSS SELECTORS

# The default look of webpages made with the most basic HTML is quite unattractive. The aesthetically
# pleasing pages we see today are made using CSS. CSS is used to add style to webpages. The fact that
# all pages for a company have the same style is usually a result that they all use the same CSS file.
# The general way these CSS files work is by defining how each of the elements of a webpage will look. The
# title, headings, itemized lists, tables, and links, for example, each receive their own style including
# font, color, size, and distance from the margin, among others.

# To do this CSS leverages patterns used to define these elements, reffered to as selectors. An example of
# pattern we used in a previous video is table but there are many many more. If we want to grab data from 
# a webpage and we happen to know a selector that is unique to the part of the page, we can use the
# html_nodes() function.

# However, knowing which selector to use can be quite complicated. To demonstrate this we will try to
# extract the recipe name, total preparation time, and list of ingredients from this guacamole recipe.
# Looking at the code for this page, it seems that the task is impossibly complex. However, selector
# gadgets actually make this possible. SelectorGadget is a piece of software that allows you to
# interatively determine what CSS selector you need to extract specific components from the webpage. If
# you plan on scraping data other than tables, we highly recommend you install it. A Chrome expansion
# is available which permits you to turn on the gadget highlighting parts of the page as you click through,
# showing the necessary selector to extract those segments.

# For the guacamole recipe page, we already have done this and determined that we need the following
# selectors:

h <- read_html("http://www.foodnetwork.com/recipes/alton-brown/guacamole-recipe-1940609")
recipe <- h %>% html_node(".o-AssetTitle__a-HeadlineText") %>% html_text()
prep_time <- h %>% html_node(".m-RecipeInfo__a-Description--Total") %>% html_text()
ingredients <- h %>% html_node(".o-Ingredients__a-Ingredient") %>% html_text()

# You can see how complex the selectors are. In any case we are now ready to extract what we want
# and create a list:

guacamole <- list(recipe, prep_time, ingredients)
guacamole

# Sice recipe pages from this website follow this general layout, we can use this code to create
# a function that extracts this information:

get_recipe <- function(url){
  h <- read_html(url)
  recipe <- h %>% html_node(".o-AssetTitle__a-HeadlineText") %>% html_text()
  prep_time <- h %>% html_node(".m-RecipeInfo__a-Description--Total") %>% html_text()
  ingredients <- h %>% html_nodes(".o-Ingredients__a-Ingredient") %>% html_text()
  return(list(recipe = recipe, prep_time = prep_time, ingredients = ingredients))
}

# and then use it on any of their webpages:

get_recipe("http://www.foodnetwork.com/recipes/food-network-kitchen/pancakes-recipe-1913844")

# There are several other powerful tools provided by rvest. For example, the functions html_form(),
# set_values(), and submit_form() permit you to query a webpage from R.

# ASSESSMENT: WEB SCRAPING

# Introduction: Questions 1-3

# Load the following web page, which contains information about Major League Baseball payrolls, into R:
library(rvest)
url <- "https://web.archive.org/web/20181024132313/http://www.stevetheump.com/Payrolls.htm"
h <- read_html(url)

# We learned that tables in html are associated with the table node. Use the html_nodes() function and
# the table node type to extract the first table. Store it in an object nodes:
nodes <- html_nodes(h, "table")

# The html_nodes() function returns a list of objects of class xml_node. We can see the content of each
# one using, for example, the html_text() function. You can see the content for an arbitrarily picked
# component like this:
html_text(nodes[[8]])

# If the content of this object is an html table, we can use the html_table() function to convert it to
# a data frame:
html_table(nodes[[8]])

# Question 1

# Many tables on this page are team payroll tables, with columns of rank, team, and one or more money
# values.

# Convert the first four tables in nodes to data frames and inspect them.

# Which of the first four nodes are tables of team payroll.

# None
# Table 1
# Table 2 [X]
# Table 3 [X]
# Table 4 [X]

html_table(nodes[[1]])
html_table(nodes[[2]])
html_table(nodes[[3]])
html_table(nodes[[4]])

# Question 2

# For the last 3 components of nodes, which of the following are true? (Check all correct answers)

# All three entries are tables [X]
# All three entries are tables of payroll per team
# The last entry shows the average across all teams through time, not payroll per team [X]
# None of the three entries are tables of payroll per team

length(nodes) # 21
html_table(nodes[[19]])
html_table(nodes[[20]])
html_table(nodes[[21]])

# Question 3

# Create a table called tab_1 using entry 10 of nodes. Create a table tab_2 using entry 19 of nodes.

tab_1 <- html_table(nodes[[10]])
tab_2 <- html_table(nodes[[19]])

# Note that the column names should be c("Team", "Payroll", "Average"). You can see that these column
# names are actually in the first data row of each table, and that tab_1 has an extra first column No.
# that should be removed so that the column names for both tables match.

# Remove the extra column in tab_1, remove the first row of each dataset, and change the column names
# for each table to c("Team", "Payroll", "Average"). Use a full_join() by the Team to combine these
# two tables.

head(tab_1)
head(tab_2)
names(tab_1)
tab_1 <- tab_1 %>% select(X2, X3, X4)
tab_1 <- setNames(tab_1, c("Team", "Payroll", "Average"))
tab_1 <- slice(tab_1, 2:31)
tab_2 <- setNames(tab_2, c("Team", "Payroll", "Average"))
tab_2 <- slice(tab_2, 2:31)
head(tab_1)
head(tab_2)
full_join(tab_1, tab_2, by = "Team")

# How many rows are in the joined data table?
full_tab <- full_join(tab_1, tab_2, by = "Team")
nrow(full_tab) # 58

# Introduction to Questions 4 and 5

# The Wikipedia page on opinion polling for the Brexit referendum, in which the United Kingdom voted
# to leave the European Union in June 2016, contains several tables. One table contains the results
# of all polls regarding the referendum over 2016.

# Use the rvest library to read the HTML from this Wikipedia page (make sure to copy both line of the URL):
library(rvest)
library(tidyverse)
url <- "https://en.wikipedia.org/w/index.php?title=Opinion_polling_for_the_United_Kingdom_European_Union_membership_referendum&oldid=896735054"

# Question 4

# Assign tab to be the html nodes of the "table" class.

# How many tables are in this Wikipedia page?

h <- read_html(url)
nodes <- html_nodes(h, "table")
length(nodes) #41

# Question 5

# Inspect the first several html tables using html_table() with the argument fill=TRUE (you can read about
# this argument in the documentation). Find the first table that has 9 columns with the first column named
# "Date(s) conducted".

# What is the first table number to have 9 columns where the first column is named "Date(s) conducted"?

html_table(nodes[[1]])
html_table(nodes[[2]])
html_table(nodes[[3]])
html_table(nodes[[4]])
html_table(nodes[[5]])
html_table(nodes[[6]])

nodes[[6]] %>% html_table(fill = TRUE) %>% names() # inspect column names

## SECTION 3: STRING PROCESSING

# STRING PROCESSING OVERVIEW

# In the String Processing section, we use case studies that help demonstrate how string processing
# is a powerful tool useful for overcoming many data wrangling challenges. You will see how the
# original raw data was processed to create the data frames we have used in courses throughout this series.

# After completing this section, you will be able to:

# Remove unwanted characters from text
# Extract numeric values from text
# Find and replace characters
# Extract specific parts of strings
# Convert free form text into more uniform formats
# Split strings into multiple values
# Use regular expressions (regex) to process strings

# 3.1. STRING PROCESSING: PART 1

# STRING PARSING

# The most common tasks in string processing include:

# Extracting numbers from strings
# Removing unwanted characters from text
# Finding and replacing characters
# Extracting specific parts of strings
# Converting free form text to more uniform formats
# Splitting strings into multiple values

# The stringr package in the tidyverse contains string processing functions that follow a similar
# naming format (str_functionname) and are compatible with the pipe.

# read in raw numbers data from Wikipedia
url <- "https://en.wikipedia.org/w/index.php?title=Gun_violence_in_the_United_States_by_state&direction=prev&oldid=810166167"
murders_raw <- read_html(url) %>%
  html_nodes("table") %>%
  html_table() %>%
  .[[1]] %>%
  setNames(c("state", "population", "total", "murder_rate"))

# inspect data and column classes
head(murders_raw)
class(murders_raw$population)
class(murders_raw$total)

# DEFINING STRINGS: SINGLE AND DOUBLE QUOTES AND HOW TO ESCAPE

# Define a string by surrounding text with either single quotes or double quotes.

# To include a single quote inside a string, use double quotes on the outside. To include a double
# quote inside a string, use single quotes on the outside.

# The cat() function displays a string as it is represented inside R.

# To include a double quote inside of a string surrounded by double quotesm use the backslash `\` to
# escape the double quote. Escape a single quote to include it inside of a string define by single quotes.

# We will see additional uses of the escape later.

s <- "Hello!" # Double quotes define a string
s <- 'Hello!' # Single quotes define a string
s <- `Hello!` # backquotes do not

s <- "10"" # error, unclosed quotes
s <- '10"' # correct

# cat shows what the string actually looks like inside R
cat(s)

s <- "5'"
cat(s)

# to include both single and double quotes in a string, escape with \
s <- '5'10"' # error
s <- "5'10"" # error
s <- '5\'10"' # correct
cat(s)
s <- "5'10\""
cat(s)

# STRINGR PACKAGE

# The main types of string processing tasks are detecting, locating, extracting and replacing
# elements of strings.

# The stringr package from the tidyverse includes a variety of string processing functions that
# begin with str_ and take the string as the first argument, which makes them compatible with
# the pipe.

# direct conversion to numeric fails because of commas
murders_raw$population[1:3]
as.numeric(murders_raw$population[1:3])

library(tidyverse) # includes stringr

# CASE STUDY 1: US MURDERS DATA

# Use str_detect() to determine whether a string contains a certain pattern.

# Use the str_replace_all() function to replace all instances of one pattern with another pattern.
# To remove a pattern, replace with the empty string ""

# The parse_number() function removes punctuation from strings and converts them to numeric

# mutate_at() performs the same transformation on the specified column numbers

# detect whether there are commas in murders_raw
commas <- function(x) any(str_detect(x, ","))
murders_raw %>% summarize_all(funs(commas))

# replace commas with the empty string and convert to numeric
test_1 <- str_replace_all(murders_raw$population, ",", "")
test_1 <- as.numeric(test_1)

# parse number also removes commas and converst to numeric
test_2 <- parse_number(murders_raw$population)
identical(test_1, test_2)

murders_new <- murders_raw %>% mutate_at(2:3, parse_number)
murders_new %>% head

# ASSESSMENT: STRING PROCESSING PART 1

# Question 1

# Which of the following is NOT an application of string parsing?

# Removing unwanted characters from text
# Extracting numeric values from text
# Formatting numbers and characters so they can easily be displayed in deliverables like
# papers and presentations [X]
# Splitting strings into multiple values

# Question 2

# Which of the following commands would not give you an error?

# cat(" LeBron James is 6'8\" ") [X]
# cat(' LeBron James is 6'8" ')
# cat(` LeBron James is 6'8" `)
# cat(" LeBron James is 6\'8" ")

cat(" LeBron James is 6'8\" ")

# Question 3

# Which of the following are advantages of the stringr package over string processing functions
# in base R? Select all that apply.

# Base R functions are rarely used for string processing by data scientists so it's not worth learning
# them

# Functions in stringr all start with "str_", which makes them easy to look up using autocomplete [X]

# Stringr functions work better with pipes [X]

# The order of arguments is more consistent in stringr functions than in base R. [X]

# You have a data frame of monthly sales and profits in R:

dat <- data.frame(Month = c("January", "February", "March", "April", "May"),
                  Sales = c("$128,568", "$109,523", "$115,468", "$122,274", "$117,921"),
                  Profit = c("$16,234", "$12,876", "17,920", "$15,825", "$15,437"))
head(dat)

# Which of the following commands could convert the sales and profits columns to numeric? Select all
# that aplply.

# 1
dat %>% mutate_at(2:3, parse_number) # [X]

#2
dat %>% mutate_at(2:3, as.numeric) # error

# 3
dat %>% mutate_all(parse_number) # [X]

# 4

dat %>% mutate_at(2:3, funs(str_replace_all(., c("\\$|,"), ""))) %>%
  mutate_at(2:3, as.numeric) # [X]

# 3.2. STRING PROCESSING PART 2

# CASE STUDY 2: REPORTED HEIGHTS

# In the raw heights data, many students did not report their height as the number of inches as
# requested. There are many entries with real height information but in the wrong format, which
# we can extract with string processing.

# When there are both text and numeric entries in a column, the column will be a character vector. 
# Converting this column to numeric will result in NAs for some entries.

# To correct problematic entries, look for patterns that are shared across large numbers of entries,
# then define rules that identify those patterns and use these rules to write string processing
# tasks.

# Use suppressWarnings() to hide warning messages for a function.

# load raw heights data and inspect
library(dslabs)
data(reported_heights)
class(reported_heights$height)

# convert to numeric, inspect, count NAs
x <- as.numeric(reported_heights$height)
head(x)
sum(is.na(x))

# keep only entries that result in NAs
reported_heights %>% mutate(new_height = as.numeric(height)) %>%
  filter(is.na(new_height)) %>%
  head(n = 10)

# calculate cutoffs that cover 99.999% of human population
alpha <- 1/10^6
qnorm(1-alpha/2, 69.1, 2.9)
qnorm(alpha/2, 63.7, 2.7)

# keep only entries that either result in NAs or are outside the plausible range of heights
not_inches <- function(x, smallest = 50, tallest = 84){
  inches <- suppressWarnings(as.numeric(x))
  ind <- is.na(inches) | inches < smallest | inches > tallest
  ind
}

# number of problematic entries
problems <- reported_heights %>%
  filter(not_inches(height)) %>%
  .$height
length(problems)

# 10 examples of x'y or x'y" or x'y\"
pattern <- "^\\d\\s*\\d{1,2}\\.*\\d*'*\"*$"
str_subset(problems, pattern) %>% head(n = 10) %>% cat

# 10 examples of x.y or x,y
pattern <- "^[4-6]\\s*[\\.|,]\\s*([0-9]|10|11)$"
str_subset(problems, pattern) %>% head(n = 10) %>% cat

# 10 examples of entries in cm rather than inches
ind <- which(between(suppressWarnings(as.numeric(problems))/2.54, 54, 81))
ind <- ind[!is.na(ind)]
problems[ind] %>% head(n = 10) %>% cat

# REGEX

# A regular expression (regex) is a way to describe a specific pattern of characters of text. A
# set of rules has been designed to do this specifically and efficiently.

# stringr functions can take a regex as a pattern.

# str_detect() indicates whether a pattern is present in a string.

# The main difference between a regex and a regular string is that a regex can include special
# characters.

# The | symbol inside a regex means "or"

# Use '\\d' to represent digits. The backslash is used to distinguish it from the character 'd'. In R,
# you must use two backslashes for digits in regular expressions; in some other languages, you will only
# use one backslash for regex special characters.

# str_view() highlights the first occurrence of a pattern, and the str_view_all() function highlights all
# occurrences of the pattern.

# load stringr through the tidyverse
library(tidyverse)

# detect whether a comma is present
pattern <- ","
str_detect(murders_raw$population, pattern)

# show the subset of strings including "cm"
str_subset(reported_heights$height, "cm")

# use the "or" symbol inside a regex
yes <- c("180 cm", "70 inches")
no <- c("180", "70''")
s <- c(yes, no)
str_detect(s, "cm") | str_detect(s, "inches")
str_detect(s, "cm|inches")

# highlight the first occurrence of a pattern
str_view(s, pattern)

# highlight all instance of a pattern
str_view_all(s, pattern)

# CHARACTER CLASSES, ANCHORS AND QUANTIFIERS

# Define strings to test your regular expressions, including some elements that match and some that
# do not. This allows you to check for the two types of errors: failing to match and matching incorrectly.

# Square brackets define character classes: groups of characters that count as matching the pattern. You
# can use ranges to define character classes, such as [0-9] for digits and [a-zAZ] for all letters.

# Anchors define patterns that must start or end at specific places. ^ and $ represent the beginning and
# end of the string respectively.

# Curly braces are quantifiers that state how many times a certain character can be repeated in the
# pattern. \\d{1,2} matches exactly 1 or 2 consecutive digits.

# s was defined in the previous video
yes <- c("5", "6", "5'10", "5 feet", "4'11")
no <- c("", ".", "Five", "Six")
s <- c(yes, no)
pattern <- "\\d"

# [56] means 5 or 6
str_view(s, "[56]")

# [4-7] means 4, 5, 6 or 7
yes <- as.character(4:7)
no <- as.character(1:3)
s <- c(yes, no)
str_view(s, "[4-7]")

# ^ means start of string, $ means end of string
pattern <- "^\\d$"
yes <- c("1", "5", "9")
no <- c("12", "123", " 1", "a4", "b")
s <- c(yes, no)
str_view(s, pattern)

# curly braces define quantifiers: 1 or 2 digits
pattern <- "^\\d{1,2}$"
yes <- c("1", "5", "9", "12")
no <- c("123", "a4", "b")
str_view(c(yes,no), pattern)

# combining character class, anchors and quantifier
pattern <- "^[4-7]'\\d{1,2}\""
yes <- c("5'7\"", "6'2\"", "5'12\"")
no <- c("6,2\"", "6.2\"", "I am 5'11\"", "3'2\"", "64")
str_detect(yes, pattern)
str_detect(no, pattern)

# SEARCH AND REPLACE WITH REGEX

# str_replace() replaces the first instance of the detected pattern with a specified string.
# Spaces are characters and R does not ignore them. Spaces are specified by the special character \\s.
# Additional quantifiers include *, + and ?. * means 0 or more instances of the previous character,
# ? means 0 or 1 instances, and + means 1 or more instances.
# Before removing characters from strings with functions like str_replace() and str_replace_all(), 
# consider whether that replacement would have unintended effects.

# number of entries matching our desired pattern
pattern <- "^[4-7]'\\d{1,2}\"$"
sum(str_detect(problems, pattern))

# inspect examples of entries with problems
problems[c(2, 10, 11, 12, 15)] %>% str_view(pattern)
str_subset(problems, "inches")
str_subset(problems, "''")

# replace or remove feet/inches words before matching
pattern <- "^[4-7]'\d{1,2}$"
problems %>%
  str_replace("feet|ft|foot", "'") %>% # replace feet, ft, foot with '
  str_replace("inches|in|''|\"", "") %>% # remove all inches symbols
  str_detect(pattern) %>%
  sum()

# R does not ignore whitespace
identical("Hi", "Hi ")

# \\s represents whitespace
pattern_2 <- "^[4-7]'\\s\\d{1,2}\"$"
str_subset(problems, pattern_2)

# * means 0 or more instances of a character
yes <- c("AB", "A1B", "A11B", "A111B", "A1111B")
no <- c("A2B", "A21B")
str_detect(yes, "A1*B")
str_detect(no, "A1*B")

# test how *, ? and + differ
data.frame(string = c("AB", "A1B", "A11B", "A111B", "A1111B"),
           none_or_more = str_detect(yes, "A1*B"),
           none_or_once = str_detect(yes, "A1?B"),
           once_or_more = str_detect(yes, "A1+B"))

# update pattern by adding optional spaces before and after the feet symbol
pattern <- "^[4-7]\\s*'\\s*\\d{1,2}$"
problems %>%
  str_replace("feet|ft|foot", "'") %>% # replace feet, ft, foot with '
  str_replace("inches|in|''|\"", "") %>% # remove all inches symbols
  str_detect(pattern) %>%
  sum()

# GROUPS WITH REGEX

# Groups are defined using parentheses.
# Once we define groups, we can use the function str_match() to extract values these groups define.
# str_extract() extracts only strings that match a pattern, not the values defined by groups.
# You can refer to the ith group with \\i. For example, refer to the value in the second group with \\2.

# define regex with and without groups
pattern_without_groups <- "^[4-7],\\d*$"
pattern_with_groups <- "^([4-7]),(\\d*)$"

# create examples
yes <- c("5,9", "5,11", "6,", "6,1")
no <- c("5'9", ",", "2,8", "6.1.1")
s <- c(yes, no)

# demonstrate the effect of groups
str_detect(s, pattern_without_groups)
str_detect(s, pattern_with_groups)

# demonstrate the difference between str_match and str_extract
str_match(s, pattern_with_groups)
str_extract(s, pattern_with_groups)

# improve the pattern to recognize more events
pattern_with_groups <- "^([4-7]),(\\d*)$"
yes <- c("5,9", "5,11", "6,", "6,1")
no <- c("5'9", ",", "2,8", "6.1.1")
s <- c(yes, no)
str_replace(s, pattern_with_groups, "\\1'\\2")

# final pattern
pattern_with_groups <- "^([4-7])\\s*[,\\.\\s+]\\s*(\\d*)$"

# combine stringr commands with the pipe
str_subset(problems, pattern_with_groups) %>% head
str_subset(problems, pattern_with_groups) %>%
  str_replace(pattern_with_groups, "\\1'\\2") %>% head

# TESTING AND IMPROVING

# Wrangling with regular expressions is often an iterative process of testing the approach, looking for
# problematic entries, and improving the patterns.

# Use the pipe to connect stringr functions.

# It may not be worth writing code to correct every unique problem in the data, but string processing
# techniques are flexible enough for most needs.

# function to detect entries with problems
not_inches_or_cm <- function(x, smallest = 50, tallest = 84){
  inches <- suppressWarnings(as.numeric(x))
  ind <- !is.na(inches) &
    ((inches >= smallest & inches <= tallest) |
       (inches/2.54 >= smallest & inches/2.54 <= tallest))
  !ind
}

# identify entries with problems
problems <- reported_heights %>%
  filter(not_inches_or_cm(height)) %>%
  .$height
length(problems)

converted <- problems %>%
  str_replace("feet|ft|foot", "'") %>% # convert feet symbols to '
  str_replace("inches|in|''|\"", "") %>% # remove inches symbols
  str_replace("^([4-7])\\s*[,\\.\\s+]\\s*(\\d*)$", "\\1'\\2") # change format

# find proportion of entries that fit the pattern after reformatting
pattern <- "^[4-7]\\s*'\\s*\\d{1,2}$"
index <- str_detect(converted, pattern)
mean(index)

converted[!index] # show problems

# ASSESSMENT: STRING PROCESSING PART 2

# Question 1

# In the video, we use the function not_inches to identify heights that were incorrectly entered
not_inches <- function(x, smallest = 50, tallest = 84){
  inches <- suppressWarnings(as.numeric(x))
  ind <- is.na(inches) | inches < smallest | inches > tallest
  ind
}

# In this function, what TWO types of values are identified as not being correctly formatted in inches?

# Values that specifically contain apostrophes ('), periods (.) or quotations (")
# Values that result in NA's when converted to numeric [X]
# Values less than 50 inches or greater than 84 inches [X]
# Values that are stored as character class, because most are already classed as numeric

# Question 2

# Which of the following arguments, when passed to the function not_inches(), would return the vector
# c(FALSE)?

# c(175)
# c("5'8\"")
# c(70) [X]
# c(85) (the height of Shaquille O'Neal in inches)

# Question 3

# Our function not_inches() returns the object ind. Which answer correctly describes ind?

# ind is a logical vector of TRUE and FALSE, equal in length to vector x (in the arguments list).
# TRUE indicates that a height entry is incorrectly formatted. [X]

# ind is a logical vector of TRUE and FALSE, equal in length to vector x (in the arguments list).
# TRUE indicates that a height entry is correctly formatted.

# ind is a data frame like our reported_heights table but with an extra column of TRUE or FALSE.
# TRUE indicates that a height entry is incorrectly formatted.

# ind is a numeric vector equal to reported_heights$height but with incorrectly formatted heights
# replaced with NAs.

not_inches(c("65", "68", "48"))

# Question 4

# Given the following code
s <- c("70", "5 ft", "4'11", "", ".", "Six feet")
s

# What pattern vector yields the following result?

# `70`
# `5 ft`
# `4'11`
# .
# Six feet

# pattern <- "\\d|ft" [X]
# pattern <- "\d|ft"
# pattern <- "\\d\\d|ft"
# pattern <- "\\d|feet"

pattern <- "\\d|ft"
str_view_all(s, pattern)

# Question 5

# You enter the following set of commands into your R console. What is your printed result?
animals <- c("cat", "puppy", "Moose", "MONKEY")
pattern <- "[a-z]"
str_detect(animals, pattern)

# TRUE TRUE TRUE FALSE

# Question 6

# You enter the following set of commands into your R console. What is your printed result?
animals <- c("cat", "puppy", "Moose", "MONKEY")
pattern <- "[A-Z]$"
str_detect(animals, pattern)

# FALSE FALSE FALSE TRUE

# Question 7

# You enter the following set of commands into your R console. What is your printed result?
animals <- c("cat", "puppy", "Moose", "MONKEY")
pattern <- "[a-z]{4,5}"
str_detect(animals, pattern)

# FALSE TRUE TRUE FALSE

# Question 8

# Given the following code:
animals <- c("moose", "monkey", "meerkat", "mountain lion")

# Which TWO patterns would yield the following result?
str_detect(animals, pattern)
# TRUE TRUE TRUE TRUE

pattern <- "mo*" [X]
pattern <- "mo?" [X]
pattern <- "mo+"
pattern <- "moo*"

# Question 9

# You are working on some data from different universities. You have the following vector:
schools <- c("U. Kentucky", "Univ New Hampshire", "Univ. of Massachusetts", "U California",
             "California State University")

# You want to clean this data to match the full names of each university.

# Which one of the following commands could accomplish this?

# 1
schools %>%
  str_replace("Univ\\.?|U\\.?", "University ") %>%
  str_replace("^University of |^University ", "University of ")

# 2
schools %>%
  str_replace("^Univ\\.?\\s|^U\\.?\\s", "University ") %>%
  str_replace("^University of |^University ", "University of ") # [X]

# 3
schools %>%
  str_replace("Univ\\.?\\s|^U\\.?\\s", "University") %>%
  str_replace("University ", "University of ")

# 4
schools %>%
  str_replace("^Univ\\.?\\s|^U\\.?\\s", "University") %>%
  str_replace("University ", "University of ")

# Question 10

# Rather than using the pattern_with_groups vector from the video, you accidentally write in the
# following code:
problems <- c("5.3", "5,5", "6 1", "5 .11", "5, 12")
pattern_with_groups <- "^([4-7])[,\\.](\\d*)$"
str_replace(problems, pattern_with_groups, "\\1'\\2")

# What is your result?

# [1] "5'3" "5'5" "6 1" "5 .11" "5, 12" [X]
# [1] "5.3" "5,5" "6 1" "5 .11" "5, 12"
# [1] "5'3" "5'5" "6'1" "5. 11" "5, 12"
# [1] "5'3" "5'5" "6'1" "5'11" "5'12"

# Question 11

# You notice your mistake and correct your pattern regex to the following
problems <- c("5.3", "5,5", "6 1", "5 .11", "5, 12")
pattern_with_groups <- "^([4-7])[,\\.\\s](\\d*)$"
str_replace(problems, pattern_with_groups, "\\1'\\2")

# [1] "5'3" "5'5" "6'1" "5 .11" "5, 12"

# Question 12

# In our example, we use the following code to detect height entries that do not match our pattern of x'y":
converted <- problems %>%
  str_replace("feet|foot|ft", "'") %>%
  str_replace("inches|in|''|\"", "") %>%
  str_replace("^([4-7])\\s*[,\\.\\s+]\\s*(\\d*)$", "\\1'\\2")

pattern <- "^[4-7]\\s*'\\s*\\d{1,2}$"
index <- str_detect(converted, pattern)
converted[!index]

# Which answer best describes the differences between the regex string we use as an argument in
# str_replace("^([4-7])\\s*[,\\.\\s+]s*(\\d*)$", "\\1'\\2") and the regex string in
# pattern <- "^[4-7]\\s*'\\s*\\d{1,2}$"?

# The regex used in str_replace() looks for either a comma, period or space between the feet and inches
# digits, while the pattern regex just looks for an apostrophe; the regex in str_replace() allows for
# one or more digits to be entered as inches, while the pattern regex only allows for one or two digits.

# The regex used in str_replace() allows for additional spaces between the feet and inches digits, but
# the pattern regex does not.

# The regex used in str_replace() looks for either a comma, period or space between the feet and inches
# digits, while the pattern regex just looks for an apostrophe; the regex in str_replace allows none or
# more digits to be entered as inches, while the pattern regex only allows for the number 1 or 2 to be used.

# The regex used in str_replace() looks for either a comma, period or space between the feet and inches
# digits, while the pattern regex just looks for an apostrophe; the regex in str_replace() allows for none
# or more digits to be entered as inches, while the pattern regex only allows for one or two digits.[X]

# Question 13

# You notice a few entries that are not being properly converted using your str_replace() and str_detect()
# code:
yes <- c("5 feet 7inches", "5 7")
no <- c("5ft 9 inches", "5 ft 9 inches")
s <- c(yes, no)

converted <- s %>%
  str_replace("feet|foot|ft", "'") %>%
  str_replace("inches|in|''|\"", "") %>%
  str_replace("^([4-7])\\s*[,\\.\\s+]\\s*(\\d*)$", "\\1'\\2")

pattern <- "^[4-7]\\s*'\\s*\\d{1,2}$"
str_detect(converted, pattern)
# [1] TRUE TRUE FALSE FALSE

# It seems like the problem may be due to spaces around the words feet|foot|ft and inches|in. What
# is another way you could fix this problem?

# Correct response
converted <- s %>%
  str_replace("\\s*(feet|foot|ft)\\s*", "'") %>%
  str_replace("\\s*(inches|in|''|\")\\s*", "") %>%
  str_replace("^([4-7])\\s*[,\\.\\s+]\\s*(\\d*)$", "\\1'\\2")

# 3.3. STRING PROCESSING PART 3

# SEPARATE WITH REGEX

# The extract() function behaves similarly to the separate() function but allows extraction of groups
# from regular expressions.

# first example - normally formatted heights
s <- c("5'10", "6'1")
tab <- data.frame(x = s)

# the separate and extract functions behave similarly
tab %>% separate(x, c("feet", "inches"), sep = "'")
tab %>% extract(x, c("feet", "inches"), regex = "(\\d)'(\\d{1,2})")

# second example - some heights with unusual formats
s <- c("5'10", "6'1\"", "5'8inches")
tab <- data.frame(x = s)

# separate fails because it leaves in extra characters, but extract keeps only the digits because
# of regex groups
tab %>% separate(x, c("feet", "inches"), sep = "'", fill = "right")
tab %>% extract(x, c("feet", "inches"), regex = "(\\d)'(\\d{1,2})")

# USING GROUPS AND QUANTIFIERS

# Four clear patterns of entries have arisen along with some other minor problems:

# 1. Many students measuring exactly 5 or 6 feet did not enter any inches. For example, 6' - our
# pattern requires that inches be included.
# 2. Some students measuring exactly 5 or 6 feet entered just that number.
# 3. Some of the inches were entered with decimal points. For example, 5'7.5''. Our pattern only
# looks for two digits.
# 4. Some entries have spaces at the end, for example 5 ' 9.
# 5. Some entries are in meters and some of these use European decimals: 1.6, 1,7.
# 6. Two students added cm.
# 7. One student spelled out the numbers: Five foot eight inches.

# It is not necessarily clear that it is worth writing code to handle all these cases since they
# might be rare enough. However, some give us an opportunity to learn some more regex techniques
# so we will build a fix.

# Case 1
# For case 1, if we add a '0 to, for example, convert all 6 to a 6'0, then our pattern will match.
# This can be done using groups using the following code:
yes <- c("5", "6", "5")
no <- c("5'", "5''", "5'4")
s <- c(yes, no)
str_replace(s, "^([4-7])$", "\\1'0")

# The pattern says it has to start (^), be followed with a digit between 4 and 7, and then end there ($).
# The parenthesis defines the group that we pass as \\1 to the replace regex.

# Cases 2 and 4
# We can adapt this code slightly to handle case 2 as well which covers the entry 5'. Note that the 5' is
# left untouched by the code above. This is because the extra ' makes the pattern not match since we have 
# to end with a 5 or 6. To handle case 2, we want to permit the 5 or 6 to be followed by no or one symbol
# for feet. So we can simply add '{0,1} after the ' to do this. We can also use the none or once special
# character ?. As we saw previously, this is different from * which is none or more. We now see that this
# code also handles the fourth case as well.
str_replace(s, "^([56])'?$", "\\1'0")

# Note that here we only permit 5 and 6 but not 4 and 7. This is because heights of exactly 5 and exactly
# 6 feet tall are quite common, so we assume those that typed 5 or 6 really meant either 60 or 72 inches.
# However, heights of exactly 4 or exactly 7 feet tall are so rare that, although we accept 84 as a valid
# entry, we assume that a 7 was entered in error.

# Case 3
# We can use quantifiers to deal with case 3. These entries are not matched because the inches include
# decimals and our pattern does not permit this. We need allow the second group to include decimals and
# not just digits. This means we must permit zero or one period . followed by zero or more digits. So
# we will use both ? and *. Also remember that for this particular case, the period needs to be escaped
# since it is a special character (it means any character except a line break).

# So we can adapt our pattern, currently "^[4-7]\\s*'\\s*\\d{1,2}$, to permit a decimal at the end:
pattern <- "^[4-7]\\s*'\\s*(\\d+\\.?\\d*)$"

# Case 5
# Case 5, meters using commas, we can approach similarly to how we converted the x.y to x'y. A difference
# is that we require that the first digit is 1 or 2:
yes <- c("1,7", "1, 8", "2, ")
no <- c("5,8", "5,3,2", "1.7")
s <- c(yes, no)
str_replace(s, "^([1,2])\\s*,\\s*(\\d*)$", "\\1\\.\\2")

# We will later check if the entries are meters using their numeric values.

# TRIMMING

# In general, spaces at the start or end of the string are uninformative. These can be particularly
# deceptive because sometimes they can be hard to see.
s <- "Hi "
cat(s)
identical(s, "Hi")

# This is a general enough problem that there is a function dedicated to removing them: str_trim.
str_trim("5 ' 9 ")

# TO UPPER AND TO LOWER CASE

# One of the entries writes out numbers as words: Five foot eight inches. Although not efficient, we
# could add 12 extra str_replace to convert zero to 0, one to 1, and so on. To avoid having to write
# two separate operations for Zero and zero, One and one, etc., we can use the str_to_lower() function
# to make all words lower case first:
s <- c("Five feet eight inches")
str_to_lower(s)

# PUTTING IT INTO A FUNCTION

# We are now ready to define a procedure that handles converting all the problematic cases.

# We can now put all this together into a function that takes a string vector and tries to convert
# as many strings as possible to a single format. Below is a function that puts together the previous
# code replacements:
convert_format <- function(s){
  s %>%
    str_replace("feet|foot|ft", "'") %>% # convert feet symbols to '
    str_replace_all("inches|in|''|\"|cm|and", "") %>% # removes inches and other symbols
    str_replace("^([4-7])\\s*[,\\.\\s+]\\s*(\\d*)$", "\\1'\\2") %>% # change x.y, x,y, x y
    str_replace("^([56])'?$", "\\1'0") %>% # add 0 when 5 or 6
    str_replace("^([12])\\s*,\\s*(\\d*)$", "\\1.\\2") %>% # change european decimal
    str_trim() # remove extra space
}

# We can also write a function that coverts words to numbers:
words_to_numbers <- function(s){
  str_to_lower(s) %>%
    str_replace_all("zero", "0") %>%
    str_replace_all("one", "1") %>%
    str_replace_all("two", "2") %>%
    str_replace_all("three", "3") %>%
    str_replace_all("four", "4") %>%
    str_replace_all("five", "5") %>%
    str_replace_all("six", "6") %>%
    str_replace_all("seven", "7") %>%
    str_replace_all("eight", "8") %>%
    str_replace_all("nine", "9") %>%
    str_replace_all("ten", "10") %>%
    str_replace_all("eleven", "11")
}

# Now we can see which problematic entries remain:
converted <- problems %>% words_to_numbers %>% convert_format
remaining_problems <- converted[not_inches_or_cm(converted)]
pattern <- "^[4-7]\\s*'\\s*\\d+\\.?\\d*$"
index <- str_detect(remaining_problems, pattern)
remaining_problems[!index]

# PUTTING IT ALL TOGETHER

# We are not ready to put everything we've done so far together and wrangle our reported heights data as
# we try to recover as many heights as possible. The code is complex but we will break it down into parts.

# We start by cleaning up the height column so that the heights are closer to a feet'inches format. We
# added an original heights column so we can compare before and after.

# Let's start by writing a function that cleans up strings so that all the feet and inches formats use
# the same x'y format when appropriate.
pattern <- "^([4-7])\\s*'\\s*(\\d+\\.?\\d*)$"
smallest <- 50
tallest <- 84
new_heights <- reported_heights %>% 
  mutate(original = height, 
         height = words_to_numbers(height) %>% convert_format()) %>%
  extract(height, c("feet", "inches"), regex = pattern, remove = FALSE) %>%
  mutate_at(c("height", "feet", "inches"), as.numeric) %>%
  mutate(guess = 12*feet + inches) %>%
  mutate(height = case_when(
    !is.na(height) & between(height, smallest, tallest) ~ height, #inches 
    !is.na(height) & between(height/2.54, smallest, tallest) ~ height/2.54, #centimeters
    !is.na(height) & between(height*100/2.54, smallest, tallest) ~ height*100/2.54, #meters
    !is.na(guess) & inches < 12 & between(guess, smallest, tallest) ~ guess, #feet'inches
    TRUE ~ as.numeric(NA)))

new_heights <- new_heights %>% select(-guess)

# We can check all the entries we converted using the following code
new_heights %>%
  filter(not_inches(original)) %>%
  select(original, height) %>%
  arrange(height) %>%
  view()

# Let's take a look at the shortest students in our dataset using the following code:
new_heights %>% arrange(height) %>% head(n = 7)

# We see heights of 53, 54, and 55. In the original heights column, we also have 51 and 52. These
# short heights are very rare and it is likely that the students actually meant 5'1, 5'2, 5'3, 5'4,
# and 5'5. But because we are not completely sure, we will leave them as reported.

# STRING SPLITTING

# The function str_split() splits a string insto a character vector on a delimiter,
# such as a comma, space, or underscore. By default, str_split() generates a list
# with one element for each original string. Use the function argument
# simplify = TRUE to have str_split() return a matrix instead.

# The map() function from the purrr package applies the same function to each
# element of a list. To extract the ith entry of each element x, use map(x, i).

# map() always returns a list. Use map_chr() to return a character vector and
# map_int() to return an integer.

# read raw murders data line by line
filename <- system.file("extdata/murders.csv", package = "dslabs")
lines <- readLines(filename)
lines %>% head()

# split at commas with str_split function, remove row of column names
x <- str_split(lines, ",")
x %>% head()
col_names <- x[[1]]
x <- x[-1]

# extract first element of each list entry
library(purrr)
map(x, function(y) y[1]) %>% head()
map(x, 1) %>% head()

# extract columns 1-5 as characters, then convert to proper format
dat <- data.frame(parse_guess(map_chr(x, 1)),
                  parse_guess(map_chr(x, 2)),
                  parse_guess(map_chr(x, 3)),
                  parse_guess(map_chr(x, 4)),
                  parse_guess(map_chr(x, 5))) %>%
  setNames(col_names)
dat %>% head

# more efficient code for the same thing
dat <- x %>%
  transpose() %>%
  map( ~ parse_guess(unlist(.))) %>%
  setNames(col_names) %>%
  as.data.frame()

# the simplify argument makes str_split return a matrix instead of a list
x <- str_split(lines, ",", simplify = TRUE)
col_names <- x[1,]
x <- x[-1,]
x %>% as.data.frame() %>%
  setNames(col_names) %>%
  mutate_all(parse_guess)

# CASE STUDY: EXTRACTING A TABLE FROM A PDF

# One of the datasets provided in dslabs shows scientific funding rates by
# gender in the Netherlands
library(dslabs)
data("research_funding_rates")
research_funding_rates

# The data come from a paper published in the prestigious journal PNAS. However,
# the data are not provided in a spreadsheet; they are in a table in a PDF document.
# We could extract the numbers by hand, but this could lead to human error. Instead,
# we can try to wrangle the data using R.

# Downloading the data

# We start by downloading the PDF document then importing it into R using the
# following code:
library(pdftools)
temp_file <- tempfile()
url <- "http://www.pnas.org/content/suppl/2015/09/16/1510159112.DCSupplemental/pnas.201510159SI.pdf"
download.file(url, temp_file)
txt <- pdf_text(temp_file)
file.remove(temp_file)

# If we examine the object text we noteice that it is a character vector with an
# entry for each page. So we keep the page we want using the following code:
raw_data_research_funding_rates <- txt[2]

# The steps above can actually be skipped because we include the raw data in the
# dslabs package as well:
data("raw_data_research_funding_rates")

# Looking at the download

# Examining this object,
raw_data_research_funding_rates %>% head

# we see that it is a long string. Each line on the page, including the table rows,
# is separated by the symbol for newline: \n.

# We can therefore create a list with the lines of the text as elements:
tab <- str_split(raw_data_research_funding_rates, "\n")

# Because we start off with just one element in the string, we end up with a list
# with just one entry:
tab <- tab[[1]]

# By examining this object,
tab %>% head

# we see that the information for the column names is the third and fourth entries:
the_names_1 <- tab[3]
the_names_2 <- tab[4]

# In the table, the column information is spread across two lines. We want to create
# one vector with one name for each column. We can do this using some of the
# functions we have just learned.

# Extracting the table data

# Let's start with the first line:
the_names_1

# We want to remove the leading space and everything following the comma. We can
# use regex for the latter. Then we can obtain the elements by splitting using the
# space. We want to split only when there are 2 or more spaces to avoid splitting
# success rate. So we use the regex \\s{2,} as follows:
the_names_1 <- the_names_1 %>%
  str_trim() %>%
  str_replace_all(",\\s.", "") %>%
  str_split("\\s{2,}", simplify = TRUE)
the_names_1

# Now let's look at the second line:
the_names_2

# Here we want to trim the leading space and then split by space as we did for
# the first line:
the_names_2 <- the_names_2 %>%
  str_trim() %>%
  str_split("\\s+", simplify = TRUE)
the_names_2

# Now we can join these to generate one name for each column:
tmp_names <- str_c(rep(the_names_1, each = 3), the_names_2[-1], sep = "_")
the_names <- c(the_names_2[1], tmp_names) %>%
  str_to_lower() %>%
  str_replace_all("\\s", "_")
the_names

# Now we are ready to get the actual data. By examining the tab object, we
# notice that the information is in lines 6 through 14. We can use str_split()
# again to achieve our goal:
new_research_funding_rates <- tab[6:14] %>%
  str_trim %>%
  str_split("\\s{2,}", simplify = TRUE) %>%
  data.frame(stringsAsFactors = FALSE) %>%
  setNames(the_names) %>%
  mutate_at(-1, parse_number)
new_research_funding_rates %>% head()

# We can see that the objects are identical
identical(research_funding_rates, new_research_funding_rates)

# RECODING

# Change long factor names with the recode() function from the tidyverse.

# Other similar functions include recode_factor() and fct_recoder() in the
# forcats package in the tidyverse. The same result could be obtained using the
# case_when() function, but recode() is more efficient to write.

# life expectancy time series for Caribbean countries
library(dslabs)
data("gapminder")
gapminder %>%
  filter(region == "Caribbean") %>%
  ggplot(aes(year, life_expectancy, color = country)) +
  geom_line()

# display long country names
gapminder %>%
  filter(region == "Caribbean") %>%
  filter(str_length(country) >= 12) %>%
  distinct(country)

# recode long country names and remake plot
gapminder %>% filter(region == "Caribbean") %>%
  mutate(country = recode(country,
                          'Antigua and Barbuda' = "Antigua",
                          'Dominican Republic' = "DR",
                          'St. Vincent and the Grenadines' = "St. Vincent",
                          'Trinidad and Tobago' = "Trinidad")) %>%
  ggplot(aes(year, life_expectancy, color = country)) +
  geom_line()

# ASSESSMENT PART 1: STRING PROCESSING PART 3

# In this part of the assessment, you will answer several multiple choice
# questions that review the concepts of string processing. You can answer
# these questions without using R, although you may find it helpful to
# experiment with commands in your console.

# In the second part of the assessment, you will import a real dataset and
# use string processing to clean it for analysis. This will require you to
# write code in R.

# Question 2

# You have the following table, schedule:
schedule <- data.frame(day = c("Monday", "Tuesday"),
                       staff = c("Mandy, Chris and Laura",
                                 "Steve, Ruth and Frank"))
schedule

# You want to turn this into a more useful data frame.

# Which two commands would properly split the text in the "staff" column into
# each individual name? Select ALL that apply

# 1
str_split(schedule$staff, ",|and")

# 2
str_split(schedule$staff, ", | and ") # [X]

# 3
str_split(schedule$staff, ",\\s|\\sand\\s") # [X]

# 4
str_split(schedule$staff, "\\s?(,|and)\\s?")

# Question 3

# You have the following table, schedule:
schedule

# What code would successfully turn your "Schedule" table into the following
# tidy table?

# 1 [X]
tidy <- schedule %>%
  mutate(staff = str_split(staff, ", | and ")) %>%
  unnest(c(staff))

# 2
tidy <- separate(schedule, staff, into = c("s1", "s2", "s3"), sep = ",") %>%
  gather(key = s, value = staff, s1:s3)

# 3
tidy <- schedule %>%
  mutate(staff = str_split(staff, ", | and ", simplify = TRUE)) %>%
  unnest(c())

# Question 4

# Using the gapminder data, you want to recode countries longer than 12 letters
# in the region "Middle Africa" to their abbreviations in a new column,
# "country_short". Which code would accomplish this?

# 1
dat <- gapminder %>% filter(region == "Middle Africa") %>%
  mutate(recode(country,
                "Central African Republic" = "CAR",
                "Congo, Dem. Rep." = "DRC",
                "Equatorial Guinea" = "Eq. Guinea"))

# 2
dat <- gapminder %>% filter(region == "Middle Africa") %>%
  mutate(country_short = recode(country,
                                c("Central African Republic", "Congo, Dem. Rep.",
                                  "Equatorial Guinea"),
                                c("CAR", "DRC", "Eq. Guinea")))

# 3
dat <- gapminder %>% filter(region == "Middle Africa") %>%
  mutate(country = recode(country,
                          "Central African Republic" = "CAR",
                          "Congo, Dem. Rep." = "DRC",
                          "Equatorial Guinea" = "Eq. Guinea"))

# 4 [X]
dat <- gapminder %>% filter(region == "Middle Africa") %>%
  mutate(country_short = recode(country,
                                "Central African Republican" = "CAR",
                                "Congo, Dem. Rep." = "DRC",
                                "Equatorial Guinea" = "Eq. Guinea"))

gapminder %>% filter(region == "Middle Africa") %>%
  filter(str_length(country) >= 12) %>% 
  mutate(country_short = recode(country,
                                "Central African Republic" = "CAR",
                                "Congo, Dem. Rep." = "DRC",
                                "Equatorial Guinea" = "Eq. Guinea")) %>%
  distinct(country, country_short)

# ASSESSMENT PART 2: STRING PROCESSING PART 3

# Import a raw Brexit referendum polling data from Wikipedia
library(rvest)
library(tidyverse)
library(stringr)
url <- "https://en.wikipedia.org/w/index.php?title=Opinion_polling_for_the_United_Kingdom_European_Union_membership_referendum&oldid=896735054"
tab <- read_html(url) %>% html_nodes("table") 
polls <- tab[[6]] %>% html_table(fill = TRUE)

# You will use a variety of string processing techniques learned in this section
# to reformat these data.

# Question 5

# Some rows in this table do not contain polls. You can identify these by the lack
# of the percent sign (%) in the Remain column.

# Update polls by changing the column names to c("dates", "remain", "leave",
# "undecided", "lead", "samplesize", "pollster", "poll_type", "notes") and only
# keeping rows that have a percent sign (%) in the remain column.

# How many rows remain in the polls data frame?

polls <- polls %>%
  filter(str_detect(Remain, "%")) %>%
  setNames(c("dates", "remain", "leave", "undecided", "lead", "samplesize", "pollster", "poll_type", "notes"))
nrow(polls) # 129

# Question 6

# The remain and leave columns are both given in the format "48.1%": percentages
# out of 100% with a percent symbol.

# Which of these commands converts the remain vector to a proportion between
# 0 and 1?

# 1
as.numeric(str_remove(polls$Remain, "%"))

# 2
as.numeric(polls$Remain) / 100

# 3
parse_number(polls$Remain)

# 4
str_remove(polls$Remain, "%") / 100

# 5 [X]
as.numeric(str_replace(polls$Remain, "%", "")) / 100

# 6 [X]
parse_number(polls$Remain) / 100

# Question 7

# The undecided column has some "N/A" values. These "N/A"s are only present
# when the remain and leave columns total 100%, so they should actually be zeros.

# Use a function from stringr to convert "N/A" in the undecided column to 0. The
# format of your command should be function_name(polls$undecided, "arg1", "arg2").

# What function replaces function_name?

# What argument replaces arg1?

# What argument replaces arg2?

str_replace(polls$Undecided, "N/A", "0")

# Question 8

# The dates column contains the range of dates over which the poll was conducted.
# The format is "8-10 Jan" where the poll had a start date of 2016-01-08 and end
# date of 2016-01-10. Some polls go across month boundaries (16 May - 12 June).

# The end data of the poll will always be one or two digits, followed by a space,
# followed by the month as one or more letters (either capital or lowercase). In
# these data, all month abbreviations or names have 3, 4 or 5 letters.

# Write a regular expression to extract the end day and month from dates. Insert
# it into the skeleton code bellow:

# temp <- str_extract_all(polls$dates, # regex #)
# end_date <- sapply(temp, function(x) x[length(x)]) # take last element (handles 
# polls that cross month boundaries))

# Which of the following regular expressions correctly extracts the end day and
# month when inserted into the blank # regex # code above?

# "\\d?\\s[a-zA-Z]?"
# "\\d+\\s[a-zA-Z]+" [X]
# "\\d+\\s[A-Z]+"
# "[0-9]+\\s[a-zA-Z]+"
# "\\d{1,2}\\s[a-zA-Z]+" [X]
# "\\d{1,2}[a-zA-Z]+"
# "\\d+\\s[a-zA-Z]{3,5} [X]

temp <- str_extract_all(polls$dates, "[0-9]+\\s[a-zA-Z]+")
end_date <- sapply(temp, function(x) x[length(x)])

## SECTION 4: DATES, TIMES, AND TEXT MINING

# In the Dates, Times, and Text Mining section, you will learn how to deal with
# dates and times in R and also how to generate numerical summaries from text
# data.

# Handle dates and times in R
# Use the lubridate package to parse dates and times in different formats
# Generate numerical summaries from text data and apply data visualization
# and analysis techniques to those data.

# 4.1. DATES, TIMES, AND TEXT MINING

# DATES AND TIMES

# Dates are a separate data type in R. The tidyverse includes functionality
# for dealing with dates through the lubridate package.

# Extract the year, month and day from a date object with the year(), month(),
# and day() functions.

# Parsers convert strings into dates with the standard YYY-MM-DD format (ISO
# 8601). Use the parser with the name corresponding to the string format of
# year, month, and day (ymd(), ydm(), myd(), mdy(), dmy(), dym()).

# Get the current time with the Sys.time() function. Use the now() function
# instead to specify a time zone.

# You can extract values from time objects with the hour(), minute() and second()
# functions.

# Parsers convert strings into times (for example, hms()). Parsers can also create
# combined date-time objects (for example, mdy_hms()).

# inspect the startdate column of 2016 polls data, a Date type
library(tidyverse)
library(dslabs)
data("polls_us_election_2016")
polls_us_election_2016 %>% head
class(polls_us_election_2016$startdate)
as.numeric(polls_us_election_2016$startdate) %>% head

# ggplot is aware of dates
polls_us_election_2016 %>% filter(pollster == "Ipsos" & state == "U.S.") %>%
  ggplot(aes(startdate, rawpoll_trump)) +
  geom_line()

# lubridate: the tidyverse date package
library(lubridate)

# select some randome dates from polls
set.seed(2)
dates <- sample(polls_us_election_2016$startdate, 10) %>% sort
dates

# extract month, day, year from date strings
data.frame(date = dates,
           month = month(dates),
           day = day(dates),
           year = year(dates))

month(dates, label = TRUE) # extract month label

# ymd works on mixed date styles
x <- c(20090101, "2009-01-02", "2009 01 03", "2009-1-4",
       "2009-1, 5", "Created on 2009 1 6", "200901 !!! 07")
ymd(x)

# different parsers extract year, month and day in different orders
x <- "09/01/02"
ymd(x)
mdy(x)
ydm(x)
myd(x)
dmy(x)
dym(x)

now() # current time in your time zone
now("GMT") # current time in GMT
now() %>% hour() # current hour
now() %>% minute() # current minute
now() %>% second() # current second

# parse time
x <- c("12:34:56")
hms(x)

# parse datetime
x <- "Nov/2/2012 12:34:56"
mdy_hms(x)

# TEXT MINING

# The tidytext package helps us convert free form text into a tidy table.
# Use unnest_tokens() to extract individual words and other meaningful chunks
# of text.
# Sentiment analysis assigns emotions or a positive/negative score to tokens.
# You can extract sentiments using get_sentiments(). Common lexicons for
# sentiment analysis are "bing", "afinn", "nrc", and "loughran".

# With the exception of labels used to represent categorical data, we have
# focused on numerical data, but in many applications data starts as text. Well
# known examples are spam filtering, cyber-crime prevention, counter-terrorism
# and sentiment analysis.

# In all these examples, the raw data is composed of free form texts. Our task
# is to extract insights from these data. In this section, we learn how to
# generate useful numerical summaries from text data to which we can apply some
# of the powerful data visualization and analysis techniques we have learned.

# CASE STUDY: TRUMP TWEETS

# During the 2016 US presidential election, then-candidate Donald J. Trump used
# his Twitter account as a way to communicate with potential voters. On August 6,
# 2016 Todd Vaziri tweeted about Trump that "Every non-hyperbolic tweet is from
# iPhone (his staff). Every hyperbolic tweet is from Android (from him)." Data
# scientist David Robinson conducted an analysis to determine if data supported
# this assertion. Here we go through David's analysis to learn some of the basics
# of text mining.

# We will use the following libraries:
library(tidyverse)
library(ggplot2)
library(lubridate)
library(tidyr)
library(scales)
set.seed(1)

# In general, we can extract data directly from Twitter using the rtweet package.
# However, in this case, a group has already compiled data for us and made it
# available here: https://www.thetrumparchive.com/

url <- 'https://drive.google.com/file/d/16wm-2NTKohhcA26w-kaWfhLIGwl_oX95/view'
trump_tweets <- map(2009:2017, ~sprintf(url, .x)) %>%
  map_df(jsonlite::fromJSON, simplifyDataFrame = TRUE) %>%
  filter(!is_retweet & !str_detect(text, '^"')) %>%
  mutate(created_at = parse_date_time(created_at, orders = "a b! d! H!:M!:S! z!* Y!", tz="EST"))

# For convenience we include the result of the code above in the dslabs package:
library(dslabs)
data("trump_tweets")

# This is a data frame with information about the tweet:
head(trump_tweets)

# The variables that are included are:
names(trump_tweets)

# The help file ?trump_tweets provides details on what each variable represents.
# The tweets are represented by the text variable:
trump_tweets %>% select(text) %>% head

# and the source variable tells us the device that was used to compose and
# upload each tweet:
trump_tweets %>% count(source) %>% arrange(desc(n))

# We can extract to remove the Twitter for part of the source and filter out
# retweets:
trump_tweets %>%
  extract(source, "source", "Twitter for (.*)") %>%
  count(source)

# We are interested in what happened during the campaign, so for the analysis
# here we will focus on what was tweeted between the day Trump announced his
# campaign and election day. So we define the following table:
campaign_tweets <- trump_tweets %>%
  extract(source, "source", "Twitter for (.*)") %>%
  filter(source %in% c("Android", "iPhone") &
           created_at >= ymd("2015-06-17") &
           created_at <= ymd("2016-11-08")) %>%
  filter(!is_retweet) %>%
  arrange(created_at)

# We can now use data visualization to explore the possibility that two different
# groups were tweeting from these devices. For each tweet, we will extract the hour,
# in the east coast (EST), it was tweeted then compute the proportion of tweets
# tweeted at each hour for each device.
ds_theme_set()
campaign_tweets %>%
  mutate(hour = hour(with_tz(created_at, "EST"))) %>%
  count(source, hour) %>%
  group_by(source) %>%
  mutate(percent = n / sum(n)) %>%
  ungroup %>%
  ggplot(aes(hour, percent, color = source)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(labels = percent_format()) +
  labs(x = "Hour of day (EST)",
       y = "% of tweets",
       color = "")

# We notice a big peak for the Android in early hours of the morning, between
# 6 and 8 AM. There seems to be a clear difference in these patterns. We will
# therefore assume that two different entities are using these two devices.
# Now we will study how their tweets differ. To do this we introduce the
# tidytext package.

# TEXT AS DATA

# The tidytext package helps us convert free form text into a tidy table. Having
# the data in this format greatly facilitates data visualization and applying
# statistical techniques.
library(tidytext)

# The main function needed to achieve this is unnest_tokens(). A token refers to
# the units that we are considering to be a data point. The most common tokens
# will be words, but they can also be single characters, ngrams, sentences, lines,
# or a pattern defined by regex. The functions will take a vector of strings and
# extract the tokens so that each one gets a row in the new table. Here is a simple
# example:
example <- data_frame(line = c(1, 2, 3, 4),
                      text = c("Roses are red,", "Violets are blue,", "Sugar is sweet,", "And so are you."))
example
example %>% unnest_tokens(word, text)

# Now let's look at a quick example with a tweet number 3008:
i <- 3008
campaign_tweets$text[i]
campaign_tweets[i,] %>%
  unnest_tokens(word, text) %>%
  select(word)

# Note that the function tries to convert tokens into words and strips characters
# important to twitter such as # and @. A token in twitter is not the same as in
# the regular English language. For this reason, instead of using the default token,
# words, we define a regex that captures twitter characters. The pattern appears
# complex but all we are defining is a pattern that starts with @, # or neither
# and is followed by any combination of letters or digits:
pattern <- "([^A-Za-z\\d#@']|'(?![A-Za-z\\d#@]))"

# We can now use the unnest_tokens() function with the regex option and approximately
# extract the hashtags and mentions:
campaign_tweets[i,] %>%
  unnest_tokens(word, text, token = "regex", pattern = pattern) %>%
  select(word)

# Another minor adjustment we want to make is remove the links to pictures:
campaign_tweets[i,] %>%
  mutate(text = str_replace_all(text, "https://t.co/[A-Za-z\\d+]+|&amp;", "")) %>%
  unnest_tokens(word, text, token = "regex", pattern = pattern) %>%
  select(word)

# Now we are ready to extract the words for all our tweets.
tweet_words <- campaign_tweets %>%
  mutate(text = str_replace_all(text, "https://t.co/[A-Za-z\\d]+|&amp;", "")) %>%
  unnest_tokens(word, text, token = "regex", pattern = pattern)

# And we can now answer questions such as "what are the most commonly used words?"

tweet_words %>%
  count(word) %>%
  arrange(desc(n))

# It is not surprising that these are the top words. The top words are not
# informative. The tidytext package has a database of these commonly used words,
# referred to as stop words, in text mining:
stop_words

# If we filter out rows representing stop words with filter(!word %in%
# stop_words$word):
tweet_words <- campaign_tweets %>%
  mutate(text = str_replace_all(text, "https://t.co/[A-Za-z\\d]+|&amp;", "")) %>%
  unnest_tokens(word, text, token = "regex", pattern = pattern) %>%
  filter(!word %in% stop_words$word)

# We end up with a much more informative set of top 10 tweeted words:
tweet_words %>%
  count(word) %>%
  top_n(10, n) %>%
  mutate(word = reorder(word, n)) %>%
  arrange(desc(n))

# Some exploration of the resulting words reveals a couple of unwanted
# characteristics in our tokens. First, some of our tokens are just numbers
# (years, for example). We want to remove these and we can find them using
# the regex ^\d+$. Second, some of our tokens come from a quote and they start
# with '. We want to remove then ' when it's at the start of a word, so we will
# use str_replace(). We add these two lines to the code above to generate our
# final table:
tweet_words <- campaign_tweets %>%
  mutate(text = str_replace_all(text, "https://t.co/[A-Za-z\\d]+|&amp;", "")) %>%
  unnest_tokens(word, text, token = "regex", pattern = pattern) %>%
  filter(!word %in% stop_words$word &
           !str_detect(word, "^\\d+$")) %>%
  mutate(word = str_replace(word, "^'", ""))

# Now that we have all our words in a table, along with information about what
# device was used to compose the tweet they came from, we can start exploring
# which words are more common when comparing Android to iPhone.

# For each word we want to know if it is more likely to come from an Android
# tweet or an iPhone tweet. We previously introduced the odds ratio, a summary
# statistic useful for quantifying these differences. For each device and a
# given word, let's call it y, we compute the odds or the ratio between the
# proportion of words that are y and not y and compute the ratio of those odds.
# Here we will have many proportions that are 0 so we use the 0.5 correction.
android_iphone_or <- tweet_words %>%
  count(word, source) %>%
  spread(source, n, fill = 0) %>%
  mutate(or = (Android + 0.5) / (sum(Android) - Android + 0.5) /
           ((iPhone + 0.5) / (sum(iPhone) - iPhone + 0.5)))
android_iphone_or %>% arrange(desc(or))
android_iphone_or %>% arrange(or)

# We already see somewhat of a pattern in the types of words that are being
# tweeted more in one device versus the other. However, we are not interested
# in specific words but rather in the tone. Vaziri's assertion is that the Android
# tweets are more hyperbolic. So how can we check this with data? Hyperbolic is a 
# hard sentiment to extract from words as it relies on interpreting phrases.
# However, words can be associated to more basic sentiments such as anger, fear,
# joy and surprise. In the next section we demonstrate basic sentiment analysis.

# SENTIMENT ANALYSIS

# In sentiment analysis we assign a word to one or more "sentiment". Although this
# approach will miss context dependent sentiments, such as sarcasm, when performed
# on large numbers of words, summaries can provide insights.

# The first step in sentiment analysis is to assign a sentiment to each word. The
# tidytext package includes several maps or lexicons in the object sentiments:
sentiments

# There are several lexicons in the tidytext package that give different sentiments.
# For example, the bing lexicon divides words into positive and negative. We can
# see this using the tidytext function get_sentiments():
get_sentiments("bing")

# The AFINN lexicon assigns a sore between -5 and 5, with -5 the most negative
# and 5 the most positive.
get_sentiments("afinn")

# The loughran and nrc lexicons provide several different sentiments:
get_sentiments("loughran") %>% count(sentiment)
get_sentiments("nrc") %>% count(sentiment)

# To start learning about how these lexicons were developed, read this help
# file: ?sentiments.

# For the analysis here we are interested in exploring the different sentiments
# of each tweet, so we will use the nrc lexicon:
nrc <- get_sentiments("nrc") %>%
  select(word, sentiment)

# We can combine the words and sentiments using inner_join(), which will only
# keep words associated with a sentiment. Here are 10 random words extracted
# from the tweets:
tweet_words %>% inner_join(nrc, by = "word") %>%
  select(source, word, sentiment) %>% sample_n(10)

# Now we are ready to perform a quantitative analysis comparing Android and iPhone
# by comparing the sentiments of tweets posted from each device. Here we could
# perform a tweet by tweet analysis, assigning a sentiment to each tweet. However,
# this is somewhat complex since each tweet will have several sentiments attached
# to it, one for each word appearing in the lexicon. For illustrative purposes, we
# will perform a much simpler analysis: we will count and compare the frequencies
# of each sentiment appears for each device.

sentiment_counts <- tweet_words %>%
  left_join(nrc, by = "word") %>%
  count(source, sentiment) %>%
  spread(source, n) %>%
  mutate(sentiment = replace_na(sentiment, replace = "none"))
sentiment_counts

# Because more words were used on the Android than on the phone:
tweet_words %>% group_by(source) %>% summarize(n = n())

# for each sentiment we can compute the odds of being in the device: proportion
# of words with sentiment versus proportion of words without and then compute
# the odds ratio comparing the two devices:
sentiment_counts %>%
  mutate(Android = Android / (sum(Android) - Android),
         iPhone = iPhone / (sum(iPhone) - iPhone),
         or = Android/iPhone) %>%
  arrange(desc(or))

# So we do see some difference and the order is interesting: the largest three
# sentiments are disgust, anger, and negative! But are they statistically
# significant? How does this compare if we are just assigning sentiments at
# random?

# To answer this question we can compute, for each sentiment, an odds ratio and
# confidence interval. We will add the two values we need to form a two-by-two
# table and the odds ratio:
library(broom)
log_or <- sentiment_counts %>%
  mutate(log_or = log( (Android / (sum(Android) - Android)) / (iPhone / (sum(iPhone) - iPhone))),
         se = sqrt(1/Android + 1/(sum(Android) - Android) + 1/iPhone + 1/(sum(iPhone) - iPhone)),
         conf.low = log_or - qnorm(0.975)*se,
         conf.high = log_or + qnorm(0.975)*se) %>%
  arrange(desc(log_or))

log_or

# A graphical visualization shows some sentiments that are clearly overrepresented:
log_or %>%
  mutate(sentiment = reorder(sentiment, log_or),) %>%
  ggplot(aes(x = sentiment, ymin = conf.low, ymax = conf.high)) +
  geom_errorbar() +
  geom_point(aes(sentiment, log_or)) +
  ylab("Log odds ratio for association between Android and sentiment") +
  coord_flip()

# We see that the disgust, anger, negative, sadness, and fear sentiments are
# associated with the Android in a way that is hard to explain by chance alone.
# Words not associated to a sentiment were strongly associated with iPhone source,
# which is in agreement with the original claim about hyperbolic tweets.

# If we are interested in exploring which specific words are driving these
# diferences, we can go back to our android_iphone_or object:
android_iphone_or %>% inner_join(nrc) %>%
  filter(sentiment == "disgust" & Android + iPhone > 10) %>%
  arrange(desc(or))

# We can make a graph:
android_iphone_or %>% inner_join(nrc, by = "word") %>%
  mutate(sentiment = factor(sentiment, levels = log_or$sentiment)) %>%
  mutate(log_or = log(or)) %>%
  filter(Android + iPhone > 10 & abs(log_or) > 1) %>%
  mutate(word = reorder(word, log_or)) %>%
  ggplot(aes(word, log_or, fill = log_or < 0)) +
  facet_wrap(~sentiment, scales = "free_x", nrow = 2) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# ASSESSMENT PART 1: DATES, TIMES, AND TEXT MINING

# This assessment reviews several concepts about dates, times, and text mining. In
# part 1 you will practice extracting and manipulating dates in real datasets. In
# part 2 you will walk through a sentiment analysis of a novel using steps covered
# in the previous section.

# Use the following libraries and options for coding questions:
library(dslabs)
library(lubridate)
options(digits = 3) # 3 significant digits

# Question 1

# Which of the following is the standard ISO 8601 format for dates?

# MM-DD-YYYY
# YYYY-MM-DD [X]
# YYYYMMDD
# YY-MM-DD

# Question 2

# Which of the following commands could convert this string into the correct
# date format?

dates <- c("09-01-02", "01-12-07", "02-03-04")

# 1
ymd(dates)

# 2
mdy(dates)

# 3
dmy(dates)

# 4 [X]
# It is impossible to know which format is correct without additional information.

# Question 3

# Load the brexit_polls data frame from dslabs:
data("brexit_polls")

# How many polls had a start date (startdate) in April (month number 4)?

brexit_polls %>%
  filter(month(startdate) == 4) %>% nrow() # 25

# Use the round_date() function on the enddate column with the argument
# unit = "week". How many polls ended the week of 2016-06-12?

brexit_polls %>%
  filter(round_date(enddate, unit = "week") == as.Date("2016-06-12")) %>% nrow() # 13

# Question 4

# Use the weekdays() function from lubridate to determine the weekday on which each
# poll ended (enddate).

# On which weekday did the greatest number of polls end?

brexit_polls %>%
  mutate(weekday = weekdays(enddate)) %>%
  group_by(weekday) %>% summarize(n = n()) %>% arrange(desc(n))

# Sunday, also:

table(weekdays(brexit_polls$enddate))

# Question 5

# Load the movielens data frame from dslabs
data(movielens)

# This data frame contains a set of about 100,000 movie reviews. The timestamp
# column contains the review date as the number of seconds since 1970-01-01 (epoch).

# Convert the timestamp column to dates using the lubridate as_datetime() function.

data <- movielens %>%
  mutate(date = as_datetime(timestamp))

# Which year had the most movie reviews? #2000

data %>%
  mutate(review_year = year(date)) %>%
  group_by(review_year) %>% summarize(n = n()) %>%
  arrange(desc(n))

# Which hour of the day had the most movie reviews? #20

data %>%
  mutate(review_hour = hour(date)) %>%
  group_by(review_hour) %>% summarize(n = n()) %>%
  arrange(desc(n))

# Also,

dates <- as_datetime(movielens$timestamp)
reviews_by_year <- table(year(dates)) # count reviews by year
names(which.max(reviews_by_year)) # name of year with most reviews

reviews_by_hour <- table(hour(dates)) # count reviews by hour
names(which.max(reviews_by_hour)) # name of hour with most reviews

# ASSESSMENT PART 2: DATES, TIMES, AND TEXT MINING

# In this part of the assessment, you will walk through a basic text mining and
# sentiment analysis task

# Project Gutenberg is a digital archive of public domain books. The R package
# gunterbergr facilitates the importation of these texts into R. We will combine
# this with the tidyverse and tidytext libraries to practice text mining.

# Use the libraries and options:
library(tidyverse)
library(gutenbergr)
library(tidytext)
options(digits = 3)

# You can see the books and documents available in gutenbergr like this:
gutenberg_metadata

# Question 6

# Use str_detect() to find the ID of the novel Pride and Prejudice.

# How many different ID numbers are returned? # 6

gutenberg_metadata %>%
  filter(str_detect(title, "Pride and Prejudice"))

# Question 7

# Notice that there are several versions of the book. The gutenberg_works()
# function filters this table to remove replicates and include only English
# language works. Use this function to find the ID for Pride and Prejudice.

# What is the corect ID number? # 1342

gutenberg_works(title == "Pride and Prejudice")$gutenberg_id

# Question 8

# Use the gutenberg_download() function to download the text for Pride and
# Prejudice. Use the tidytext package to create a tidy table with all the words
# in the text. Save this object as words.

# How many words are present in the book?

pride.prejudice <- gutenberg_download(1342,
                                      mirror = "http://mirrors.xmission.com/gutenberg/")
words <- pride.prejudice %>%
  unnest_tokens(word, text)
nrow(words) # 122342

# Question 9

# Remove stop words from the words object. Recall that stop words are defined in
# the stop_words data frame from the tidytext package.

# How many words remain?

words <- words %>%
  filter(!word %in% stop_words$word)
nrow(words) # 37448

# Also, words <- words %>% anti_join(stop_words)

# Question 10

# After removing stop words, detect and then filter out any token that contains
# a digit from words

# How many words remain?

words <- words %>% filter(!str_detect(word, "\\d+"))
nrow(words) # 37320

# Question 11

# Analyze the most frequent words in the novel after removing stop words and
# tokens with digits.

# How many words appear more than 100 times in the book?

words %>% count(word) %>%
  filter(n > 100) %>% nrow() # 24

# What is the most common word in the book?

words %>% count(word) %>%
  arrange(desc(n)) # elizabeth

words %>% count(word) %>%
  top_n(1, n) %>% pull(word)

# How many times does the most common word appear? # 597

words %>% count(word) %>%
  top_n(1, n) %>% pull(n)

# Question 12

# Define the afinn lexicon:
afinn <- get_sentiments("afinn")

# Use this afinn lexicon to assign sentiment values to words. Keep only
# words that are present in both words and the afinn lexicon. Save this data
# as afinn_sentiments.

afinn_sentiments <- inner_join(words, afinn, by = "word")

# How many elements of words have sentiments in the afinn lexicon?

afinn_sentiments %>% nrow() # 6065

# What proportion of words in afinn_sentiments have a positive value?

mean(afinn_sentiments$value > 0) # 0.563

# How many elements of afinn_sentiments have a value of 4?

sum(afinn_sentiments$value == 4) # 51

## COMPREHENSIVE ASSESSMENT: PUERTO RICO HURRICANE MORTALITY

# PROJECT INTRODUCTION

# On September 20, 2017, Hurricane Maria made landfall on Puerto Rico. It was
# the worst natural disaster on record in Puerto Rico and the deadliest Atlantic
# hurricane since 2004. However, Puerto Rico's official death statistics only
# tailed 64 deaths caused directly by the hurricane (due to structural collapse,
# debris, floods, and drownings), an undercount that slowed disaster recovery
# funding. The majority of the deaths resulted from infrastructure damage that
# made it difficult to access resources like clean food, water, power, healthcare
# and communications in the months after the disaster, and although these deaths
# were due to effects of the hurricane, they were not initially counted.

# In order to correct the misconception that few lives were lost in Hurricane
# Maria, statisticians analyzed how death rates in Puerto Rico changed after the
# hurricane and estimated the excess number of deaths likely caused by the storm.
# This analysis suggested that the actual number of deaths in Puerto Rico was
# 2,975 (95% CI: 2,658-3,290) over the 4 months following the hurricane, much
# higher than the original count.

# We will use your new data wrangling skills to extract actual daily mortality
# data from Puerto Rico and investigate whether the Hurricane Maria had an
# immediate effect on daily mortality compared to unnaffected days in September
# 2015-2017.

# You will need the following libraries and options to complete the assignment:
library(tidyverse)
library(pdftools)
options(digits = 3) # report 3 significant digits

# PUERTO RICO HURRICANE MORTALITY: PART 1

# Question 1

# In the extdata directory of the dslabs package, you will find a pdf file
# containing daily mortality data for Puerto Rico from Jan 1, 2015 to May 31,
# 2018. You can find the file like this:
fn <- system.file("extdata", "RD-Mortality-Report_2015-18-180531.pdf",
                  package = "dslabs")

# Find and open the file or open it directly from RStudio. On Windows you can type:
system("cmd.exe", input = paste("start", fn))

# Which of the following best describes this file?

# It is a table. Extracting the data will be easy.
# It is a report written in prose. Extracting the data will be impossible.
# It is a report combining graphs and tables. Extracting the data seems possible. [X]
# It shows graphs of the data. Extracting the data will be difficult.

# Question 2

# We are going to create a tidy dataset with each row representing an observation.
# The variables in this dataset will be year, month, day, and deaths.

# Use the pdftools package to read in fn using the pdf_text() function. Store the
# results in an object called txt.

txt <- pdf_text(fn)

# Describe what you see in txt.

# A table with the mortality data.

# A character string of length 12. Each entry represents the text in each page.
# The mortality data is in there somewhere. [X]

# A character string with one entry containing all the information in the file.

# An html document.

# Question 3

# Extract the ninth page of the pdf file from the object txt, then use the
# str_split() function from the stringr package so that you have each line in
# a different entry. The new line character is \n. Call this string vector x.
library(stringr)

x <- str_split(txt[9], "\\n")
head(x)

# Look at x. What best describes what you see?

# It is an empty string.
# I can see the figure shown in page 1.
# It is a tidy table.
# I can see the table! But there is a bunch of other stuff we need to get rid off. [X]

# What kind of object is x? # List

# How many entries does x have?

length(x[1]) # 1

# Question 4

# Define s to be the first entry of the x object.

# What kind of object is s?

s <- x[[1]]

# What kind of object is s? # Character vector
class(s)

# How many entries does s have? 
length(s) # 41

# Question 5

# When inspecting the string we obtained above, we see a common problem: white
# space before and after the other characters. Trimming is a common first step
# in string processing. These extra spaces will eventually make splitting the
# strings hard so we start by removing them.

# We learned about the command str_trim() that removes spaces at the start or end
# of the strings. Use this function to trim s and assign the result to s again.

s <- str_trim(s)

# After trimming, what single character is the last character of element 1 of s?

s[1] # s

# Question 6

# We want to extract the numbers from the strings stored in s. However, there are
# a lot of non-numeric characters that will get in the way. We want to remove these,
# but before doing this we want to preserve the string with the column header,
# which includes the month abbreviation.

# Use the str_which() function to find the row with the header. Save this result
# to header_index. Hint: find the first string that matches the pattern "2015"
# using the str_which() function.

header_index <- str_which(s, "2015")[1]

# What is the value of header_index?
header_index # 3

# Question 7

# We want to extract two objects from the header row: month will store the month
# and header will store the column names.

# Save the content of the header row into an object called header, then use
# str_split() to help define the two objects we need.

tmp <- str_split(s[header_index], "\\s+", simplify = TRUE)
month <- tmp[1]
header <- tmp[-1]

# What is the value of month? 
# Use header_index to extract the row. The separator here is one or more spaces.
# Also, consider using the simplify argument.

month # SEP

# What is the third value in header?

header[3] # 2017

# Question 8

# Notice that towards the end of the page defined by s you see a "Total" row
# followed by rows with other summary statistics. Create an object called
# tail_index with the index of the "Total" entry.

tail_index <- str_which(s, "Total")
tail_index # 36

# Question 9

# Because our PDF page includes graphs with numbers, some of our rows have just
# one number (from the y-axis of the plot). Use the str_count() function to
# create an object n with the count of numbers in each row.

# How many rows have a single number in them?
# You can write a regex for a number like this \\d+.

sum(str_count(s, "\\d+") == 1) # 2

# Question 10

# We are now ready to remove entries from rows that we know we don't need. The
# entry header_index and everything before it should be removed. Entries for
# which n is 1 should also be remove, and the entry tail_index and everything
# that comes after it should be removed as well.

s <- s[-(36:41)]
s <- s[-(1:3)]
remove <- str_count(s, "\\d+") == 1
s <- s[!remove]

# How many entries remain in s?
length(s) # 30

# Also,

out <- c(1:header_index, which(str_count(s, "\\d+") == 1), tail_index:length(s))
s <- s[-out]
length(s)

# Question 11

# Now we are ready to remove all text that is not a digit or space. Do this using
# regex and the str_remove_all() function.

# In regex, using the ^ inside the square brackets [] means not, like the ! means
# not in !=. To define the regex pattern to catch all non-numbers, you can type
# [^\\d]. But remember you walso want to keep spaces.

# Which of these commands produces the correct output?

#1
str_remove_all(s, "[^\\d]")

#2
str_remove_all(s, "[\\d\\s]")

#3
str_remove_all(s, "[^\\d\\s]") # [X]

#4
str_remove_all(s, "[\\d]")

s <- str_remove_all(s, "[^\\d\\s]")

# Question 12

# Use the str_split_fixed() function to convert s into a data matrix with just
# the day and death count data:

s <- str_split_fixed(s, "\\s+", n = 6)[,1:5]

# Now you are almost ready to finish. Add column names to the matrix: the first
# column should be day and the next columns should be header. Convert all values
# to numeric. Also, add a column with the month. Call the resulting object tab.

s <- matrix(as.numeric(s), ncol = ncol(s))
colnames(s) <- c("day", header)

# What was the mean number of deaths per day in September 2015?

colMeans(s) # 75.3

# What was the mean number of deaths per day in September 2016?

colMeans(s) # 78.9

# Hurricane Maria hit Puerto Rico on September 20, 2017. What was the mean number
# of deaths per day from September 1-19, 2017, before the Hurrican hit?

colMeans(s[1:19,]) # 83.7

# What was the mean number of deaths per day from September 20-30, 2017, after
# the hurricane hit?

mean(s[20:30, 4]) # 122

# This is how it is solved, with a data.frame:

tab <- s %>%
  as_data_frame() %>%
  setNames(c("day", header)) %>%
  mutate_all(as.numeric)

mean(tab$"2015") # 75.3
mean(tab$"2016") # 78.9
mean(tab$"2017"[1:19]) # 83.7
mean(tab$"2017"[20:30]) # 122.0

# Question 13

# Finish it up by changing tab to a tidy format, starting from this code outline:

# tab <- tab %>% #####(year, deaths, -day) %>%
#        mutate(deaths = as.numeric(deaths))
# tab

# What code fills the blank to generate a data frame with columns named "day",
# "year", and "deaths?

# separate
# unite
# gather [X]
# spread

tab <- tab %>% gather(year, deaths, -day) %>%
  mutate(deaths = as.numeric(deaths))

# Question 14

# Make a plot of deaths versus day with color to denote year. Exclude 2018
# since we have no data. Add a vertical line at day 20, the day that Hurricane
# Maria hit in 2017.

tab %>% filter(year < 2018) %>%
  ggplot(aes(day, deaths, color = year)) +
  geom_line() +
  geom_vline(xintercept = 20)

# Which of the following are TRUE?

# September 2015 and 2016 deaths by day are roughly equal to each other. [X]

# The day with the most deaths was the day of the hurricane: September 20, 2017

# After the hurricane in September 2017, there were over 100 deaths per day every
# day for the rest of the month. [X]

# No days before September 20, 2017 have over 100 deaths per day. [X]