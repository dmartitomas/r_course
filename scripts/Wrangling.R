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

