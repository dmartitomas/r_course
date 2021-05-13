### DATA SCIENCE: PRODUCTIVITY TOOLS

# In this course, you will learn:

# How to leverage the many useful features provided by RStudio
# How to use Unix/Linux to manage your file system
# How to start a repository on GitHub
# How to perform version control with git

# INTRODUCTION TO PRODUCTIVITY TOOLS

# General Guiding Principles:

# Be systematic when organizing your filesystem
# Automize when possible
# Minimize the use of the mouse

# What we will lean:

# Unix shell
# Git and GitHub
# R markdown

## SECTION 1. INSTALLING SOFTWARE

# R: the programming language we use to analyze data
# RStudio: the integrated desktop environment we use to edit, organize, and test R scripts
# Git (and Git Bash for Windows): version control system

# KEEPING ORGANIZED WITH RSTUDIO PROJECTS

# RStudio provides a way to keep all the components of a data analysis project organized
# into one folder and to keep track of information about this project.

# To start a project, click on File > New Project > New repository > decide the location
# of files and give a name to the project, e.g., "my-first-project". This will then
# generate a Rproj file called my-first-project.Rproj in the folder associated with the
# project, from which you can double click to start where you last left off.

# The project name will appear in the upper left corner or the upper right corner, depending
# on your operating system. When you start an RStudio session with no project, it will
# display "Project (None)".

# 1.3. INTRODUCTION TO GIT AND GITHUB

# INSTALLING GIT INTRODUCTION

# Git is a version control system, tracking changes and coordinating the editing of code.

# GitHub is a hosting system for code, which can help with your career profile.

# Git is most effectively used with Unix, but it can also interface with RStudio.

# INSTALLING GIT

# Installing Git and Git Bash on Windows

# Download Git bash from https://git-scm.com

# When asked to choose the default editor for Git, we recommend choosing nano if you do
# not already know VIM.

# The "git and optional Unix tools from Windows" option will allow you to learn Unix
# from RStudio, however, it might interfere with the Windows command line.

# Finally, change the RStudio preference so that you are using Git bash as the terminal
# (only for Windows users).

# GITHUB

# Sign up for a GitHub account, with a name that is professional, short, and easy to
# remember.

# Connect to RStudio: global options > Git/SVM, enter the path to git executables.

# To avoid typing our GitHub password every time, we create a SSH/RSA key automatically
# through RStudio with the "create RSA key" button.

# GITHUB REPOSITORIES

# Step 1. Initialize a new repo on GitHub by clicking repository > new > choose a 
# descriptive name.

# Step 2. Connect to RStudio.

# RSTUDIO, GIT, AND GITHUB

# In terminal: configure git

# git config --global user.name "Your Name"
# git config -- global user.mail "your@email.com"

# In RStudio, create project > Version control > Git

# Git pane: status symbols and color

# Git actions:

# 1. pull: pull changes from remote repo (if you are in collaboration with others)
# 2. add: stage files for commit; click on stage box under git pane
# 3. commit: commit to the local repo; click on "commit" button under git pane; add a
# commit message
# 4. push: push to the remote repo on Github

# COMPREHENSION CHECK: INSTALLING SOFTWARE

# Question 1

# Which of the following statements about R and RStudio is true?

# R is a programming language, whereas RStudio is a desktop environment # Correct
# You can use RStudio without using R, but we recommend using R in this course
# When you download RStudio, it automatically downloads and installs R too
# You can only use R on Mac OS X and Linux. Windows users must use RStudio

# Question 2

# Which of the following is NOT true about installing packages? Select all that apply.

# To install a new package, the install.packages() function can be used
# To install a new package, the drop-down menu Tools > Install packages can be used
# Installed packages will remain installed even if you upgrade R # Incorrect
# Installing a package by building from GitHub will give you the exact same version
# as on CRAN # Incorrect

# Question 3

# Which of the following commands for editing scripts is not correct

# To save a script: Ctrl+S on Windows/Linux / Command+S on Mac
# To run an entire script: Ctrl+Shift+Enter on Windows/Linux / Command+Shift+Return
# on Mac, or click "Source" on the editor pane
# To open a new script: Ctrl+Shift+N on Windows/Linux / Command+Shift+N on Mac
# To run a single line of script: Ctrl+Shift / Command+Shift while cursos pointing to
# that line, or select the chunk and click "run" # Incorrect
# To comment on selected text: Ctrl+Shift+C or Command+Shift+C for Mac

# Question 4

# Which of the following statements about keeping organized with RStudio projects
# is not correct?

# To start a new project, click on File > New Project > New directory > New project >
# {choose a file directory and project name}
# You must always start a project in a new directory # Incorrect
# RStudio provides a way to keep all components of data analysis project organized into
# one folder and to keep track of information about this project
# Creating a new R project will produce an .Rproj file associated with the project.

# Question 5

# What can you change in the global options? Select all that apply.

# Set Git / GitHub configuration for each R project.
# Move the editor pane to the upper right. # Correct
# Change the editor theme to a dark background. # Correct
# Customize key binding. # Correct

# Question 6

# What does the term "pull" mean in the context of using Git in RStudio?

# Add local files to a remote GitHub repo.
# Download changes from the remote repo to your local repository. # Correct
# Configure the RStudio environment to automatically connect to GitHub.
# Save changes made in RStudio to the local repository on your computer.

# Question 7

# What does the term "push" mean in the context of using Git in RStudio?

# Upload changes made in your local repository to a remote repository. # Correct
# Download changes from the remote repo to the RStudio environment.
# Configure the RStudio environment to automatically connect to GitHub.
# Save changes made in RStudio to the local repository on your computer.

# Question 8

# What does the term "commit" mean in the context of using Git in RStudio?

# Add local files to a remote GitHub repo.
# Download changes from the remote repo to the RStudio environment.
# Configure the RStudio environment to automatically connect to GitHub.
# Save changes made in RStudio to the local repository on your computer. # Correct

# Question 9

# Did you create a GitHub account? Enter your GitHub username below.

# dmartitomas

## SECTION 2. BASIC UNIX

# The Unix section discusses the basics of managing your filesystem from the terminal with
# Unix commands such as mv and rm.

# Absolute path vs relative path

# A full path specifies the location of a file from the root directory. It is independent
# of your present directory, and must begin with either a "/" or a "~". In this example,
# the full path to our "project-1" file is: /home/projects/project-1

# A relative path is the path relative to your present working directory. If our present
# working directory is the "projects" folder, then the relative path to our "project-1"
# file is simply: project-1

# Path shortcuts

# One period "." is your current working directory

# Two periods ".." is the parent directory (up one from your present working directory)

# A tilde "~" is your home directory

# More path examples

# Your current working directory is ~/projects and you want to move to the figs directory
# in the poject-1 folder

# Solution 1: cd ~/projects/project-1/figs (absolute)
# Solution 2: cd project-1/figs (relative)

# Your current working directory is ~/projects and you want to move to the reports folder
# in the docs directory

# Solution 1: cd ~/docs/reports (absolute)
# Solution 2: cd ../docs/reports (relative)

# Your current working directory is ~/projects/project-1/figs and you want to move to the 
# project-2 folder in the projects directory.

# Solution 1: cd ~/projects/project-2 (absolute)
# Solution 2: cd ../../project-2 (relative)

# 2.1. INTRODUCTION TO UNIX

# ORGANIZING WITH UNIX

# THE TERMINAL

# The terminal helps to organize files in the system.
# On Mac, use utilities > terminal
# On Windows, use Git bash program.
# Use the keyboard to navigate the command line.

# THE FILESYSTEM

# We refer to all the files, folders, and programs (executables) on your computer as
# the `filesystem`

# Your filesystem is organized as a series of nested folders each containing files, folders,
# and executables. 

# In Unix, folders are referred to as directories and directories that are inside other
# directories are often referred to as subdirectories.

# The home directory is where all your stuff is kept. There is a hierarchical nature to the
# file system.

# Note for Windows Users: The typical R installation will make your Documents directory
# your home directory in R. This will likely be different from your home directory in
# Git Bash. Generally, when we discuss home directories, we refer to the Unix home
# directory which for Windows, in this book, is the Git Bash Unix directory.

# WORKING DIRECTORY

# The working directory is the current location

# Each terminal window has a working directory associated with it.

# The "pwd" command will display your working directory. The "/" symbol separates
# directories, while the first "/" at the beginning of the path stands for the root
# directory. When a path starts with "/", it is a "full path", which finds the current
# directory from the root directory. "Relative path" will be introduced soon.

# "~" means the home directory.

# 2.2. WORKING WITH UNIX

# UNIX COMMANDS

# Navigate the file system with commands introduced in this section.

# Auto-complete paths, commands and file names with the "Tab" key.

# Commands

# ls # list dir content
# mkdir folder_name # create directory called "folder_name"
# rmdir folder_name # remove an empty directory as long as it is empty
# rm -r folder_name # remove dir that is not empty, "r" stands for recursive
# cd # change directory
# cd ../ # two dots represents parent dir
# cd . # single dot represents current dir
# cd ~/projects # concatenate with forward slashes
# cd ../.. # change to two parent layer beyond
# cd - # whatever dir you were before
# cd # return to the home dir

# MV AND RM: MOVING AND REMOVING FILES

# The mv command moves files

# mv will not ask you to confirm the move, and it could potentially overwrite a file.

# the rm command removes files

# rm is permanent, which is different than throwing a folder in the trash.

# Commands

# mv path-to-file path-to-destination-directory
# rm filename-1 filename-2 filename-3

# LESS: LOOKING AT A FILE

# less allows you to quickly look at the content of a file

# Use q to exit the less page

# Use the arrows to navigate in the less page

# Commands

# less cv.tex

# PREPARING FOR A DATA SCIENCE PROJECT

# Ideally, files (code, data, output) should be structured and self-contained

# In a project, we prefer using relative paths (path relative to the default working
# directory) instead of the full path so that code can run smoothly on other individuals
# computers.

# It is good practice to write a README.txt file to introduce the file structure to 
# facilitate collaboration and for your future reference.

# Commands

# In terminal

# cd ~ # move to home directory
# mkdir projects # make a new directory called projects
# cd projects # move to ~/projects directory
# mkdir murders # make new directory called murders inside of projects
# cd murders # move to ~/projects/murders/
# mkdir data rda # make two new directories, one is data and the other is rda folder
# ls # to check if we indeed have one data folder and one rda folder
# pwd # check the current working directory
# mkdir figs # make a directory called figs to store figures

# In RStudio

# Pick existing directory as new project
getwd() # to confirm current working directory
save() # save into .rda file, .RData is also fine but less preferred
ggsave("figs/barplot.png") # save a plot generated by ggplot to a dir called "figs"

# COMPREHENSION CHECK PART 1: UNIX

# Question 1

# It is important to know which directory, or folder, you're in when you are working from
# the command line in Unix. Which line of code will tell you the current working directory?

# cd

# pwd # Correct

# rm

# echo

# Question 2

# You can't use your computer's mouse in a terminal. How can you see a line of code that you
# executed previously?

# Type pwd

# Type echo

# Use the up arrow # Correct

# Press the enter key

# Question 3

# Assume a student types pwd and gets the following output printed to the screen:
# /Users/student/Documents. Then, the student enters the following commands in sequence:

# mkdir projects
# cd projects

# What will be printed to the screen if the student types `pwd` after executing the two
# lines of code shown above?

# /Users/student/Documents

# /Users/student/Documents/projects # Correct

# /Users/student

# cd: projects: No such file or directory

# Question 4

# Which of the following statements does NOT correctly describe the utility of a command
# in Unix?

# The q key exits the viewer when you use less to view a file.

# The command ls lists files in the current directory.

# The command mkdir makes a new directory and moves into it. # False

# The mv command can move a file and change the name of a file.

# Question 5

# The following is the full path to a homework assignment file called "assignment.txt":
# /Users/student/Documents/projects/homeworks/assignment.txt.

# Which line of code will allow you to move the assignment.txt file from the "homeworks"
# directory into the parent directory "projects"?

# mv assignment.txt

# mv assignment.txt .

# mv assignment.txt .. # Correct

# mv assignment.txt /projects

# Question 6

# You want to move a file called assignment.txt into your projects directory. However, there
# is already a file called assignment.txt in the projects directory. 

# What happens when you execute the "move" (mv) command to move the file into the new
# directory?

# The moved "assignment.txt" file replaces the old "assignment.txt" file that was in the
# "projects" directory with no warning. # Correct

# An error message warns you that you are about to overwrite an existing file and asks
# if you want to proceed.

# An error message tells you that a file already exists with that name and asks you to
# rename the new file.

# The moved "assignment.txt" file is automatically renamed "assigment.txt (copy)" after
# it is moved into the "projects" directory.

# Question 7

# What does each of ~, ., .., / represent, respectively?

# Current directory, Home directory, Root directory, Parent directory
# Home directory, Current directory, Parent directory, Root directory # Correct
# Home directory, Hidden directory, Parent directory, Root directory
# Root directory, Current directory, Parent directory, Home directory
# Home directory, Parent directory, Home directory, Root directory

# Question 8

# Suppose you want to delete your project directory at ./myproject. The directory is
# not empty - there are still files inside of it.

# Which command should you use?

# rmdir myproject

# rmdir ./myproject

# rm - r myproject # Correct

# rm ./myproject

# Question 9

# The source() function reads a script from a url or file and evaluates it. Check ?source
# in the R console for more information.

# Suppose you have an R script at ~/myproject/R/plotfig.R and getwd() shows ~/myproject/result,
# and you are running your R script with source('~/myproject/R/plotfig.R').

# Which R function should you write in plotfig.R in order to correctly produce a plot in
# ~/myproject/result/fig/barplot.png?

# ggsave('fig/barplot.png'), because this is the relative path to the current working
# directory. # Correct

# ggsave('../result/fig/barplot.png'), because this is the relative path to the source
# file ("plotfig.R")

# ggsave('result/fig/barplot.png'), because this is the relative path to the project
# directory

# ggsave('barplot.png'), because this is the file name.

# Question 10

# Which of the following statements about the terminal are not correct?

# echo is similar to cat and can be used to print.

# The up arrow can be used to go back to a command you just typed

# You can click on the terminal to change the position of the cursos. # Incorrect

# For a long command that spans three lines, we can use the up-arrow to navigate the
# cursos to the first line. # Incorrect

# Question 11

# Which of the following statements about the filesystem is NOT correct?

# The home directory is where the system files that come with your computer exist. # Incorrect

# The name of the home directory is likely the same as the username on the system.

# File systems on Windows and Mac are different in some ways.

# Root directory is the directory that contains all directories.

# Question 12

# Which of the following meanings for options following less are not correct?

# -g: Highlights current match of any searched string

# -i: case-insensitive searches

# -s: automatically save the search object # Incorrect

# -X: leave file contents on screen when less exits

# Question 13

# Which of the following statements is incorrect about preparation for a data science
# project?

# Always use absolute paths when working on a data science projects # Incorrect

# Saving .RData every time you exit R will keep your collaborator informed of
# what you did # Incorrect

# Use ggsave to save generated files for use in a presentation or a report

# Saving your code in a Word file and inserting output images is a good idea for
# making a reproducible report. # Incorrect

## SECTION 3. REPRODUCIBLE REPORTS

# REPRODUCIBLE REPORTS WITH R MARKDOWN

# The final output is usually a report, textual descriptions and figures, and tables.

# The aim is to generate a reproducible report in R markdown and knitr.

# Features of Rmarkdown: code and text can be combined to the same document and figures and
# tables are automatically added to the file.

# R MARKDOWN

# You can learn more about R Markdown at markdowntutorial.com

# R markdown is a format for literate programming documents. Literate programming weaves
# instructions, documentation and detailed comments in between machine executable code,
# producing a document that describes the program that is best for human understanding.

# Start an R markdown document by clicking on File > New File > the R Markdown.

# The output could be HTML, PDF, or Microsoft Word, which can be changed in the header
# output, e.g., pdf_document / html_document/p>

# Code

# a sample code chunk
#```{r}
summary(pressure)
#```
# When echo=FALSE, code will be hidden in output file
#```{r echo=FALSE}
summary(pressure)
#```

# use a descriptive name for each chunk for debugging purposes
#```{r pressure-summary}
summary(pressure)
#```

# KNITR

# The knitr package is used to compile R markdown documents.

# The first time you click the "knit" button on the editor pane a pop-up window will
# prompt you to install packages, but after that is completed, the button will automatically
# knit your document.

# github_document gives a .md file, which will give you the best presentation on GitHub.

# output: html_document
# output: pdf_document
# output: word_document
# output: github_document

# COMPREHENSION CHECK: REPRODUCIBLE REPORTS

# Question 1

# Why might you want to create a report using R Markdown?

# R Markdown has better spell-checking tools than other word processors

# R Markdown allows you to automatically add figures to the final document # Correct

# R Markdown final reports have smaller file sizes than Word documents

# R Markdown documents look identical to the final report

# Question 2

# You have a vector of student heights called heights. You want to generate a histogram
# of these heights in a final report, but you don't want the code to show up in the final
# report. You want to name the R chunk "histogram" so that you can easily find the chunk
# later.

# Which of the following R chunks does everything you want to do?

# Correct
# ```{r histogram, echo=FALSE}
# hist(heights)
# ```

# Question 3

# Below is a section of R markdown code that generates a report.

# ---
# title: "Final Grade Distribution"
# output: pdf_document
# ---
# ```{r, echo=FALSE}
# load(file="my_data.Rmd")
# summary(grades)
# ```

# Select the statement that describes the file report generated by the R markdown code
# above

# A PDF document called "Final Grade Distribution" that prints a summary of the "grades"
# object. The code to load the file and produce the summary will not be included in the
# final report. # Correct

# A PDF document called "Final Grade Distribution" that prints a summary of the "grades"
# object. The code to load the file and produce the summary will be included in the final
# report.

# An HTML document called "Final Grade Distribution" that prints a summary of the "grades"
# object. The code to load the file and produce the summary will not be included in the
# final report.

# A PDF document called "Final Grade Distribution" that is empty because the argument
# echo=FALSE was used.

# Question 4

# The user specifies the output file format of the final report when using R Markdown.

# Which of the following file types is NOT an option for the final output?

# .rmd # Correct

# .pdf

# .doc

# .html

# Question 5

# ```{r, echo=F}
# n <- nrow(mtcars)
# ```

# Here `r n` cars are compared

# What will be the output for the above R markdown file when knit to HTML?

# The only output is the text: Here 32 cars are compared. # Correct

# Since we have echo=F, the code chunk is not evaluated, therefore we will have both
# the code and the text: Here `r n` cars are compared.

# The code will be displayed as well as Here 32 cars are compared.

# R cannot comprehend the value of n, we will get an error.

## SECTION 4. GIT AND GITHUB

# 4.1. GIT AND GITHUB

# Reasons to use Git and GitHub

# 1. Version control: Permits us to keep track of changes we made to code, to revert
# back to previous version of files, to test ideas using new branches and decide if
# we want to merge to the original.

# 2. Collaboration: On a centralized repo, multiple people may make changes to the
# code and keep versions synced. A pull request allows anyone to suggest changes to
# your code.

# 3. Sharing code

# To effectively permit version control and collaboration, files move across four
# different areas: Working Directory, Staging Area, Local Repository, and Upstream
# Repository.

# Start your Git journey with either coning and existing repo or intializing a new one.

# USING GIT AT THE COMMAND LINE

# Recap: there are four stages: working directory, staging area, local repository,
# and upstream repository.

# Clone an existing upstream repository (copy repo url from clone button, and type
# "git clone <url>"), and all three local stages are the same as upstream remote.

# The working directory is the same as the working directory in RStudio. When we
# edit files we only change the files in this place.

# git status: tells how the files in the working directory are related to the files
# in other stages.

# edits in the staging area are not tracked by the version control system by default -
# we add a file to the staging area by git add command.

# git commit: to commit files from the staging area to local repository, we need to add
# a message stating what we are doing by git commit -m "something"

# git log: keeps track of all the changes we have made to the local repository

# git push: allows moving from the local repository to upstream repository, only if you
# have the permission (e.g., if it is yours)

# git fetch: update local repository to be like the upstream repository, from upstream
# to local

# git merge: make the updated local sync with the working directory and staging area

# To change everything in one shot (from upstream to working dir), use git pull (equivalent
# to combining git fetch + git merge)

# Code

# pwd
# mkdir git-example
# cd git-example
# git clone https://github.com/rairizarry/murders.git
# cd murders
# ls
# git status
# echo "test" >> new-file.txt
# echo "temporary" >> tmp.txt
# git add new-file.txt
# git status
# git commit -m "adding a new file"
# git status
# echo "adding a second line" >> new-file.txt
# git add new-file.txt
# git commit -m "minor change to new-file" new-file.txt
# git status
# git add
# git log new-file.txt
# git push
# git fetch
# git merge

# CREATING A GITHUB REPOSITORY

# Recap: two ways to get started, one is cloning an existing repository, the other is
# initializing your own

# Create our own project on our computer (independent of Git) on our own machine

# Create an upstream repo on Github, copy repo's url

# Make a local git repository: On the local machine, in the project directory, use git
# init. Now git starts tracking everything in the local repo

# Now we need to start moving files into our local repo and connect local repo to the
# upstream remote by git remote add origing <url>

# Note: The first time you push to a new repository, you may also need to use these
# git push options: git push --set-upstream origin master. If you need to run these
# arguments but forget to do so, you will get an error with a reminder.

# Code

# cd ~/projects/murders
# git init
# git add README.txt
# git commit -m "First commit. Adding README.txt file just to get started"
# git remote add origin "https://github.com/rairizarry/murders.git"
# git push # you may need to add these arguments the first time: --set-upstream origin
# master

# COMPREHENSION CHECK: GIT AND GITHUB

# Question 1

# Which statement describes reasons why we recommend using git and GitHub when working
# on data analysis projects?

# Git and GitHub facilitate fast, high-throughput analysis of large data sets.

# Git and GitHub allow easy version control, collaboration, and resource sharing. # Correct

# Git and GitHub have graphical interfaces that make it easy to learn to code in R.

# Git and GitHub is good for long-term storage of private data.

# Question 2

# Select the steps necessary to:

# 1. Create a directory called "project-clone",
# 2. Clone the contents of a git repo at the following URL into that directory
# (https://github.com/user123/repo123.git), and
# 3. List the contents of the cloned repo.

# Correct

# mkdir project-clone
# cd project-clone
# git clone https://github.com/user123/repo123.git
# ls

# Question 3

# You have successfully cloned a GitHub repository onto your local system. The cloned
# repository contains a file called "heights.txt" that lists the heights of students
# in a class. One student was missing from the dataset, so you add that student's height
# using the following command: echo "165" >> heights.txt

# Next you enter the command git status to check the status of the GitHub repository.

# What message is returned and what does it mean?

# Correct

# modified: heights.txt, no changes added to commit

# This message means that the heights.txt file was modified, but the changes have not been
# staged or committed to the local repository.

# Question 4

# You cloned your own repository and modified a file within it on your local system. Next, 
# you executed the following series of commands to include the modified file in the upstream
# directory, but it didn't work. Here is the code you typed.

# git add modified_file.txt
# git commit -m "minor changes to file" modified_file.txt
# git pull

# What is preventing the modified file from being added to the upstream repository?

# The wrong option is being used to add a descriptive message to the commit.
# git push should be used instead of git pull # Correct
# git commit should come before git add
# The git pull command line needs to include the file name

# Question 5

# You have a directory of scripts and data files on your computer that you want to share
# with collaborators using GitHub. You create a new repository on your GitHub account
# called "repo123" that has the following URL: https://github.com/user123/repo123.git

# Which of the following sequences of commands will convert the directory on your 
# computer to a GitHub directory and create and add a descriptive "read me" file
# to the new repository?

# Correct

# echo "A new repository with my scripts and data" > README.txt
# git init
# git add README.txt
# git commit -m "First commit. Adding README file."
# git remote add origin `https://github.com/user123/repo123.git`
# git push

# Question 6

# You have made a local change to a file in your R project, which is associated with a
# GitHub repository. You add your changes and push, but you receive a message:

# Everything up-to-date

# Which of the following commands did you forget to do?

# Correct: git commit

# Question 7

# Suppose you previously cloned a repository with git clone. Running git status shows:

# On branch master
# Your branch is up to date with 'origin/master'.
# nothing to commit, working tree clean

# However, you know that there are some changes in the upstream repository.

# How will you sync these changes with one command?

# git fetch
# git pull # Correct
# git merge origin/master
# git merge upstream/master
# git push

## SECTION 5. ADVANCED UNIX

# ADVANCED UNIX. PART 1

