### DATA SCIENCE: PROBABILITY

## SECTION 1. DISCRETE PROBABILITY

# 1.1. INTRODUCTION TO DISCRETE PROBABILITY

# DISCRETE PROBABILITY

# The probability of an event is the proportion of times the event occurs when we repeat the experiment
# independently under the same conditions.

# Pr(A) = probability of event A

# An `event` is defined as an outcome that can occur when something happens by chance.

# We can determine probabilities related to discrete variables (picking a bead, choosing 48 Dems and 52
# Republicans from 100 likely voters) and continuous variables (height over 6 feet).

# MONTE CARLO SIMULATIONS

# Monte Carlo simulations model the probability of different outcomes by repeating a random process
# a large enough number of times that the results are similar to what would be observed if the
# process were repeated forever.

# The sample() function repeats lines of code a set number of times. It is used with sample() and
# similar functions to run Monte Carlo simulations.

beads <- rep(c("red", "blue"), times = c(2, 3)) # create an urn with 2 red, 3 blue
beads # view beads object
sample(beads, 1) # sample 1 bead at random

B <- 10000 # number of times to draw 1 bead
events <- replicate(B, sample(beads, 1)) # draw 1 bead, B times
tab <- table(events) # make a table of outcome counts
tab # view count table
prop.table(tab) # view table of outcome proportions

# SETTING THE RANDOM SEED

# The set.seed() function

# Before we continue, we will briefly explain the following important line of code:

set.seed(1986)

# Throughout this book we use random number generators. This implies that many of the results
# presented can actually change by chance, which then suggests that a frozen version of the book
# may show a different result than what you obtain when you try to code as shown in the book.
# This actually is fine since the results are random and change from time to time. However, if
# you want to ensure that results are exactly the same every time you run them, you can set R's
# random number generation seed to a specific number. Above we set it to 1986. We want to avoid
# using the same seed every time. A popular way to pick the seed is the year - month - day. For
# example, we picked 1986 on December 20, 2018: 2018 - 12 - 20 = 1986.

# You can learn more about setting the seed by looking at the documentation:

?set.seed

# In the exercises, we may ask you to set the seed to assure that the results you obtain are
# exactly what we expect them to be.

# Important note on seeds in R 3.5 and R 3.6

# R was recently updated to version 3.6 in early 2019. In this update, the default method for
# setting the seed changed. This means that exercises, videos, textbook excerpts and other
# code you encounter online may yield a different result based on your version of R.

# If you are running R 3.6, you can revert to the original seed setting behavior by adding
# the argument sample.kind = "Rounding". For example:

set.seed(1)
set.seed(1, sample.kind = "Rounding") # will make R 3.6 (4.0?) generate a seed as in 3.5

# Using the sample.kind = "Rounding" argument will generate a message:

# non-uniform 'Rounding' sampler used

# This is not a warning or a cause for alarm - it is a confirmation that R is using the
# alternate seed generation method, and you should expect to receive this message in your
# console.

# If you use R 3.6, you should always use the second form of set.seed() in this course
# series (outside of DataCamp assignments). Failure to do do may result in an otherwise
# correct answer being rejected by the grader. In most cases where a seed is required, you
# will be reminded of this fact.

# USING THE MEAN FUNCTION FOR PROBABILITY

# An important application of the mean() function

# In R, applying the mean() function to a logical vector returns the proportion of elements
# that are TRUE. It is very common to use the mean() function in this way to calculate
# probabilities and we will do so throughout the course.

# Suppose you have the vector beads from a previous video:

beads <- rep(c("red", "blue"), times = c(2, 3))
beads

# To find the probability of drawing a blue bead at random, you can run:

mean(beads == "blue") # 0.6

# This code is broken down into steps inside R. First, R evaluates the logical statement
# beads == "blue", which generates the vector:

# FALSE FALSE TRUE TRUE TRUE

# When the mean function is applied, R coerces the logical values to numeric values,
# changing TRUE to 1 and FALSE to 0.

# 0 0 1 1 1

# The mean of the zeros and ones thus gives the proportion of TRUE values. As we have
# learned and will continue to see, probabilities are directly related to the proportion
# of events that satisfy a requirement.

# PROBABILITY DISTRIBUTIONS

# The probability distribution for a variable describes the probability of observing each
# possible outcome.

# For discrete categorical variables, the probability distribution is defined by the
# proportions for each group.

# INDEPENDENCE

# Conditional probabilities compute the probability that an event occurs given information
# about dependent events. For example, the probability of drawing a second kind given that
# the first draw is a king is:

# Pr(Card 2 is a king | Card 1 is a king) = 3 / 51

# If two events A and B are independent, Pr(A|B) = Pr(A)

# To determine the probability of multiple events occurring, we use the multiplication rule.

# Equations

# The multiplication rule for independent events is:

# Pr(A and B and C) = Pr(A) x Pr(B) x Pr(C)

# The multiplication rule for dependent events considers the conditional probability of
# both events occurring:

# Pr(A and B) = Pr(A) x Pr(B|A)

# We can expand the multiplication rule for dependent events to more than 2 events:

# Pr(A and B and C) = Pr(A) x Pr(B|A) x Pr(C|A and B)

# ASSESSMENT: INTRODUCTION TO DISCRETE PROBABILITY

# Probability of cyan

# One ball will be drawn at random from a box containing 3 cyan balls, 5 magenta balls, and
# 7 yellow balls. 

# What is the probability that the ball will be cyan?

balls <- rep(c("cyan", "magenta", "yellow"), times = c(3, 5, 7))
balls

mean(balls == "cyan") # 0.2

# Probability of not cyan

# One ball will be drawn at random from a box containing 3 cyan balls, 5 magenta balls, and
# 7 yellow balls.

# What is the probability that the ball will not be cyan?

mean(balls != "cyan") # 0.8

# Sampling without replacement

# Instead of taking just one draw, consider taking two draws. You take the second draw
# without returning the first draw to one box. We call this sampling without replacement.

# What is the probability that the first draw is cyan and that the second draw is not
# cyan?

mean(balls == "cyan") * (12/14) # 3/15 * 12/14 ~= .0171

# Sampling with replacement

# Now repeat the experiment, but this time, after taking the first draw and recording the
# color, return it back to the box and shake the box. We call this sampling with
# replacement.

# What is the probability that the first draw is cyan and that the second draw is not
# cyan?

mean(balls == "cyan") * mean(balls != "cyan")

# DataCamp Assessment: Introduction to discrete probability

# Exercise 1. Probability of cyan - generalized

# In the edX exercises for this section, we calculated some probabilities by hand. Now
# we'll calculate those probabilities using R.

# One ball will be drawn at random from a box containing: 3 cyan balls, 5 magenta balls,
# and 7 yellow balls.

# Define a variable p as the probability of choosing a cyan ball from the box.

# Print the value of p.

cyan <- 3
magenta <- 5
yellow <- 7

# Assign a variable p as the probability of choosing a cyan ball from the box.

p <- cyan / (cyan + magenta + yellow)

# Print the variable `p` to the console

p

# Exercise 2. Probability of not cyan - generalized

# We defined the variable p as the probability of choosing a cyan ball from a box
# containing: 3 cyan balls, 5 magenta balls, and 7 yellow balls.

# What is the probability that the ball you draw from the box will NOT be cyan?

# Using the probability of choosing a cyan ball p, calculate the probability of choosing
# any other ball.

1 - p

# Exercise 3. Sampling without replacement - generalized

# Instead of taking just one draw, consider taking two draws. You take the second draw
# without returning the first draw to the box. We call this sampling without replacement.

# What is the probability that the first draw is cyan and that the second draw is not
# cyan?

# Calculate the conditional probability p_2 of choosing a ball that is not cyan after
# one cyan ball has been removed from the box.

# Calculate the joint probability of both choosing a cyan ball on the first draw and a 
# ball that is not cyan on the second draw using p_1 and p_2.

# The variable `p_1` is the probability of choosing a cyan ball from the box on the
# first draw.

p_1 <- cyan / (cyan + magenta + yellow)

# Assign a variable `p_2` as the probability of not choosing a cyan ball on the second
# draw without replacement.

p_2 <- (magenta + yellow) / (cyan - 1 + magenta + yellow)

# Calculate the probability that the first draw is cyan and the second draw is not cyan
# using `p_1` and `p_2`.

p_1 * p_2

# Exercise 4. Samplling with replacement - generalized

# Now repeat the experiment but this time, after taking the first draw and recording the
# color, return it back to the box and shake the box. We call this sampling with replacement.

# What is the probability that the first draw is cyan and that the second draw is not cyan?

# Calculate the probability p_2 of choosing a ball that is not cyan on the second draw, with
# replacement.

# Next, use p_1 and p_2 to calculate the probability of choosing a cyan ball on the first
# draw and a ball that is not cyan on the second draw (after replacing the first ball).

p_1 <- cyan / (cyan + magenta + yellow)

# Assign a variable p_2 as the probability of not choosing a cyan ball on the second draw
# with replacement.

p_2 <- (magenta + yellow) / (cyan + magenta + yellow)

# Calculate the probability that the first draw is cyan and the second draw is not cyan
# using p_1 and p_2.

p_1 * p_2

# 1.2. COMBINATIONS AND PERMUTATIONS

# paste() joins two strings and inserts a space in between.

# expanse.grid() gives the combinations of 2 vectors or lists.

# permutations(n, r) from the gtools package lists the different ways that r items can be
# selected from a set of n options when order matters.

# combinations(n, r) from the gtools package lists the different ways that r items can be
# selected from a set of n options when order does not matter.

# Introducing paste() and expand.grid()

# Joining strings with paste

number <- "Three"
suit <- "Hearts"

paste(number, suit)

# Joining vectors element-wise with paste

paste(letters[1:5], as.character(1:5))

# Generating combinations of 2 vectors with expand.grid

expand.grid(pants = c("blue", "black"), shirt = c("white", "grey", "plaid"))

# Generating a deck of cards

suits <- c("Diamonds", "Clubs", "Hearts", "Spades")
numbers <- c("Ace","Deuce", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
             "Ten", "Jack", "Queen", "King")
deck <- expand.grid(number = numbers, suit = suits)
deck <- paste(deck$number, deck$suit)

# Probability of drawing a king

kings <- paste("King", suits)
mean(deck %in% kings)

# Permutations and Combinations

# installed the gtools package

library(gtools)
permutations(5,2) # ways to choose 2 numbers in order from 1:5
all_phone_numbers <- permutations(10, 7, v = 0:9)
n <- nrow(all_phone_numbers)
index <- sample(n, 5)
all_phone_numbers[index,]

permutations(3,2) # order matters
combinations(3,2) # order does not matter

# Probability of drawing a second king given that one king is drawn

hands <- permutations(52, 2, v = deck)
first_card <- hands[,1]
second_card <- hands[,2]
sum(first_card %in% kings)

sum(first_card %in% kings & second_card %in% kings) / sum(first_card %in% kings)

# Probability of a natural 21 blackjack

aces <- paste("Ace", suits)
facecard <- c("King", "Queen", "Jack", "Ten")
facecard <- expand.grid(number = facecard, suit = suits)
facecard <- paste(facecard$number, facecard$suit)

hands <- combinations(52, 2, v = deck) # all possible hands

# Probability of a natural 21 given that the ace is listed first in `combinations`

mean(hands[,1] %in% aces & hands[,2] %in% facecard)

# Probability of a natural 21 checking for both ace first and ace second

mean((hands[,1] %in% aces & hands[,2] %in% facecard) | (hands[,2] %in% aces &
                                                          hands[,1] %in% facecard))

# Monte Carlo simulation of natural 21 in blackjack

# Code for one hand of blackjack

hand <- sample(deck, 2)
hand

# Code for B = 10,000 hands of blackjack

B <- 10000
results <- replicate(B, {
  hand <- sample(deck, 2)
  (hand[1] %in% aces & hand[2] %in% facecard | hand[2] %in% aces & hand[1] %in% facecard)
})
mean(results)

# THE BIRTHDAY PROBLEM

# duplicated() takes a vector and returns a vector of the same length with TRUE for any
# elements that have appeared previously in that vector.

# We can compute the probability of shared birthdays in a group of people by modeling
# birthdays as random draws from the numbers 1 through 365. We can then use this sampling
# model of birthdays to run a Monte Carlo simulation to estimate the probability of shared
# birthdays.

# The birthday problem

# Checking for duplicated birthdays in one 50 person group

n <- 50
bdays <- sample(1:365, n, replace = TRUE) # generate n random birthdays
any(duplicated(bdays)) # check if any birthdays are duplicated

# Monte Carlo simulation with B = 10,000 replicates

B <- 10000
results <- replicate(B, { # returns vector of B logical values
  bdays <- sample(1:365, n, replace = TRUE)
  any(duplicated(bdays))
})
mean(results) # calculates proportion of groups with duplicated bdays

# sapply

# Some functions automatically apply element-wise to vectors, such as sqrt() and *.

# However, other functions do not operate element-wise by default. This includes
# functions we define ourselves.

# The function sapply(x, f) allows any other function f to be applied element-wise
# to the vector x.

# The probability of an event happening is 1 minus the probability of that event
# not happening:

# Pr(event) = 1 - Pr(no event)

# We can compute the probability of shared birthdays mathematically:

# Pr(shared bdays) = 1 - Pr(no shared bdays) = 1 - (1 x 364/365 x 3636/365 x ... x 365 - n + 1 / 365)

# Function for birthday problem using Monte Carlo simulations

# Note that the function body of compute_prob() is the code that we wrote in the
# previous video. If we write this code as a function, we can use sapply() to apply
# this function to several values of n.

# Function to calculate probability of shared birthdays across n people

compute_prob <- function(n, B = 10000){
  same_day <- replicate(B, {
    bdays <- sample(1:365, n, replace = TRUE)
    any(duplicated(bdays))
  })
  mean(same_day)
}

n <- seq(1, 60)

# Element-wise operation over vectors and sapply

x <- 1:10
sqrt(x) # sqrt operates on each element of the vector

y <- 1:10
x * y # * operates element-wise on both vectors

compute_prob(n) # does not iterate over the vector n without sapply

sapply(x, sqrt) # this is equivalent to sqrt(x)

prob <- sapply(n, compute_prob) # element-wise application of compute_prob to n
plot(n, prob)

# Computing birthday problem probabilities with sapply

# Function for computing exact probability of shared birthdays for any n

exact_prob <- function(n){
  prob_unique <- seq(365, 365-n+1) / 365 # vector of fractions for multiplication rule
  1 - prod(prob_unique) # calculate probability of no shared bdays and substract from 1
}

# Applying function element-wise to vector of n values

eprob <- sapply(n, exact_prob)


# Plotting Monte Carlo results and exact probabilities on same graph

plot(n, prob) # plot Monte Carlo results
lines(n, eprob, col = "red") # add line for exact prob

# HOW MANY MONTE CARLO EXPERIMENTS ARE ENOUGH?

# The larger the number of Monte Carlo replicates B, the more accurate the estimate.

# Determining the appropriate size for B can require advanced statistics.

# One practical approach is to try many sizes for B and look for sizes that provide
# stable estimates.

# Estimating a practical value of B

# This code runs Monte Carlo simulations to estimate the probability of shared
# birthdays using several B values and plot the results. When B is large enough
# that the estimated probability stays stable, then we have selected a useful
# value of B.

B <- 10^seq(1, 5, len = 100) # defines vector of many B values
compute_prob <- function(B, n = 22){ # function to run Monte Carlo simulation with each B
  same_day <- replicate(B, {
    bdays <- sample(1:365, n, replace = TRUE)
    any(duplicated(bdays))
  })
  mean(same_day)
}

prob <- sapply(B, compute_prob) # apply compute_prob to many values of B
plot(log10(B), prob, type = "l") # plot a line graph of estimates

# DataCamp Assessment: Combinations and Permutations

# Exercise 1. Independence

# Imagine you draw two balls from a box containing colored balls. You either
# replace the first ball before you draw the seond or you leave the first ball
# out of the box when you draw the second ball.

# Under which situation are there two draws independent of one another?

# Remember than two events A and B are independent if Pr(A and B) = Pr(A) * Pr(B)

# 1. You don't replace the first ball before drawing the next.

# 2. You do replace the first ball before drawing the next. # This is the correct answer

# 3. Neither situation describes independent events.

# 4. Both situations describe independence events

# Exercise 2. Sampling with replacement

# Say you've drawn 5 balls from the a box that has 3 cyan balls, 5 magenta balls, and
# 7 yellow balls, with replacement, and all have been yellow balls.

# What is the probability that the next one is yellow?

# Assign the variable p_yellow as the probability of choosing a yellow ball on the
# first draw.

# Using the ariable p_yellow, calculate the probability of choosing a yellow ball on the
# sixth draw.

cyan <- 3
magenta <- 5
yellow <- 7

# Assign the variable p_yellow as the probability that a yellow ball is drawn from the box.

p_yellow <- yellow / (cyan + magenta + yellow)

# Using the variable p_yellow, calculate the probability of drawing a yellow ball on the sixth
# draw. Print this value to the console.

p_yellow

# Exercise 3. Rolling a die

# If you roll a 6-sided die once, what is the probability of not seeing a 6? If you roll a 6-
# sided die six times, what is the probability of not seeing a 6 on any of those rolls?

# Assign the variable p_no6 as the probability of not seeing a 6 on a single roll.

# Then, calculate the probability of not seeing a 6 on six rolls using p_no6.

# Assign the variable p_no6 as the probability of not seeing a 6 on a single roll.

p_no6 <- 5 / 6

# Calculate the probability of not seeing a 6 on six rolls using p_no6. Print your results
# on the console: do not assign it to a variable

p_no6^6

# Exercise 4. Probability the Celtics win a game

# Two teams, say the Celtics and the Cavs, are playing a seven game series. The Cavs are a
# better team and have a 60% chance of winning a game.

# What is the probability that the Celtics win at least one game? Remember that the Celtics
# must win one of the fist four games, or the series will be over!

# Calculate the probability that the Cavs will win the first four games of the series.

# Calculate the probability that the Celtics win at least one game in the first four games
# of the series.

# Assign the variable p_cavs_win4 as the probability that the Cavs will win the first four
# games of the series.

p_cavs_win4 <- 0.6^4

# Using the variable p_cavs_win4, calculate the probability that the Celtics win at least
# one game in the first four games of the series.

1 - p_cavs_win4

# Exercise 5. Monte Carlo simulation for Celtics winning a game

# Create a Monte Carlo simulation to confirm your answer to the previous problem by
# estimating how frequently the Celtics win at least 1 of 4 games. Use B <- 10000
# simulations.

# The provided sample code simulates a single series of four random games, simulated_games.

# Use the replicate function for B <- 10000 simulations of a four game series. The results
# of replicate should be stored to a variable named celtic_wins.

# Within each simulation, replicate the sample code to simulate a four-game series named
# simulated_games. Then use the any function to indicate whether the four-game series contains
# at least one win for the Celtics. Perform these operations in two separate steps.

# Use the mean function on celticc_wins to find the proportion of simulations that contain
# at least one win for the Celtics out of four games.

# This line of example code simulates four independent random games where the Celtics either
# lose or win. Copy this example code to use within the replicate function.

simulated_games <- sample(c("lose","win"), 4, replace = TRUE, prob = c(0.6,0.4))

# The variable B specifies the number of times we want the simulation to run. Let's run the
# Monte Carlo simulation 10,000 times.

B <- 10000

# Use the set.seed function to make sure your answer matches the expected result after
# random sampling.

set.seed(1, sample.kind = "Rounding")

# Create an object called celtic_wins that replicates two steps for B iterations: (1)
# generating a random four-game series simulated_games using the example code, then
# (2) determining whether the simulated series contains at least one win for the
# Celtics

celtic_wins <- replicate(B, {
  simulated_games <- sample(c("lose","win"), 4, replace = TRUE, prob = c(0.6,0.4))
  any(simulated_games == "win")
})

# Calculate the frequency ouf of B iterations that the Celtics won at least one game.
# Print your answer to the console

mean(celtic_wins)

# 1.3. ADDITION RULE AND MONTY HALL

# THE ADDITION RULE

# The addition rule states that the probability of event A or event B is the probability
# of event A plus the probability of event B minuts the probability of both events A and
# B happeing together.

# Pr(A or B) = Pr(A) + Pr(B) - Pr(A and B)

# Example: the addition rule for a natural 21 in blackjack

# We apply the addition rule where A = drawing an ace then a facecard and B = drawing a
# facecard then an ace. Note that in this case, both events A and B cannot happen at the
# same time, so Pr(A and B) = 0.

# Pr(ace then facecard) = 4/52 * 16/51
# Pr(facecard then ace) = 16/52 * 4/51
# Pr(ace then facecard | facecard then ace) = 4/52 * 16/51 + 16/52 * 4/51 = 0.0483

# THE MONTY HALL PROBLEM

# Monte Carlo simulations can be used to simulate random outcomes, which makes them
# useful when exploring ambiguous or less intuitive problems like the Monty Hall problem.

# In the Monty Hall problem, contestants choose one of three doors that may contain a 
# prize. Then, one of the doors that was not chosen by the contestant and does not contain
# a prize is revealed. The contestant can then choose whether to stick with the original
# choice or switch to the remaining unopened door.

# Although it may seem intuitively like the contestant has a 1 in 2 chance of winning
# regardless of whether they stick or switch, Monte Carlos simulations demonstrate that
# the actual probability of winning is 1 in the 3 with the stick strategy and 2 in 3
# with the switch strategy.

# Monte Carlo simulation of stick strategy

B <- 1000
stick <- replicate(B, {
  doors <- as.character(1:3)
  prize <- sample(c("car", "goat", "goat")) # puts prizes in random order
  prize_door <- doors[prize == "car"] # notes which door has the prize
  my_pick <- sample(doors, 1) # notes which door is chosen
  show <- sample(doors[!doors %in% c(my_pick, prize_door)], 1) # opens door with no prize that isn't chosen
  stick <- my_pick # stick with original door
  stick == prize_door # test whether the original door has the prize
})

mean(stick) # probability of choosing prize door when sticking ~ 0.358

# Monte Caro simulation of switch strategy

switch <- replicate(B, {
  doors <- as.character(1:3)
  prize <- sample(c("car", "goat", "goat")) # puts prizes in random order
  prize_door <- doors[prize == "car"] # notes which door has the prize
  my_pick <- sample(doors, 1) # notes which door is chosen first
  show <- sample(doors[!doors %in% c(my_pick, prize_door)], 1) # opens door with no prize that isn't chosen
  switch <- doors[!doors %in% c(my_pick, show)] # switch to the door that wasn't chosen first or opened
  switch == prize_door # test whether the switched door has the prize
})

mean(switch) # probability of choosing prize door when switching ~ 0.685

# DataCamp Assessment: The Addition Rule and Monty Hall

# Exercise 1. The Cavs and the Warriors

# Two teams, say the Cavs and the Warriors, are playing a seven game championship series.
# The first to win four games wins the series. The teams are equally good, so they each
# have a 50-50 chance of winning each game.

# If the Cavs lose the first game, what is the probability that they win the series?

# Assign the number of remaining games to the variable n.

# Assign a variable outcomes as a vector of possible outcomes in a single game, where 0
# indicates a loss and 1 indicates a win for the Cavs.

# Assign a variable `l` to a list of possible outcomes in all remaining games. Use the
# rep function to create a list of n games, where each game consists of list(outcomes).

# Use the expand.grid function to identify which combinations of game outcomes result
# in the Cavs winning the number of games necessary to win the series.

# Use the rowSums function to identify which combinations of game outcomes result in
# the Cavs winning the number of games necessary to win the series.

# Use the mean function to calculate the proportion of outcomes that result in the Cavs
# winning the series and print your answer to the console.

# Assign a variable `n` as the number of remaining games.

n <- 6

# Assign a variable `outcomes` as a vector of possible game outcomes, where 0 indicates
# a loss and 1 indicates a win for the Cavs.

outcomes <- 0:1

# Assign a variable `l` to a list of all possible outcomes in all remaining games. Use
# the rep function on `list(outcomes)` to create a list of length `n`

l <- rep(list(outcomes), times = n)

# Create a data frame named possibilites that contains all combinations of possible
# outcomes for the remaining games.

possibilities <- expand.grid(l)

# Create a vector named `results` that indicates whether each row in the data frame
# possibilities contains enough wins for the Cavs to win the series.

results <- rowSums(possibilities)

# Calculate the proportion of `results` in which the Cavs win the series. Print the
# outcome to the console.

mean(results >= 4)

# Exercise 2. Confirm the results of the previous question with a Monte Carlo
# simulation to estimate the probability of the Cavs winning the series after losing
# the first game.

# Use the replicate function to replicate the sample code for B <- 10000 simulations.

# Use the sample function to simulate a series of 6 games with random, independent
# outcomes of either a loss for the Cavs (0) or a win for the Cavs (1) in that
# order. Use the default probabilities to sample.

# Use the sum function to determine whether a simulated series contained at least
# 4 wins for the Cavs.

# Use the mean function to find the proportion of simulations in which the Cavs
# win at least 4 of the remaining games. Print your answer to the console.

# The variable `B` spcifies the number of times we want the simulation to run. Let's
# run the Monte Carlo simulation 10,000 times.

B <- 10000

# Use the set.seed function to make sure your answer matches the expected result
# after random sampling.

set.seed(1, sample.kind = "Rounding")

# Create an object called `results` that replicates for `B` iterations a simulated
# series and determines whether that series contains at least four wins for the Cavs.

results <- replicate(B, {
  series <- sample(0:1, 6, replace = TRUE)
  sum(series) > 3
})

# Calculate the frequency out of `B` iterations that the Cavs won at least four games
# in the remainder of the series. Print your answer to the console.

mean(results)

# Exercise 3. A and B play a series - part 1

# Two teams, A and B, are playing a seven series game series. Team A is better than
# team B and has a p > 0.5 chance of winning each game.

# Use the function sapply to compute the probability, call it Pr of winning for

p <- seq(0.5, 0.95, 0.025)

# Then plot the result plot(p, Pr)

# Given a value `p`, the probability of winning the series for the underdog team B can
# be computed with the following function based on a Monte Carlo simulation:

prob_win <- function(p){
  B <- 10000
  result <- replicate(B, {
    b_win <- sample(c(1,0), 7, replace = TRUE, prob = c(1-p, p))
    sum(b_win) >= 4
  })
  mean(result)
}

# Apply the prob_win function across the vector of probabilities that team A will win to
# determine the probability that team B will win. Call this object Pr.

Pr <- sapply(p, prob_win)

# Plot the probability `p` on the x-axis and `Pr` on the y-axis.

plot(p, Pr)

# Exercise 4. A and B play a series - part 2

# Repeat the previous exercise, but now keep the probability that team A wins fixed at
# p <- 0.75 and compute the probability for different series lengths. For example, wins in
# best of 1 game, 3 games, 5 games, and so on through a series that lasts 25 games.

# Use the seq function to generate a list of odd numbers ranging from 1 to 25.

# Use the function sapply to compute the probability, call it Pr, of winning during a series
# of different lengths.

# Then plot the result plot(N, Pr).

# Given a value `p`, the probability of winning the series for the underdog team B can be
# computed with the following function based on a Monte Carlo simulation:

prob_win <- function(N, p = 0.75){
  B <- 10000
  result <- replicate(B, {
    b_win <- sample(c(1,0), N, replace = TRUE, prob = c(1-p, p))
    sum(b_win) >= (N+1) / 2
  })
  mean(result)
}

# Assign the variable N as the vector of series lengths. Use only odd numbers ranging from
# 1 to 25 games.

N <- seq(1, 25, 2)

# Apply the prob_win function across the vector of series lengths to determine the
# probability that team B will win. Call this object `Pr`.

Pr <- sapply(N, prob_win)

# Plot the number of games in the series N on the x-axis and Pr on the y-axis

plot(N, Pr)

# 1.4. ASSESSMENT: DISCRETE PROBABILITY

library(gtools)
library(tidyverse)

# Question 1: Olympic running

# In the 200m dash finals in the Olympics, 8 runners compete for 3 medals (order matters). In
# the 2012 Olympics, 3 of the 8 runners were from Jamaica and the other 5 were from different
# countries. The three medals were all won by Jamaica (Usain Bolt, Yohan Blake, and Warren Weir).

# Use the information above to help you answer the following four questions.

# Question 1a

# How many different ways can the 3 medals be distributed across 8 runners?

nrow(permutations(8, 3))

# Question 1b

# How many different ways can the three medals be distributed among the 3 runners from Jamaica?

nrow(permutations(3, 3))

# Question 1c

# What is the probability that all 3 medals are won by Jamaica?

1 / nrow(combinations(8, 3))

# Q1 by the book

medals <- permutations(8, 3)
nrow(medals)

jamaica <- permutations(3, 3)
nrow(jamaica)

nrow(jamaica) / nrow(medals) # 0.017

# Question 1d

# Run a Monte Carlo simulation on this vector representing the countries of the 8 runners in this
# race:

runners <- c("Jamaica", "Jamaica", "Jamaica", "USA", "Ecuador", "Netherlands", "France", "South Africa")

# For each iteration of the Monte Carlo simulation, within a replicate() loop, select 3 runners
# representing the 3 medalists and check whether they are all from Jamaica. Repeat this simulation
# 10,000 times. Set the seed to 1 before running the loop.

# Calculate the probability that all the runners are from Jamaica.

set.seed(1, sample.kind = "Rounding")

B <- 10000

all_jamaica <- replicate(B, {
  results <- sample(runners, 3)
  all(results == "Jamaica")
})
mean(all_jamaica)

# Question 2: Restaurant management

# A restaurant manager wants to advertise that his lunch special offers enough choices to eat
# different meals every day of the year. He doesn't think his current special actually allows
# that number of choices, but wants to change his special if needed to allow at least 265 choices.

# A meal at the restaurant includes 1 entree, 2 sides, and 1 drink. He currently offers a choice
# of 1 entree from a list of 6 options, a choice of 2 different sides from a list of 6 options,
# and a choice of 1 drink from a list of 2 options.

# Question 2a

# How many meal combinations are possible with the current menu?

entrees <- nrow(combinations(6, 1))
sides <- nrow(combinations(6, 2))
drinks <- nrow(combinations(2, 1))

entrees * sides * drinks # 180

# Question 2b

# The manager has one additional drink he could add to the special.

# How many combinations are possible if he expands his original special to 3 drink options?

drinks <- nrow(combinations(3, 1))

entrees * sides * drinks # 270

# Question 2c

# The manager decides to add the third drink but needs to expand the number of options. The
# manager would prefer not to change his menu further and wants to know if he can meet his
# goal by letting customers choose more sides.

# How many meal combinations are there if customers can choose from 6 entrees, 3 drinks,
# and select 3 sides from the current 6 options?

sides <- nrow(combinations(6, 3))

entrees * sides * drinks # 360

# Question 2d

# The manager is concerned that customers may not want 3 sides with their meal. He is
# willing to increase the number of entree choices instead, but if he adds too many
# expensive options it could eat into profits. He wants to know how many entree choices
# he would have to offer in order to meet his goal.

# Write a function that takes the number of entree choices and returns the number of
# meal combinations possible given that the number of entree options, 3 drink choices,
# and a selection of 2 sides from 6 options.

# Use sapply() to apply the function to entree option counts ranging from 1 to 12.

# What is the minimum number of entree options required in order to generate more
# than 365 combinations?

meals <- function(n){
  entrees <- nrow(combinations(n, 1))
  sides <- nrow(combinations(6, 2))
  drinks <- nrow(combinations(3, 1))
  entrees * sides * drinks
}

meal_combinations <- sapply(1:12, meals)

meal_combinations >= 365

data.frame(entrees = 1:12, combos = meal_combinations) %>%
  filter(combos > 365) %>% min(.$entrees) # 9

# Questions 2e

# The manager isn't sure he can afford to put that many entree choices on the lunch
# menu and thinks it would be cheaper for him to expand the number of sides. He
# wants to know how many sides he would have to offer to meet his goal of at least
# 365 combinations.

# Write a function that takes a number of side choices and returns the number of meal
# combinations possible given 6 entree choices, 3 drink choices, and a selection of 
# 2 sides from the specified number of side choices.

# Use sapply() to apply the function to side counts ranging from 2 to 12.

# What is the minimum number of side options required in order to generate more than
# 365 combinations?

side_choices <- function(n){
  6 * nrow(combinations(n, 2)) * 3
}

combos <- sapply(2:12, side_choices)

data.frame(sides = 2:12, combos = combos) %>%
  filter(combos > 365) %>% min(.$sides) # 7

# Questions 3 and 4: Esophageal cancer and alcohol/tobacco use, part 1

# Case-control studies help determine whether certain exposures are associated with
# outcomes such as developing cancer. The built-in dataset esoph contains data from
# a case-control study in France comparing  people with esophageal cancer (cases,
# counted in ncases) to people without esophageal cancer (controls, counted in
# ncontrols) that are carefully matched on a variety of demographic and medical
# characteristics. The study compares alcohol intake in grams per day (alcgp) and
# tobacco intake in grams per day (tobgp) across cases and controls grouped by
# age range (agegp).

# The dataset is available in base R and can be called with the variable name esoph:

head(esoph)

# Each row contains one group of the experiment. Each group has a different combination
# of age, alcohol consumption, and tobacco consumption. The number of cancer cases and
# number of controls (individual without cancer) are reported for each group.

# Question 3a

# How many groups are in the study?

nrow(esoph)

# Question 3b

# How many cases are there? Save this value as all_cases for later questions.

all_cases <- esoph %>% summarize(cases = sum(ncases)) %>% .$cases

# Simpler option:

all_cases <- sum(esoph$ncases)

# Question 3c

# How many controls are there? Save this value as all_controls for later.

all_controls <- sum(esoph$ncontrols)

# Question 4a

# What is the probability that a subject in the highest alcohol consuption
# group is a cancer case?

esoph %>% filter(alcgp == "120+") %>% summarize(p = sum(ncases) / (sum(ncases) + sum(ncontrols))) %>%
  .$p # 0.401

# Question 4b

# What is the probability that a subject in the lowest alcohol consumption group is a cancer case?

esoph %>% filter(alcgp == "0-39g/day") %>%
  summarize(p = sum(ncases) / (sum(ncases) + sum(ncontrols))) %>% .$p

# Questions 4c

# Given that a person is a case, what is the probability that they smoke 10g or more a day?

cases <- esoph %>% filter(ncases > 0) %>%
  summarize(cases = sum(ncases)) %>% .$cases

smokers <- esoph %>% filter(ncases > 0 & tobgp %in% c("10-19", "20-29", "30+")) %>%
  summarize(cases = sum(ncases)) %>% .$cases

smokers / cases # 0.61

# By the book

tob_cases <- esoph %>%
  filter(tobgp != "0-9g/day") %>%
  pull(ncases) %>% sum()

tob_cases / all_cases

# Question 4d

# Given that a person is a control, what is the probability that they smoke 10g or
# more a day?

tob_control <- esoph %>%
  filter(tobgp != "0-9g/day") %>%
  pull(ncontrols) %>% sum()

tob_control / all_controls

# Questions 5 and 6: Esophageal cancer and alcohol/tobacco use, part 2

# Question 5a

# For cases, what is the probability of being in the highest alcohol group?

alcohol_cases <- esoph %>%
  filter(alcgp == "120+") %>%
  pull(ncases) %>% sum()

alcohol_cases / all_cases

# Question 5b

# For cases, what is the probability of being in the highest tobacco group?

tob_cases <- esoph %>%
  filter(tobgp == "30+") %>%
  pull(ncases) %>% sum()

tob_cases / all_cases

# Question 5c

# For cases, what is the probability of being in the highest alcohol group and
# the highest tobacco group?

tob_alc_cases <- esoph %>%
  filter(tobgp == "30+" & alcgp == "120+") %>%
  pull(ncases) %>% sum()

tob_alc_cases / all_cases

# For cases, what is the probability of being in the highest alcohol group or
# the highest tobacco group?

tob_or_alc_cases <- esoph %>%
  filter(tobgp == "30+" | alcgp == "120+") %>%
  pull(ncases) %>% sum()

tob_or_alc_cases / all_cases

# Question 6a

# For controls, what is the probability of being in the highest alcohol group?

alc_controls <- esoph %>%
  filter(alcgp == "120+") %>%
  pull(ncontrols) %>% sum()

alc_controls / all_controls

# Question 6b

# How many times more likely are cases than controls to be in the highest alcohol
# group?

(alcohol_cases / all_cases) / (alc_controls / all_controls)

# Question 6c

# For controls, what is the probability of being in the highest tobacco group?

high_tob_control <- esoph %>%
  filter(tobgp == "30+") %>%
  pull(ncontrols) %>% sum()

high_tob_control / all_controls

# Question 6d

# For controls, what is the probability of being in the highest alcohol group
# and the highest tobacco group?

high_both_controls <- esoph %>%
  filter(tobgp == "30+" & alcgp == "120+") %>%
  pull(ncontrols) %>% sum()

high_both_controls / all_controls

# Question 6e

# For controls, what is the probability of being in the highest alcohol group
# or the highest tobbaco group?

high_or_ctrl <- esoph %>%
  filter(tobgp == "30+" | alcgp == "120+") %>%
  pull(ncontrols) %>% sum()

high_or_ctrl / all_controls

# Question 6f

# How many times more likely are cases than controls to be in the highest
# alcohol group or the highest tobacco group?

(tob_or_alc_cases / all_cases) / (high_or_ctrl / all_controls)

## SECTION 2. CONTINUOUS PROBABILITY

# The cumulative distribution function (CDF) is a distribution function for continuous
# data x that reports the proportion of the data below a for all values of a:

# F(a) = Pr(x <= a)

# The CDF is the probability distribution function for continuous variables. For example,
# to determine the probability that a male student is higher than 70.5 inches given a 
# vector of male heights x, we can use the CDF:

# Pr(x > 70.5) = 1 - Pr(x <= 70.5) = 1 - F(70.5)

# The probability that an observation is in between two values a, b is F(b) - F(a).

# Cumulative distribution function

library(tidyverse)
library(dslabs)
data(heights)
x <- heights %>% filter(sex == "Male") %>% pull(height)

# Given a vector x, we can define a function for computing the CDF of x using:

F <- function(a) mean(x <= a)
1 - F(70) # probability of male taller than 70 inches ~37.7%

# Theoretical distribution

# pnorm(a, avg, s) gives the value of the cumulative distribution function F(a) for
# the normal distribution defined by average avg and standard deviation s.

# We say that a random quantity is normally distributed with average avg and standard
# deviation s if the approximation pnorm(a, avg, s) holds for all values of a.

# If we are willing to use the normal approximation for height, we can estimate the
# distribution simply from the mean and standard deviation of our values.

# If we treat the height data as discrete rather than categorical, we see that the data
# are not very useful because integer values are more common than expected due to
# rounding. This is called discretization.

# With rounded data, the normal approximation is particularly useful when computing
# probabilities of intervals of length 1 that include exactly one integer.

# Using pnorm() to calculate probabilities

# Given male heights x, we can estimate the probability that a male is taller than
# 70.5 inches using:

1 - pnorm(70.5, mean(x), sd(x))

# Discretization and the normal approximation

# Plot distribution of exact heights in data

plot(prop.table(table(x)), xlab = "a = Height in inches", ylab = "Pr(x = a)")

# Probabilities in actual data over length 1 ranges containing an integer

mean(x <= 68.5) - mean(x <= 67.5)
mean(x <= 69.5) - mean(x <= 68.5)
mean(x <= 70.5) - mean(x <= 69.5)

# Probabilities in normal approximation match well

pnorm(68.5, mean(x), sd(x)) - pnorm(67.5, mean(x), sd(x))
pnorm(69.5, mean(x), sd(x)) - pnorm(68.5, mean(x), sd(x))
pnorm(70.5, mean(x), sd(x)) - pnorm(69.5, mean(x), sd(x))

# Probabilities in actual data over other ranges don't match normal approx well

mean(x <= 70.9) - mean(x <= 70.1)
pnorm(70.9, mean(x), sd(x)) - pnorm(70.1, mean(x), sd(x))

# Probability density

# The probability of a single value is not defined for a continuous distribution.

# The quantity with the most similar interpretation to the probability of a single
# value is the probability density function f(x).

# The probability density f(x) is defined such that integral of f(x) over a range
# gives the CDF of that range.

# F(a) = Pr(X <= a) = Int[-inf, a] f(x) dx

# In R, the probability density function for the normal distribution is given by
# dnorm(). We will see uses of dnorm() in the future.

# Note that dnorm() gives the density function and pnorm() gives the distribution
# function, which is the integral of the density function.

# Plotting the probability density

# We can use dnorm() to plot the density curve for the normal distribution. dnorm(z)
# gives the probability density f(z) of a certain z-score, so we can draw a curve
# by calculating the density over a range of possible values of z.

# First, we generate a series of z-scores covering the typical range of the normal
# distribution. Since we know 99.7% of observations will be within -3 <= z <= 3,
# we can use a value of z slightly larger than 3 and this will cover most likely
# the values of the normal distribution. Then, we calculate f(z), which is dnorm()
# of the series of z-scores. Last, we plot z against f(z)

x <- seq(-4, 4, length = 100)
data.frame(x, f = dnorm(x)) %>%
  ggplot(aes(x, f)) +
  geom_line()

# Note that dnorm() gives densities for the standard normal distribution by default.
# Probabilities for alternative normal distributions with mean `mu` and standard
# deviation `sigma` can be evaluated with: dnorm(z, mu, sigma)

# Monte Carlo simulations

# rnorm(n, avg, s) generates n random numbers from the normal distribution with 
# average avg and standard deviation s.

# By generating random numbers from the normal distribution, we can simulate height
# data with similar properties to our dataset. Here we generate simulated height
# data using the normal distribution.

# Generating normally distributed random numbers

# Define x as male heights from dslabs

x <- heights %>% filter(sex == "Male") %>% pull(height)

# Generate simulated height data using normal distribution - both datasets have n observations.

n <- length(x)
avg <- mean(x)
s <- sd(x)
simulated_heights <- rnorm(n, avg, s)

# Plot distribution of simulated heights

data.frame(simulated_heights = simulated_heights) %>%
  ggplot(aes(simulated_heights)) +
  geom_histogram(color = "black", binwidth = 2)

# Monte Carlo simulation of tallest person over 7 feet

B <- 10000
tallest <- replicate(B, {
  simulated_data <- rnorm(800, avg, s) # generate 800 normally distributed random heights
  max(simulated_data) # determine the tallest height
})
mean(tallest >= 7 * 12) # proportion of times that tallest person exceeded 7 feet (84 inches).

# Other continuous distributions

# You may encounter other continuous distributions (Student t, chi-squared, exponential, gamma,
# beta, etc.).

# R provides functions for density (d), quantile (q), probability distribution (p), and random
# number generation (r) for many of this distributions.

# Each distribution has a matching abbreviation (for example, norm() or t()) that is paired with
# the related function abbreviations (d, p, q, r) to create appropriate functions.

# For example, use rt() to generate random numbers for a Monte Carlo simulation using the
# Student t distribution.

t <- rt(800, avg, s)
data.frame(t = t) %>%
  ggplot(aes(t)) +
  geom_histogram(color = "black", binwidth = 2)

# DataCamp Assessment: Continuous Probability

# Exercise 1. Distribution of female heights - part 1

# Assume the distribution of female heights is approximated by a normal distribution with a 
# mean of 64 inches and a standard deviation of 3 inches. If we pick a female at random, what
# is the probability that she is 5 feet or shorter?

# Use pnorm() to define the probability that a height will take a value less than 5 feet
# given the stated distribution.

# Assign a variable `female_avg` as the average female height.

female_avg <- 64

# Assign a variable `female_sd` as the standard deviation for female heights.

female_sd <- 3

# Using variables `female_avg` and `female_sd`, calculate the probability that a randomly
# selected female is shorter than 5 feet. Print this value to the console.

pnorm(5 * 12, female_avg, female_sd)

# Exercise 2. Distribution of female heights - part 2

# Assume the distribution of female heights is approximated by a normal distribution with a
# mean of 64 inches and a standard deviation of 3 inches. If we pick a female at random, what
# is the probability that she is 6 feet or taller?

# Use pnorm to define the probability that a height will take a value of 6 feet or taller.

# Using variables `female_avg` and `female_sd`, calculate the probability that a randomly
# selected female is 6 feet or taller. Print this value to the console.

1 - pnorm(6 * 12, female_avg, female_sd)

# Exercise 3. Distribution of female heights - part 3

# Assume the distribution of female heights is approximated by a normal distribution with a
# mean of 64 inches and a standard deviation of 3 inches. If we pick a female at random, what
# is the probability that she is between 61 and 67 inches?

# Use pnorm() to define the probability that a randomly chosen woman will be shorter than
# 67 inches.

# Substract the probability that a randomly chose woman will be shorter than 61 inches.

# Using variables `female_avg` and `female_sd`, calculate the probability that a randomly
# selected female is between the desired height range. Print this value to the console.

pnorm(67, female_avg, female_sd) - pnorm(61, female_avg, female_sd)

# Exercise 4. Distribution of female heights - part 4

# Repeat the previous exercise, but convert everything to centimeters. That is, multiply
# every height, including the standard deviation, by 2.54. What is the answer now?

# Convert the average height and standard deviation to centimeters by multiplying each
# value by 2.54.

# Repeat the previous calculation using pnorm() to define the probability that a randomly
# choen woman will have a height between 61 and 67 inches, converted to centimeters by
# multiplying each value by 2.54.

# Assign a variable `female_avg` as the average female height. Convert this value to
# centimeters.

female_avg <- 64 * 2.54

# Assign a variable `female_sd` as the standard deviation for female heights. Convert
# this value to centimeters.

female_sd <- 3 * 2.54

# Using variables `female_avg` and `female_sd`, calculate the probability that a randomly
# selected female is between the desired height range. Print this value to the console.

pnorm(67 * 2.54, female_avg, female_sd) - pnorm(61 * 2.54, female_avg, female_sd)

# Exercise 5. Probability of 1 SD from average

# Compute the probability that the height of a randomly chosen female is within 1 sd from
# the average height.

# Calculate the values for heights one standard deviation taller and shorter than the
# average.

# Calculate the probability that a randomly chosen woman will be within 1 sd from the
# average height.

# Assign a variable `female_avg` as the average female height.

female_avg <- 64

# Assign a variable `female_sd` as the standard deviation for female heights.

female_sd <- 3

# To a variable named `taller`, assign the value of a height that is one sd taller than
# average.

taller <- female_avg + female_sd

# To a variable named `shorter`, assign the value of a height that is one sd shorter
# than average.

shorter <- female_avg - female_sd

# Calculate the probability that a randomly selected female is between the desired height
# range. Print this value to the console.

pnorm(taller, female_avg, female_sd) - pnorm(shorter, female_avg, female_sd)

# Exercise 6. Distribution of male heights

# Imagine the distribution of male adults is approximately normal with an average of 69
# inches and a standard deviation of 3 inches. How tall is a male in the 99th percentile?

# Determine the height of a man in the 99th percentile, given an average height of 69
# inches and a standard deviation of 3 inches.

# Assign a variable `male_avg` as the average male height.

male_avg <- 69

# Assign a variable `male_sd` as the standard deviation for male heights.

male_sd <- 3

# Determine the height of a man in the 99th percentile of the distribution.

qnorm(0.99, male_avg, male_sd)

# Exercise 7. Distribution of IQ scores

# The distribution of IQ scores is approximately normally distributed. The average is
# 100 and the standard deviation is 15. Suppose you want to know the distribution of
# the person with the highest IQ in your school district, where 10,000 people are
# born each year.

# Generate 10,000 IQ scores 1,000 times using a Monte Carlo simulation. Make a histogram
# of the highest IQ scores.

# Use the function rnorm() to generate a random distribution of 10,000 values with a given
# average and standard deviation.

# Use the function max() to return the largest value from a supplied vector.

# Repeat the previous steps a total of 1,000 times. Store the vector of the top 1,000 IQ
# scores as highestIQ.

# Plot the histogram of values using the function hist().

# The variable `B` specified the number of times we want the simulation to run.

B <- 1000

# Use the `set.seed` function to make sure your answer matches the expected result after
# random number generation.

set.seed(1, sample.kind = "Rounding")

# Create an object called `highestIQ` that contains the highest IQ score from each random
# distribution of 10,000 people.

highestIQ <- replicate(B, {
  IQ_scores <- rnorm(10000, 100, 15)
  max(IQ_scores)
})

# Make a histogram of the highest IQ scores.

hist(highestIQ)

# 2.2. ASSESSMENT: CONTINUOUS PROBABILITY

# Questions 1 and 2. ACT scores, part 1

# The ACT is a standardized college admissions test used in the United States. The four
# multi-part questions in this assessment involve some ACT test scores and answering 
# probability questions about them.

# For the three year period 2016-2018, ACT standardized test scores were approximately
# normally distributed with a mean of 20.9 and standard deviation of 5.7. (Real ACT scores
# are integers between 1 and 36, but we will ignore this detail and use continuous values
# instead.)

# First we'll simulate an ACT test score dataset and answer some questions about it.

# Set the seed to 16, then use the rnorm() to generate a normal distribution of 10000
# tests with a mean of 20.9 and standard deviation of 5.7. Save these values as act_scores.
# You'll be using this dataset throughout these four multi-part questions.

set.seed(16, sample.kind = "Rounding")

# Question 1a

# What is the mean of act_scores?

act_scores <- rnorm(10000, 20.9, 5.7)
mean(act_scores) # 20.84

# Question 1b

# What is the standard deviation of act_scores?

sd(act_scores) # 5.675

# Question 1c

# A perfect score of 36 or greater (the maximum reported score is 36).

# In act_scores, how many perfect scores are there out of 10,000 simulated tests?

sum(act_scores >= 36) # 41

# Question 1d

# In act_scores, what is the probability of an ACT score greater than 30?

1 - mean(act_scores <= 30)

# also...

mean(act_scores > 30)

# Question 1e

# In act_scores, what is the probability of an ACT score less than or equal to 10?

mean(act_scores <= 10)

# Question 2

# Set x equal to the sequence of integers 1 to 36. Use dnorm() to determine the value
# of the probability density function over x given a mean of 20.9 and standard deviation
# of 5.7; save the result as f_x. Plot x against f_x

x <- 1:36
f_x <- dnorm(x, 20.9, 5.7)
data.frame(x, f_x) %>%
  ggplot(aes(x, f_x)) +
  geom_line()

# Which of the plots is correct? # 2 was the correct plot

# Questions 3 and 4. ACT scores, part 2

# In this 3-part question, you will convert raw ACT scores to z-scores and answer some
# questions about them.

# Covert act_scores to z-scores. Recall from data visualization (the second course in the
# series) that to standardize values (convert values into z-scores, that is, values distributed
# with a mean of 0 and standard deviation of 1), you must substract the mean and then divide by
# the standard deviation. Use the mean and standard deviation of act_scores, not the original
# values used to generate random test scores.

# Question 3a

# What is the probability of a z-score greater than 2?

z_scores <- (act_scores - mean(act_scores)) / sd(act_scores)
mean(z_scores > 2) # 0.023

# Question 3b

# What ACT score value corresponds to 2 standard deviations above the mean (z = 2)?

mean(act_scores) + sd(act_scores) * 2 # 32.19

# Question 3c

# A z-score of 2 correspondents roughly to the 97.5th percentile.

# Use qnorm() to determine the 97.5th percentile of normally distributed data with the mean
# and standard deviation observed in act_scores.

# What is the 97.5th percentile of act_scores?

qnorm(0.975, mean(act_scores), sd(act_scores))

# In this 4-part question, you will write a function to create a CDF for ACT scores.

# Write a function that takes a value and produces the probability of an ACT score less than
# or equal to that value (the CDF). Apply this function to the range 1 to 36.

# Question 4a

# What is the minimum integer score such that the probability of that score or lower is
# at least .95? Your answer should be an integer 1-36.

f <- function(s){
  mean(act_scores <= s)
}

scores <- sapply(1:36, f)

data.frame(x = 1:36, p = scores) %>%
  filter(p >= .95) %>% pull(x) %>% min()

# simpler code...

cdf <- sapply(1:36, function(x){
  mean(act_scores <= x)
})
min(which(cdf >= .95))

# Question 4b

# Use qnorm() to determine the expected 95th percentile, the value for which the probability
# of receiving that score or lower is 0.95, given a mean score of 20.9 and standard devition
# of 5.7.

# What is the expected 95th percentile of ACT scores?

qnorm(0.95, 20.9, 5.7)

# Question 4c

# As discussed in the data visualization course, we can still use quantile() to determine
# sample quantiles from the data.

# Make a vector containing the quantiles for p <- seq(0.01, 0.99, 0.01), the 1st through
# 99th percentiles of the act_scores. Save these as sample_quantiles.

# In what percentile is a score of 26? Your answer should be an integer (i.e., 60), not a
# percent or fraction. Note that a score between the 98th and 99th percentile should be
# considered the 98th percentile, for example, and that quantile numbers are used as names
# for the vector sample_quantiles.

p <- seq(0.01, 0.99, 0.01)
sample_quantiles <- quantile(act_scores, p)
names(sample_quantiles[max(which(sample_quantiles < 26))])

# Question 4d

# Make a corresponding set of theoretical quantiles using qnorm() over the interval
# p <- seq(0.01, 0.99, 0.01) with mean 20.9 and standard deviation 5.7. Save these as
# theoretical_quantiles. Make a Q-Q plot graphing sample_quantiles on the y-axis versus
# theoretical_quantiles on the x-axis.

theoretical_quantiles <- qnorm(p, 20.9, 5.7)
plot(theoretical_quantiles, sample_quantiles)

# Which of the following graphs is correct? # 4

# Better with qplot...

qplot(theoretical_quantiles, sample_quantiles) + geom_abline()

## SECTION 3. RANDOM VARIABLES, SAMPLING MODELS, AND THE CENTRAL LIMIT THEOREM

# 3.1. RANDOM VARIABLES AND SAMPLING MODELS

# Random variables are numeric outcomes resulting from a random process.

# Statistical inference offers a framework for quantifying uncertainty due to
# randomness.

# Define random variable x to be 1 if blue, 0 otherwise
beads <- rep(c("red", "blue"), times = c(2, 3))
x <- ifelse(sample(beads, 1) == "blue", 1, 0)

# Demonstrate that the random variable is different every time
ifelse(sample(beads, 1) == "blue", 1, 0) # 1
ifelse(sample(beads, 1) == "blue", 1, 0) # 0
ifelse(sample(beads, 1) == "blue", 1, 0) # 1

# SAMPLING MODELS

# A sampling model models the random behavior of a process as the sampling of
# draws from an urn.

# The probability distribution of a random variable is the probability of the
# observed value falling in any given interval.

# We can define a CDF F(a) = Pr(S <= a) to answer questions related to the probability
# of S being in any interval.

# The average of many draws of a random variable is called the expected value.

# The standard deviation of many draws of a random variable is called its
# standard error.

# Monte Carlo simulation: Chance of a casino losing money on roulette

# We build a sampling model for the random variable S that represents the
# casino's total winnings

# Sampling model 1: Define urn, then sample
color <- rep(c("Black", "Red", "Green"), c(18, 18, 2)) # Define the urn for the sampling model
n <- 1000
X <- sample(ifelse(color == "Red", -1, 1), n, replace = TRUE)
X[1:10]

# Sampling model 2: Define urn inside sample function by noting probabilities
X <- sample(c(-1, 1), n, replace = TRUE, prob = c(9/19, 10/19)) # 1000 independent draws
S <- sum(X) # Total winnings = sum of draws
S

# We use the sampling model to run a Monte Carlo simulation and use the results to
# estimate the probability of the casino losing money

n <- 1000 # Number of roulette players
B <- 10000 # Number of Monte Carlo experiments
S <- replicate(B, {
  X <- sample(c(-1, 1), n, replace = TRUE, prob = c(9/19, 10/19)) # Simulate 1000 spins
  sum(X) # Determine total profit
})

mean(S < 0) # Probability of the casino losing money

# We can plot a histogram of the observed values of S as well as the normal density
# curve based on the mean and standard deviation of S.

library(tidyverse)
s <- seq(min(S), max(S), length = 100) # Sequence of 100 values across range of S
normal_density <- data.frame(s = s, f = dnorm(s, mean(S), sd(S))) # Generate normal density for S
data.frame(S = S) %>% # Make data frame of S for histogram
  ggplot(aes(S, ..density..)) +
  geom_histogram(color = "black", binwidth = 10) +
  ylab("Probability") +
  geom_line(data = normal_density, mapping = aes(s, f), color = "blue")

# DISTRIBUTIONS VS. PROBABILITY DISTRIBUTIONS

# A random variable X has a probability distribution function F(a) that defines
# Pr(X <= a) over all values of a.

# Any list of numbers has a distribution. The probability distribution function of
# a random variable is defined mathematically and does not depend on a list of numbers.

# The results of a Monte Carlo simulation with a large enough number of observations
# will approximate the probability distribution of X.

# If a random variable is defined as draws from an urn:

# 1. The probability distribution function of the random variable is defined as the
# distribution of the list of values in the urn.

# 2. The expected value of the random variable is the average of values in the urn.

# 3. The standard error of one draw of the random variable is the standard deviation
# of the values of the urn.

# NOTATION FOR RANDOM VARIABLES

# Capital letters denote random variables (X) and lowercase letters denote observed
# variables (x).

# In the notation Pr(X = x), we are asking how frequently the random variable X is
# equal to the value x. For example, if x = 6, this statement becomes Pr(X = 6).

# CENTRAL LIMIT THEOREM

# The Central Limit Theorem (CLT) says that the distribution of the sum of a random
# variable is approximated by a normal distribution.

# The expected value of a random variable, E[X] = `mu`, is the average of the values
# in the urn. This represents the expectation of one draw.

# The standard error of one draw of a random variable is the standard deviation of the
# values in the urn.

# The expected value of the sum of draws is the number of draws times the expected
# value of the random variable.

# The standard error of the sum of independent draws of a random variable is the
# square root of the number of draws times the standard deviation of the urn.

# Equations

# These equations apply to the case where there are only two outcomes, a and b with
# proportions p and 1-p respectively. The general principles above also apply to random
# variables with more than two outcomes.

# Expected value of a random variable: a * p + b * (1 - p)

# Expected value of the sum of n draws of a random variable: n * (a * p + b * (1-p))

# Standard deviation of an urn with two values: |b - a| * sqrt(p * (1-p))

# Standard error of the sum of n draws of a random variable: sqrt(n) * |b - a| * sqrt(p * (1-p))

# DataCamp Assessment: Random Variables and Sampling Models

# Exercise 1. American Roulette probabilities

# An American roulette wheel has 18 red, 18 black, and 2 green pockets. Each red
# and black pocket is associated with a number from 1 to 36. The two remaining
# green slots feature "0" and "00". Players place bets on which pocket they think
# a ball will land in after the wheel is spun. Players can bet on a specific number
# (0, 00, 1-36) or color (red, black, green).

# What are the chances that the ball lands in a green pocket?

# Define a variable p_green as the probability of the ball landing in a green pocket.

# Print the value of p_green.

# The varuiables `green`, `black`, and `red` contain the number of pockets for each
# color:
green <- 2
black <- 18
red <- 18

# Assign a variable `p_green` as the probability of the ball landing in a green
# pocket.

p_green <- green / (green + black + red)

# Print the variable `p_green` to the console

p_green

# Exercise 2. American Roulette payout

# In American Roulette, the payout for winning on green is $17. This means that if
# you bet $1 and it lands on green, you get $17 as a prize.

# Create a model to predict your winnings from betting on green one time.

# Use the sample function return a random value from a specified range of values.

# Use the prob = argument in the sample function to specify a vector of probabilities
# for returning each of the values contained in the vector of values being sampled.

# Take a single sample (n = 1)

# Use the `set.seed` function to make sure your answer matches the expected result
# after random sampling.
set.seed(1, sample.kind = "Rounding")

# The variables `green`, `black`, and `red` contain the number of pockets for each
# color
green <- 2
black <- 18
red <- 18

# Assign a variable `p_green` as the probability of the ball landing in a green
# pocket
p_green <- green / (green + black + red)

# Assign a variable `not_p_green` as the probability of the ball not landing in
# a green pocket

p_not_green <- 1 - p_green

# Create a model to predict the random variable `X`, your winnings from betting
# on green. Sample one time.

X <- sample(c(17, -1), 1, prob = c(p_green, p_not_green))

# Print the value of `X` to the console
X

# Exercise 3. American Roulette expected value

# In American Roulette, the payout for winning on green is $17. This means that if
# you bet $1 and it lands on green, you get $17 as a prize. In the previous exercise,
# you created a model to predict your winnings from betting on green.

# Now, compute the expected value of X, the random variable you generated previously.

# Using the chances of winning $17 (p_green) and the chances of losing $1 (p_not_green),
# calculate the expected outcome of a bet that the ball will land in a green pocket.

(17 * p_green) + (-1 * p_not_green) # ~ -$0.05

# Exercise 4. American Roulette Standard error

# The standard error of a random variable X tells us the difference between a random
# variable and its expected value. You calculated a random variable X in exercise 2
# and the expected value of that random variable in exercise 3.

# Now, compute the standard error of that random variable, which represents a
# single outcome after one spin of the roulette wheel.

# Compute the standard error of the random variable you generated in exercise 2, or the
# outcome of any one spin of the roulette wheel.

# Recall that the payout for winning on green is $17 for a $1 bet.

# Compute the standard error of the random variable

abs(17 -(-1)) * sqrt(p_green * p_not_green)

# Exercise 5. American Roulette sum of winnings

# You modeled the outcome of a single spin of the roulette wheel, X, in exercise 2.

# Now create a random variable S that sums your winnings after betting on green
# 1,000 times.

# Use set.seed to make sure the result of your random operation matches the expected
# answer for this problem.

# Specify the number of times you want to sample from the possible outcomes.

# Use the sample function to return a random value from a vector of possible
# values.

# Be sure to assign a probability to each outcome and to indicate that you are
# sampling with replacement.

# Do not use replicate as this changes the output of random sampling and your
# answer will not match the grader.

# Define the number of bets using the variable `n`

n <- 1000

# Create a vector `X` that contains the outcomes of 1000 samples

X <- sample(c(17, -1), n, replace = TRUE, prob = c(p_green, p_not_green))

# Assign the sum of all 1000 outcomes to the variable `S`

S <- sum(X)

# Print the value of `S` to the console
S

# Exercise 6. American Roulette winnings expected value

# In the previous exercise, you generated a vector of random outcomes, S, after
# betting on green 1,000 times.

# What is the expected value of S?

# Using the chances of winning $17 (p_green) and the chances of losing $1
# (p_not_green), calculate the expected outcome of a bet that the ball will land in
# a green pocket over 1,000 bets.

# Calculate the expected outcome of 1,000 spins if you win $17 when the ball lands
# on green and you lose $1 when the ball doesn't land on green.

n * (17 * p_green + (-1 * p_not_green))

# Exercise 7. American Roulette winnings expected value

# You generated the expected value of S, the outcomes of 1,000 bets that the ball
# lands in the green pocket, in the previous exercise.

# What is the standard error of S?

# Compute the standard error of the random variable you generated in exercise 5, or
# the outcomes of 1,000 spins of the roulette wheel.

# Compute the standard error of the sum of 1,000 outcomes

sqrt(n) * abs(17 - (-1)) * sqrt(p_green * p_not_green)

# 3.2. THE CENTRAL LIMIT THEOREM (CONT'D)

# Averages and proportions

# Random variable times a constant

# The expected value of a random variable multiplied by a constant is that constant
# times its original expected value: E[aX] = a * `mu`

# The standard error of a random variable multiplied by a constant is that constant
# times its original standard error: SE[aX] = a * `sigma`

# Average of multiple draws of a random variable

# The expected value of the average of multiple draws from an urn is the expected
# value of the urn (`mu`)

# The standard deviation of the average of multiple draws from an urn is the standard
# deviation of the urn divided by the square root of the number of draws (`sigma`/ sqrt(n))

# The sum of multiple draws of a random variable

# The expected value of the sum of n draws of a random variable is n times its orginal
# expected value: E[nX] = n * `mu`

# The standard error of the sum of n draws of a random variable is sqrt(n) times its
# original standard error: SE[xN] = sqrt(n) * `sigma`

# The sum of multiple different random variables

# The expected value of the sum of different random variables is the sum of individual
# expected values for each random variable: E[X1 + X2 +···+ Xn] = `mu`1 + `mu`2 +···+ `mu`n

# The standard error of the sum of different random variables is the square root of the
# sum of squares of the individual standard errors:
# SE[X1 + X2 +···+ Xn] = sqrt(`sigma`1^2 + `sigma`2^2 +···+ `sigma`n^2)

# Transformation of random variables

# If X is a normally distributed random variable and a and b are non-random constants,
# then aX + b is also a normally distributed random variable.

# Law of large numbers

# The law of large numbers states than as n increases, the standard error of the average
# of a random variable decreases. In other words, when n is large, the average of the
# draws converges to the average of the urn.

# The law of large numbers is also known as the law of averages.

# The law of average only applies when n is very large and events are independent. It is
# often misused to make predictions about an event being "due" because it has happened
# less frequently than expected in a small sample size.

# How large is large in CLT?

# The sample size required for the Central Limit Theorem and Law of Large Numbers to
# apply differs based on the probability of success.

# If the probability of success is high, then relatively few observations are needed.

# As the probability of success decreases, more observations are needed.

# If the probability of success is extremely low, such as winning a lottery, then the
# Central Limit Theorem may not apply even with extremely large sample sizes. The normal
# distribution is not a good approximation in these cases, and other distributions such 
# as the Poisson distribution may be more appropriate.

# DatCamp Assessment: The Central Limit Theorem

# Exercise 1. American Roulette probability of winning money

# The exercises in the previous chapter explored winnings in American roulette. In this
# chapter of exercises, we will continue with the roulette example and add in the
# Central Limit Theorem.

# In the previous chapter of exercises, you created a random variable S that is the
# sum of your winnings after betting on green a number of times in American Roulette.

# What is the probability that you end up winning money if you bet on green 100 times?

# Execute tha sample code to determine the expected value avg and standard error se as
# you have done in previous exercises.

# Use the pnorm function to determine the probability of winning money.

# Assign a variable `p_green` as the probability of the ball landing in a green pocket.
p_green <- 2 / 38

# Assign a variable `p_not_green` as the probability of the ball not landing in a green
# pocket.
p_not_green <- 1 - p_green

# Define the number of bets using the variable `n`
n <- 100

# Calculate `avg`, the expected outcome of 100 spins if you win $17 when the ball lands
# on green and you lose $1 when the ball doesn't land on green
avg <- n * (17 * p_green + -1 * p_not_green)

# Compute `se`, the standard error of the sum of 100 outcomes
se <- sqrt(n) * (17 + 1) * sqrt(p_green * p_not_green)

# Using the expected value `avg` and standard error `se`, compute the probability that
# you win money betting on green 100 times.

1 - pnorm(0, avg, se)

# Exercise 2. American Roulette Monte Carlo simulation

# Create a Monte Carlo simulation that generates 10,000 outcomes of S, the sum of 100
# bets.

# Compute the average and standard deviation of the resulting list and compare them
# to the expected value (-5.263158) and standard error (40.19344) for S that you
# calculated previously.

# Use the replicate function to replicate the sample code for B <- 10000 simulations.

# Within replicate, use the sample function to simulate n <- 100 outcomes of either a
# win (17) or a loss (-1) for the bet. Use the order c(17, -1) and corresponding
# probabilities. Then, use the sum function to add up the winnings over all iterations
# of the model. Make sure to include sum or DataCamp may crash with a "Session Expired"
# error.

# Use the mean function to compute the average winnings.

# Use the sd function to compute the standard deviation of the winnings.

# Assign a variable `p_green` as the probability of the ball landing in a green pocket.
p_green <- 2 / 38

# Assign a variable `p_not_green` as the probability of the ball not landing in a green
# pocket.
p_not_green <- 1 - p_green

# Define the number of bets using the variable `n`
n <- 100

# The variable `B` specifies the number of times we want the simulation to run. Let's
# run the Monte Carlo simulation 10,000 times
B <- 10000

# Use the `set.seed` function to make sure your answer matches the expected result after
# random sampling
set.seed(1, sample.kind = "Rounding")

# Create an object called `S` that replicates the sample code for `B` iterations and
# sums the outcomes

S <- replicate(B, {
  X <- sample(c(17, -1), n, replace = TRUE, prob = c(p_green, p_not_green))
  sum(X)
})

# Compute the average value for `S`

mean(S)

# Compute the standard deviation of `S`

sd(S)

# Exercise 3. American Roulette Monte Carlo vs CLT

# In this chapter, you calculated the probability of winning money in American Roulette
# using the CLT.

# Now, calculate the probability of winning money from the Monte Carlo simulation. The
# Monte Carlo simulation from the previous exercise has already been pre-run for you,
# resulting in the variable S that contains a list of 10,000 simulated outcomes.

# Use the mean function to calculate the probability of winning money from the Monte
# Carlo simulation, S

# Calculate the proportion of outcomes in the vector `S` that exceed $0

mean(S > 0)

# Exercise 4. American Roulette Monte Carlo vs CLT comparison

# The Monte Carlos result and the CLT approximation for the probability of losing
# money after 100 bets are close, but not that close. What could account for this?

# 10,000 simulations is not enough. If we do more, the estimates will match.

# The CLT does not work as well when the probability of success is small. # Correct

# The difference is within rounding error.

# The CLT only works for the averages.

# Exercise 5. American Roulette average winnings per bet

# Now create a random variable Y that contains your average winnings per bet after
# betting on green 10,000 times.

# Run a single Monte Carlo simulation of 10,000 bets using the following steps (you
# do not need to replicate the sample code)

# Specify n as the number of times you want to sample from the possible outcomes

# Use the sample function to return n values from a vector of possible values: winning
# $17 or losing $1. Be sure to assign a probability to each outcome and indicate that
# you are sampling with replacement.

# Calculate the average result per bet placed using the mean function

# Use the `set.seed` function to make sure your answer matches the expected result
# after random sampling.
set.seed(1, sample.kind = "Rounding")

# Define the number of bets using the variable `n`
n <- 10000

# Assign a variable `p_green` as the probability of the ball landing in a green pocket
p_green <- 2 / 38

# Assign a variable `p_not_green` as the probability of the ball not landing in a green
# pocket
p_not_green <- 1 - p_green

# Create a vector called `X` that contains the outcomes of `n` bets

X <- sample(c(17, -1), n, replace = TRUE, prob = c(p_green, p_not_green))

# Define a variable `Y` that contains the mean outcome per bet. Print this mean to the
# console

Y <- mean(X)
Y

# Exercise 6. American Roulette per bet expected value

# What is the expected value of Y, the average outcome per bet after betting on green
# 10,000 times?

# Using the chances of winning $17 (p_green) and the chances of losing $1 (p_not_green),
# calculate the expected outcome of a bet that the ball will land in a green pocket.

# Use the expected value formula rather than a Monte Carlo simulation.

# Print this value to the console (do not assign it to a variable)

# Calculate the expected outcome of `Y`, the mean outcome per bet in 10,000 bets

17 * p_green + (-1) * p_not_green

# Exercise 7. American Roulette per bet standard error

# What is the standard error of Y, the average result of 10,000 spins?

# Compute the standard error of Y, the average result of 10,000 spins.

# Define the number of bets using the variable `n`
n <- 10000

# Compute the standard error of `Y`, the mean outcome per bet from 10,000 bets

abs(17 - -1) * sqrt(p_green * p_not_green) / sqrt(n)

# Exercise 8. American Roulette winnings per game are positive

# What is the probability that your winnings are positive after betting on green
# 10,000 times?

# Execute the code that we wrote in the previous exercises to determine the average
# and standard error

# Use the pnorm() function to determine the probability of winning more than $0

# We defined the average using the following code
avg <- 17 * p_green + -1 * p_not_green

# We defined the standard error using this equation
se <- 1 / sqrt(n) * (17 - -1) * sqrt(p_green * p_not_green)

# Given this average and standard error, determine the probability of winning more
# than $0. Print the result to the console.

1 - pnorm(0, avg, se)

# Exercise 9. American Roulette Monte Carlo again

# Create a Monte Carlo simulation that generates 10,000 outcomes of S, the average
# outcome from 10,000 bets on green.

# Compute the average and standard deviation of the resulting list to confirm the
# results from previous exercises using the Central Limit Theorem.

# Use the replicate function to model 10,000 iterations of a series of 10,000 bets.

# Each iteration inside replicate should simulate 10,000 bets and determine the
# average outcome of those 10,000 bets. If you forget to take the mean, DataCamp will
# crash with a "Session Expired" error.

# Find the average of the 10,000 average outcomes. Print this vaue to the console.

# The variable `n` specifies the number of independent bets on green
n <- 10000

# The variable `B` specifies the number of times we want the simulation to run
B <- 10000

# Use the `set.seed` function to make sure your answer matches the expected result
# after random number generation
set.seed(1, sample.kind = "Rounding")

# Generate a vector `S` that contains the average outcomes of 10,000 bets modeled
# 10,000 times

S <- replicate(B, {
  X <- sample(c(17, -1), n, replace = TRUE, prob = c(p_green, p_not_green))
  mean(X)
})

# Compute the average of `S`

mean(S)

# Compute the standard deviation of `S`

sd(S)

# Exercise 10. American Roulette comparison

# In a previous exercise, you found the probability of winning more than $0 after
# betting on green 10,000 times using the Central Limit Theorem. Then, you used a
# Monte Carlo simulation to model the average result of betting on green 10,000 times
# over 10,000 simulated series of bets.

# What is the probability of winning more than $0 as estimated by your Monte Carlo
# simulation? The code to generate the vector `S` that contains the average outcomes
# of 10,000 bets modeled 10,000 times has already been run for you.

# Calculate the probability of winning more than $0 in the Monte Carlo simulation from
# the previous exercise using the mean function.

# Compute the proportion of outcomes in the vector `S` where you won more than $0

mean(S > 0)

# Exercise 11. American Roulette comparison analysis

# The Monte Carlo result and the CLT approximation are now much closer than when we
# calculated the probability of winning for 100 bets on green. What could account
# for this difference?

# We are now computing averages instead of sums. 

# 10,000 Monte Carlos simulations was not enough to provide a good estimate.

# The CLT works better when the sample size is larger. # Correct

# It is not closer. The difference is within rounding error.

# 3.3. ASSESSMENT: RANDOM VARIABLES, SAMPLING MODELS, AND THE CENTRAL LIMIT THEOREM

# Questions 1 and 2: SAT testing

# The SAT is a standardized college admissions test used in the United States.

# This is a 6-part question asking you to determine some probabilities of what happens
# when a student guessed for all of their answers on the SAT. Use the information below
# to inform your answers for the following questions.

# An old version of the SAT college entrance exam had a -0.25 point penalty for every
# incorrect answer and awarded 1 point for a correct answer. The quantitative test
# consisted of 44 multiple-choice questions each with 5 answer choices. Suppose a 
# student chooses answers by guessing for all questions on the test.

# Question 1a

# What is the probability of guessing correctly for one question?

1 / 5 # 1 correct answer among 5 choices # 0.2

# Question 1b

# What is the expected value of points for guessing on one question?

1 * 0.2 + -0.25 * 0.8

# Also... more formally:

p <- 1 / 5
a <- 1
b <- -0.25
mu <- a * p + b * (1 - p)
mu # 0

# Question 1c

# What is the expected score of guessing on all 44 questions?

n <- 44
n * mu # 0

# Question 1d

# What is the standard error of guessing on all 44 questions?

sigma <- sqrt(n) * abs(b - a) * sqrt(p * (1 - p))
sigma

# Question 1e

# Use the Central Limit Theorem to determine the probability that a guessing student
# scores 8 points or higher on the test.

1 - pnorm(8, mu, sigma) # 0.008

# Question 1f

# Set the seed to 21, then run a Monte Carlo simulation of 10,000 students guessing on
# the test.

# What is the probability that a guessing student scores 8 points or higher?

set.seed(21, sample.kind = "Rounding")

B <- 10000
S <- replicate(B, {
  X <- sample(c(1, -0.25), n, replace = TRUE, prob = c(p, 1 - p))
  sum(X)
})
mean(S >= 8) # 0.008

# The SAT was recently changed to reduce the number of multiple choice options from
# 5 to 4 and also to eliminate the penalty for guessing.

# In this two-part question, you'll explore how that affected the expected values for
# the test.

# Question 2a

# Suppose that the number of multiple choice options is 4 and that there is no penalty
# for guessing - that is, an incorrect question gives a score of 0.

# What is the expected value of the score when guessing on this new test?

new_p <- 1 / 4
new_a <- 1
new_b <- 0
new_mu <- n * (new_a * new_p + new_b * (1 - new_p))
new_mu

# Question 2b

# Consider a range of correct answer probabilities p <- seq(0.25, 0.95, 0.05) representing
# a range of student skills

# What is the lowest p such that the probability of scoring over 35 exceeds 80%?

p <- seq(0.25, 0.95, 0.05)
a <- 1
b <- 0
exp_val <- sapply(p, function(x){
  mu <- n * a * x + b * (1 - x)
  sigma <- sqrt(n) * abs(b - a) * sqrt(x * (1 - x))
  1 - pnorm(35, mu, sigma)
})
min(p[which(exp_val > 0.8)])

# Question 3. Betting on Roulette

# A casino offers a House Special bet on roulette, which is a bet on five pockets
# (00, 0, 1, 2, 3) out of 38 total pockets. The bet pays out 6 to 1. In other words,
# a losing bet yields -$1 and a successful bet yields $6. A gambler wants to know
# the chance of losing money if he places 500 bets on the roulette House Special.

# The following 7-part question asks you to do some calculations related to this scenario.

# Question 3a

# What is the expected value of the payout for one bet?

a <- 6
b <- -1
p <- 5 / 38

mu <- a * p + b * (1 - p)
mu

# Question 3b

# What is the standard error of the payout for one bet?

sigma <- abs(b - a) * sqrt(p * (1-p))
sigma

# Question 3c

# What is the expected value of the average payout over 500 bets? Remember there is
# a difference between the expected value of the average and expected value of the sum.

mu

# Question 3d

# What is the standard error of the average payout over 500 bets? Remember there is 
# a difference between the standard error of the average and the standard error of the sum.

n <- 500
sigma / sqrt(n)

# Question 3e

# What is the expected value of the sum of 500 bets?

n * mu

# Question 3f

# What is the standard error of the sum of 500 bets?

sqrt(n) * sigma

# Question 3g

# Use pnorm() with the expected value of the sum and standard error of the sum to
# calculate the probability of losing money over 500 bets, Pr(X <= 0)

pnorm(0, n * mu, sqrt(n) * sigma)

## SECTION 4. THE BIG SHORT

# 4.1. THE BIG SHORT: INTEREST RATES EXPLAINED

# Interest rates for loans are set using the probability of loan defaults to calculate
# a rate that minimizes the probability of losing money.

# We can define the outcome of loans as a random variable. We can also define the sum
# of outcomes of many loans as a random variable.

# The Central Limit Theorem can be applied to fit a normal distribution to the sum of
# profits over many loans. We can use properties of the normal distribution to calculate
# the interest rate needed to ensure a certain probability of losing money for a given
# probability of default.

# Interest rate sampling model

n <- 1000
loss_per_foreclosure <- -200000
p <- 0.02
defaults <- sample(c(0,1), n, prob = c(1-p, p), replace = TRUE)
sum(defaults * loss_per_foreclosure)

# Interest rate Monte Carlo simulation

B <- 10000
losses <- replicate(B, {
  defaults <- sample(c(0, 1), n, prob = c(1 - p, p), replace = TRUE)
  sum(defaults * loss_per_foreclosure)
})

# Plotting expected losses

library(tidyverse)
data.frame(losses_in_millions = losses / 10^6) %>%
  ggplot(aes(losses_in_millions)) +
  geom_histogram(binwidth = 0.6, col = "black")

# Expected value and standard error of the sum of 1,000 loans

n * (p * loss_per_foreclosure + (1 - p) * 0) # expected value ~ -4M
sqrt(n) * abs(loss_per_foreclosure) * sqrt(p * (1 - p)) # standard error ~ -800K

# Calculating interest rates for expected value of 0

# We can calculate the amount x to add to each loan so that the expected value is 0
# using the equation `lp + x(1 - p) = 0`. Note that this equation is the definition
# of expected value given a loss per foreclosure `l` with foreclosure possibility `p`
# and profit `x` if there is no foreclosure (probability 1 - p).

# We solve for x = - lp / (1 - p) and calculate x:

x <- -loss_per_foreclosure * p / (1 - p)
x # ~ $4,082

# On a $180,000 loan, this equals an interest rate of:

x / 180000 # ~ 0.0226 ~ 2.2%

# Equations: Calculating interest rate for 1% probability of losing money

# We want to calculate the value of x for which Pr(S < 0) = 0.01. The expected value
# E[S] of the sum of n = 1,000 loans given our definitions of x, l and p is:

# mu[S] = (lp + x(1 - p)) * n

# And the standard error of the sum of n loans, SE[S], is:

# sigma[S] = |x - l| * sqrt(np(1 - p))

# Because we know the definition of Z-score is Z = (x - mu) / sigma, we know that
# Pr(S < 0) = Pr(Z < - mu / sigma). Thus, Pr(S < 0) = 0.01 equals:

# Pr(Z < -{lp + x(1 - p)}n / (x - l) * sqrt(np(1 - p))) = 0.01

# z <- qnorm(0.01) gives us the value of z for which Pr(Z <= z) = 0.01, meaning:

# z = -{lp + x(1 - p)}n / (x - l) * sqrt(np(1 - p))

# Solving for x gives: x = - l * (np - z * sqrt(np(1 - p))) / (n(1-p) + z * sqrt(np(1 - p)))

# Calculating interest rate for 1% probability of losing money

l <- loss_per_foreclosure
z <- qnorm(0.01)
x <- -l * (n*p - z *sqrt(n*p*(1-p))) / (n*(1-p) + z * sqrt(n*p*(1-p)))
x / 180000 # interest rate ~ 0.0347 ~ 3.5%
loss_per_foreclosure * p + x * (1 - p) # expected value of the profit per loan ~ $2,124
n * (loss_per_foreclosure * p + x * (1 - p)) # expected value of the profit over n loans ~ $2.12M

# Monte Carlo simulation for 1% probability of losing money
B <- 10000
profit <- replicate(B, {
  draws <- sample(c(x, loss_per_foreclosure), n, prob = c(1 - p, p), replace = TRUE)
  sum(draws)
})
mean(profit) # expected value of the profit over n loans ~ $2.12M
mean(profit < 0) # probability of losing money ~ 0.0104 ~ 1%

# The Big Short

# The Central Limit Theorem states that the sum of independent draws of a random
# variable follows a normal distribution. However, when the draws are not independent, 
# this assumption does not hold.

# If an event changes the probability of default for all borrowers, then the probability
# of the bank losing money changes.

# Monte Carlo simulations can be used to model the effects of unknown changes in the
# probability of default.

# Expected value with higher default rate and interest rate
p <- .04
loss_per_foreclosure <- -200000
r <- 0.05
x <- r * 180000
loss_per_foreclosure * p + x * (1 - p) # ~ $640, still positive

# Equations: Probability of losing money

# We can define our desired probability of losing money z, as:

# Pr(S < 0) = Pr(Z < - E[S] / SE[S]) = Pr(Z < z)

# If `mu` is the expected value of the urn (one loan) and `sigma` is the standard
# deviation of the urn (one loan), then E[S] = n * mu and SE[S] = sqrt(n) * sigma.

# We define the probability of losing money z = 0.01. In the first equation, we can
# see that: z = - E[S] / SE[S]. It follows that:

# z = - n * mu / (sqrt(n) * sigma) = - sqrt(n) * mu / sigma

# To find the value of n for which z is less than or equal to our desired value, we
# take z <= - sqrt(n) * mu / sigma and solve for n:

# n >= z^2 * sigma^2 / mu^2

# Calculating number of loans for desired probability of losing money

# The number of loans required is:
z <- qnorm(0.01)
l <- loss_per_foreclosure
n <- ceiling((z^2*(x-l)^2*p*(1-p))/(l*p + x*(1-p))^2) 
n # number of loans required, 22,163
n * (loss_per_foreclosure * p + x * (1 - p)) # expected profit over n loans ~ $14.18M

# Monte Carlo simulation with known default probability

# This Monte Carlo simulation estimates the expected profit given a known probability of
# default p = 0.04.
B <- 10000
p <- 0.04
x <- 0.05 * 180000
profit <- replicate(B, {
  draws <- sample(c(x, loss_per_foreclosure), n, prob = c(1 - p, p), replace = TRUE)
  sum(draws)
})
mean(profit) # ~ $14.16M

# Monte Carlo simulation with unknown default probability

# This Monte Carlo simulation estimates the expected profit given an unknown probability
# of default 0.03 <= p <= 0.05, modeling the situation where an event changes the probability
# of default for all borrowers simultaneously.
p <- 0.04
x <- 0.05 * 180000
profit <- replicate(B, {
  new_p <- 0.04 + sample(seq(-0.01, 0.01, length = 100),1)
  draws <- sample(c(x, loss_per_foreclosure), n, prob = c(1 - new_p, new_p),
                  replace = TRUE)
  sum(draws)
})
mean(profit) # expected profit ~ $14.42M
mean(profit < 0) # probability of losing money ~ 0.3491 ~ 35%
mean(profit < -10000000) # probability of losing over $10M ~ 0.2366 ~ 24%

# DataCamp Assessment: The Big Short

# Exercise 1. Bank earnings

# Say you manage a bank that gives out 10,000 loans. The default rate is 0.03 and you lose
# $200,000 in each foreclosure.

# Create a random variable S that contains the earnings of your bank. Calculate the total
# amount of money lost in this scenario.

# Using the sample function, generate a vector called defaults that contains n samples from
# a vector of c(0, 1), where 0 indicates a payment and 1 indicates a default.

# Multiply the total number of defaults by the loss per foreclosure.

# Assign the number of loans to the variable `n`
n <- 10000

# Assign the loss per foreclosure to the variable `loss_per_foreclosure`
loss_per_foreclosure <- -200000

# Assign the probability of default to the variable `p_default`
p_default <- 0.03

# Use the set.seed function to make sure your answer matches the expected result after
# random sampling
set.seed(1, sample.kind = "Rounding")

# Generate a vector called `defaults` that contains the default outcomes of `n` loans

defaults <- sample(c(0, 1), n, replace = TRUE, prob = c(1 - p_default, p_default))

# Generate `S`, the total amount of money lost across all foreclosures. Print the value
# to the console.
S <- sum(defaults * loss_per_foreclosure)
S

# Exercise 2. Bank earnings Monte Carlo

# Run a Monte Carlo simulation with 10,000 outcomes for S, the sum of losses over 10,000
# loans. Make a histogram of the results.

# Within a replicate loop with 10,000 iterations, use sample to generate a list of 10,000
# loan outcomes: payment (0) or default (1). Use the outcome order c(0,1) and probability
# of default p_default.

# Still within the loop, use the function sum() to count the number of foreclosures
# multiplied by loss_per_foreclosure to return the sum of all losses across 10,000 loans.

# Plot the histogram of values using the function hist()

# Assign the number of loans to the variable `n`
n <- 10000

# Assign the loss per foreclosure to the variable `loss_per_foreclosure`
loss_per_foreclosure <- -200000

# Assign the probability of default to the variable `p_default`
p_default <- 0.03

# Use the set.seed function to make sure your answer matches the expected result after
# random sampling
set.seed(1, sample.kind = "Rounding")

# The variable `B` specifies the number of times we want the simulation to run
B <- 10000

# Generate a list of summed loses `S`. Replicate the code from the previous exercise
# over `B` iterations to generate a list of summed loses for `n` loans. Ignore any
# warnings for now.

S <- replicate(B, {
  defaults <- sample(c(0, 1), n, replace = TRUE, prob = c(1 - p_default, p_default))
  sum(defaults * loss_per_foreclosure)
})

# Plot a histogram of `S`. Ignore any warnings for now.

hist(S)

# Exercise 3. Bank earnings expected value

# What is the expected value of S, the sum of losses over 10,000 loans? For now,
# assume a bank makes no money if the loan is paid.

# Using the chances of default (p_default), calculate the expected losses over
# 10,000 loans.

# Assign the number of loans to the variable `n`
n <- 10000

# Assign the loss per foreclosure to the variable `loss_per_foreclosure`
loss_per_foreclosure <- -200000

# Assign the probability of default to the variable `p_default`
p_default <- 0.03

# Calculate the expected loss due to default out of 10,000 loans

n * p_default * loss_per_foreclosure

# Exercise 4. Bank earnings standard error

# What is the standard error of S?

# Compute the standard error of the random variable S you generated in the previous
# exercise, the summed outcomes of 10,000 loans.

# Assign the number of loans to the variable `n`
n <- 10000

# Assign the loss per foreclosure to the variable `loss_per_foreclosure`
loss_per_foreclosure <- -200000

# Assign the probability of default to the variable `p_default`
p_default <- 0.03

# Compute the standard error of the sum of 10,000 loans

sqrt(n) * abs(loss_per_foreclosure) * sqrt(p_default * (1 - p_default))

# Exercise 5. Bank earnings interest rate (1)

# So far, we've been assuming that we make no money when people pay their loans
# and we lose a lot of money when people default on their loans. Assume we give
# out loans for $180,000. How much money do we need to make when people pay their
# loans so that our net loss is $0?

# In other words, what interest rate do we need to charge in order to not lose
# money?

# If the amount of money lost or gained equals 0, the probability of default times
# the total loss per default equals the amount earned per probability of the loan
# being paid.

# Divide the total amount needed per loan by the loan amount to determine the
# interest rate.

# Assign the loss per foreclosure to the variable `loss_per_foreclosure`
loss_per_foreclosure <- -200000

# Assign the probability of default to the variable `p_default`
p_default <- 0.03

# Assign a variable `x` as the total amount necessary to have an expected
# outcome of $0

x <- - loss_per_foreclosure * p_default / (1 - p_default)

# Convert `x` to a rate, given that the loan amount is $180,000. Print this value to
# to the console.

x / 180000

# Exercise 6. Bank earnings interest rate (2)

# With the interest rate calculated in the last example, we still lose money 50% of
# the time. What should the interest rate be so that the chance of losing money is 1
# in 20?

# In math notation, what should the interest rate be so that Pr(S < 0) = 0.05?

# Remember that we can add a constant to both sides of the equation to get:

# Pr(S - E[S] / SE[S] < - E[S] / SE[S]), which is

# Pr (Z < - [lP + x(1 - P)]n / (x - l) * sqrt(np(1-p))) = 0.05

# Let z = qnorm(0.05) give us the value of z for which: Pr(Z <= z) = 0.05

# Use the qnorm() function to compute a continuous variable at given quantile of
# the distribution to solve for z.

# In this equation, l, P, and n are known values. Once you've solved for z, solve
# for x.

# Divide x by the loan amount to calculate the rate.

# Assign the number of loans to the variable `n`
n <- 10000

# Assign the loss per foreclosure to the variable `loss_per_foreclosure`
loss_per_foreclosure <- -200000

# Assign the probability of default to the variable `p_default`
p_default <- 0.03

# Generate a variable `z` using the qnorm() function

z <- qnorm(0.05)

# Generate a variable `x` using `z`, `p_default`, `loss_per_foreclosure`, and `n`

x <- -loss_per_foreclosure *
  (n*p_default - z *sqrt(n*p_default*(1-p_default))) / 
  (n*(1-p_default) + z * sqrt(n*p_default*(1-p_default))) 

# Convert `x` to an interest rate, given that the loan amount is $180,000. Print this
# value to the console.

x / 180000

# Exercise 7. Bank earnings - minimize money loss

# The banks wants to minimize the probability of losing money. Which of the following
# achieves their goal without making interest rates go up?

# A smaller pool of loans

# A larger probability of default

# A reduced default rate # Correct

# A larger cost per loan default

# 4.2. ASSESSMENT: THE BIG SHORT

# Introduction

# These exercises review and assess the following concepts:

# Expected value and standard error of a single draw of a random variable
# Expected value and standard error of the sum of draws of a random variable
# Monte Carlo simulation of the sum of draws of a random variable
# The Central Limit Theorem approximation of the sum of draws of a random variable
# Using z-scores to calculate values related to the normal distribution and normal
# random variables
# Calculating interest/premium rates to minimize chance of losing money
# Determining a number of loans/policies required for profit
# Simulating the effects of a change in event probability

# Setup and libraries

# Run the code below to set up your environment and load the libraries you will need
# for the following exercises
library(tidyverse)
library(dslabs)

# Background

# In the motivating example `The Big Short`, we discussed how discrete and continuous
# probability concepts relate to bank loans and interest rates. Similar business
# problems are faced by the insurance industry.

# Just as banks must decide how much to charge as interest on loans based on estimates
# of loan defaults, insurance companies must decide how much to charge as premiums for
# policies given estimates of the probability that an individual will collect on that
# policy.

# We will use data from 2015 US Period Life Tables. Here is the code you will need to
# load and examine the data from dslabs.
data(death_prob)
head(death_prob)

# There are six multi-part questions for you to answer that follow.

# Questions 1 and 2: Insurance Rates, part 1

# An insurance company offers a one-year term life insurance policy that pays $150k in
# the event of death within one year. The premium (annual cost) for this policy for a 
# 50 year old female is $1,150. Suppose that in the event of a claim, the company forfeits
# the premium and loses a total of $150,000, and if there is no claim the company gains
# the premium amount of $1,150. The company plans to sell 1,000 policies to this
# demographic.

# Question 1a

# The death_prob data frame from the dslabs package contains information about the
# estimated probability of death within 1 year (prob) for different ages and sexes.

# Use death_prob to determine the death probability of a 50 year old female, p.

p <- death_prob %>% filter(age == 50 & sex == "Female") %>% pull(prob)
p

# Question 1b

# The loss in the event of the policy holder's death is -$150,000 and the gain if the
# policy holder remains alive is the premium $1,150.

# What is the expected value of the company's net profit on one policy for a 50 year
# old female?

a <- -150000
b <- 1150
mu <- a * p + b * (1 -p)
mu

# Question 1c

# Calculate the standard error of the profit on one policy for a 50 year old female.

sigma <- abs(b - a) * sqrt(p * (1 - p))
sigma

# Question 1d

# What is the expected value of the company's profit over all 1,000 policies for
# 50 year old females?

n <- 1000
profit <- n * mu
profit

# Question 1e

# What is the standard error of the sum of the expected value over all 1,000 policies
# for 50 year old females?

se <- sqrt(n) * sigma
se

# Question 1f

# Use the Central Limit Theorem to calculate the probability that the insurance
# company loses money on this set of 1,000 policies.

pnorm(0, profit, se)

# 50 year old males have a different probability of death than 50 year old females. We
# will calculate a profitable premium for 50 year old males in the following four-part
# question.

# Question 2a

# Use death_prob to determine the probability od death within one year for a 50 year old
# male.

p <- death_prob %>% filter(sex == "Male" & age == 50) %>% pull(prob)
p

# Question 2b

# Suppose the company wants its expected profits from 1,000 50 year old males with
# $150,000 life insurance policies to be $700,000. Use the formula for expected
# value of the sum of draws with the following values and solve for the premium `b`:

# E[S] = mu[S] = 700000 ; n = 1000 ; p = death probability of age 50 male ;
# a = 150000 ; b = premium

# E[S] = n * (a * p + b * (1 - p)) = 700000
# a * p + b * (1 - p) = 700000 / n
# b * (1 - p) = (700000 / n) - a * p
# b = (700000 / n) - a * p / (1 - p)

b <- ((700000 / n) - a * p) / (1 - p)
b

# Question 2c

# Using the new 50 year old male premium rate, calculate the standard error of the sum
# of 1,000 premiums

se <- sqrt(n) * abs(b - a) * sqrt(p * (1 - p))
se

# Question 2d

# What is the probability of losing money on a series of 1,000 policies to 50 year old
# males? Use the central limit theorem.

pnorm(0, n * (a * p + b * (1 - p)), se)

# Questions 3 and 4: Insurance rates, part 2

# Life insurance rates are calculated using mortality statistics from the recent past.
# They are priced such that companies are almost assured to profit as long as the
# probability of death remains similar. If an event occurs that changes the probability
# of death in a given age group, the company risks significant losses.

# In this 6-part question, we'll look at a scenario in which a lethal pandemic disease
# increases the probability of death within 1 year for a 50 year old to .015. Unable to
# predict the outbreak, the company has sold 1,000 $150,000 life insurance policies
# for $1,150.

# Question 3a

# What is the expected value of the company's profits over 1,000 policies?

b <- 1150
p <- 0.015

profit <- n * (a * p + b * (1 -p))
profit

# Question 3b

# What is the standard error of the expected value of the company's profits
# over 1,000 policies?

se <- sqrt(n) * abs(b - a) * sqrt(p * (1 - p))
se

# Question 3c

# What is the probability of the company losing money?

pnorm(0, profit, se)

# Question 3d

# Suppose the company can afford to sustain one-time loses of $1M, but larger
# losses will force it to go out of business.

# What is the probability of losing more than $1 million?

pnorm(-1000000, profit, se)

# Question 3e

# Investigate death probabilities p <- seq(.01, .03, 0.001)

# What is the lowest death probability for which the chance of losing money
# exceeds 90%?

p <- seq(0.01, 0.03, 0.001)
exp_val <- sapply(p, function(x){
  mu <- n * (a * x + b * (1 - x))
  sigma <- sqrt(n) * abs(b - a) * sqrt(x * (1 - x))
  pnorm(0, mu, sigma)
})
which(p[exp_val > 0.9])
min(p[exp_val > 0.9])

# Question 3f

# Investigate death probabilities p <- seq(.01, .03, .0025)

# What is the lowest death probability for which the chance of losing over $1M
# exceeds 90%?

p <- seq(.01, .03, .0025)
p_lose <- sapply(p, function(x){
  mu <- n * (a * x + b * (1 -x))
  sigma <- sqrt(n) * abs(b - a) * sqrt(x * (1 - x))
  pnorm(-1 * 10^6, mu, sigma)
})
data.frame(p, p_lose) %>%
  filter(p_lose > 0.9) %>%
  pull(p) %>% min()

# Question 4, which has two parts, continues the scenario from Question 3.

# Question 4a

# Define a sampling model for simulating the total profit over 1,000 loans with
# probability of claim p_loss = .015, loss of -$150,000 on a claim, and profit of
# $1,150 when there is no claim. Set the seed to 25, then run the model once.

# What is the reported profit (or loss) in millions (that is, divided by 10^6)?

set.seed(25, sample.kind = "Rounding")

p_loss <- .015
profit <- sample(c(a, b), n, replace = TRUE, prob = c(p_loss, 1 - p_loss))
sum(profit) / 10^6

# Question 4b

# Set the seed to 27, then run a Monte Carlo simulation of your sampling model with
# 10,000 replicates to simulate the range of profits/losses over 1,000 loans.

# What is the observed probability of losing $1 million or more?

set.seed(27, sample.kind = "Rounding")

B <- 10000
S <- replicate(B, {
  outcomes <- sample(c(a, b), n, replace = TRUE, prob = c(p_loss, 1 - p_loss))
  sum(outcomes)
})
mean(S < -1 * 10^6)

# Questions 5 and 6: Insurance rates, part 3

# Question 5, which has 4 parts, continues the pandemic scenario from Questions 3
# and 4.

# Suppose that there is a massive demand for life insurance due to the pandemic, and
# the company wants to find a premium cost for which the probability of losing money
# is under 5%, assuming the death rate stays table at p = 0.015.

# Question 5a

# Calculate the premium required for a 5% chance of losing money given n = 1000 loans,
# probability of death p = 0.015, and loss per claim l = -150000. Save this premium as
# x for use in further questions.

# Pr(S < 0) = 0.05 ; Pr(S - E[S] / SE[S] < - E[S] / SE[S])

n <- 1000
p <- 0.015
l <- -150000
z <- qnorm(0.05)
x <- -l * (n*p - z *sqrt(n*p*(1-p))) / (n*(1-p) + z * sqrt(n*p*(1-p)))

# Question 5b

# What is the epexpected profit per policy at this rate?

mu <- l * p + x * (1 - p)
mu

# Question 5c

# What is the expected profit over 1,000 policies

profit <- n * mu

# Question 5d

# Run a Monte Carlo simulation with B = 10,000 to determine the probability of
# losing money on 1,000 policies given the new premium x, loss on a claim of
# $150,000 and probability of claim p = 0.015. Set the seed to 28 before running
# your simulation.

# What is the probability of losing money here?

set.seed(28, sample.kind = "Rounding")

B <- 10000
outcomes <- replicate(B, {
  X <- sample(c(l, x), n, replace = TRUE, prob = c(p, 1 - p))
  sum(X)
})
mean(outcomes < 0)

# The company cannot predict whether the pandemic death rate will stay stable. Set
# the seed to 29, then write a Monte Carlo simulation that for each of B = 10000
# iterations:

# 1. Randomly changes p by adding a value between -0.01 and 0.01 with
# sample(seq(-0.01, 0.01, length = 100), 1)
# 2. Uses the new random p to generate a sample of n = 1,000 policies with premium
# x and loss per claim l = -150,000.
# 3. Returns the profit over n policies (sum of random variable)

set.seed(29, sample.kind = "Rounding")

# The outcome should be a vector of B total profits. Use the results of the Monte
# Carlo situation to answer the following three questions.

# Question 6a

# What is the expected value over 1,000 policies

profits <- replicate(B, {
  new_p <- p + sample(seq(-0.01, 0.01, length = 100), 1)
  draws <- sample(c(l, x), n, replace = TRUE, prob = c(new_p, 1 - new_p))
  sum(draws)
})

mean(profits)

# Question 6b

# What is the probability of losing money?

mean(profits < 0)

# Question 6c

# What is the probability of losing more than $1 million?

mean(profits < -1 * 10^6)
