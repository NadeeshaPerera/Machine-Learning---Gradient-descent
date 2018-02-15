######################################################################################################
# Name : Nadeesha Perera
# Date : 02/03/2018
# Topic : BUAN 6341 - Applied Machine Learning - Assignment 1
# Purpose : Implement Linear and Logistic Regression
#####################################################################################################

# Clear the environment
rm(list = ls(all = TRUE))

# Load the required libraries
library(data.table)
library(ggplot2)

#######################################################################

# Load the data
data <- read.csv("OnlineNewsPopularity.csv", header = TRUE)

# Selecting only predictive and target variables
data1 <- data[,3:61]

# See structure of data
str(data1)

# Define train sample size
train_size <- floor(0.7*nrow(data1))

# set seed to make sample reproducible
set.seed(5926)
train_ind <- sample(seq_len(nrow(data1)), size = train_size)

# Divide data to train and test
Train <- data1[train_ind,]
test <- data1[-train_ind,]


# Define validation sample size
valid_size <- floor(0.3*nrow(Train))

# set seed to make sample reproducible
set.seed(9027)
valid_ind <- sample(seq_len(nrow(Train)), size = valid_size)

# Divide train data to train and validation
validation <- Train[valid_ind,]
train <- Train[-valid_ind,]



summary(train)

# Exploratory analysis

# A weird data point in n_non_stop_words
plot(train$n_non_stop_words, train$shares, col = "blue", main = "n_non_stop_words against shares", xlab = "n_non_stop_words", ylab = "shares")
# Only one data point much further from the rest
# Remove this weird data point
train <- train[which(train$n_non_stop_words != 1042), ]
summary(train)

# A weird data point in n_tokens_content
plot(train$n_tokens_content, train$shares, col = "blue", main = "n_tokens_content against shares", xlab = "n_tokens_content", ylab = "shares")
# There are few data points in this range. The distribution is right skewed. So let be

# Look at kw_max_min
plot(train$kw_max_min, train$shares, col = "blue", main = "kw_max_min against shares", xlab = "kw_max_min", ylab = "shares")
# Several data points out of range. Since curve is right skewed, just let be

# Look at self_reference_max_shares
plot(train$self_reference_max_shares, train$shares, col = "blue", main = "self_reference_max_shares against shares", xlab = "self_reference_max_shares", ylab = "shares")
# Several data points out of range. Since curve is right skewed, just let be


# Remove the rows with NA values from data set
train <- na.omit(train)
summary(train)


# Understand the variables

# Distribution across independent variables
# hist(train$n_tokens_title, col = "blue") #normal
# hist(train$n_tokens_content, col = "blue") #skewed
# hist(train$n_unique_tokens, col = "blue", breaks = 50) #single column
# hist(train$n_non_stop_words, col = "blue", breaks = 50) #single column
# hist(train$n_non_stop_unique_tokens, col = "blue", breaks = 50) #single column
# hist(train$num_hrefs, col = "blue", breaks = 50) #skewed
# hist(train$num_self_hrefs, col = "blue", breaks = 50) #skewed
# hist(train$num_imgs, col = "blue", breaks = 50) #single column & flat few
# hist(train$num_videos, col = "blue", breaks = 50) #single column & flat few
# hist(train$average_token_length, col = "blue", breaks = 50) #normal with few outlier
# hist(train$num_keywords, col = "blue") #left skewed
# hist(train$data_channel_is_lifestyle, col = "blue") #dummy 
# hist(train$data_channel_is_entertainment, col = "blue") #dummy
# hist(train$data_channel_is_bus, col = "blue") #dummy
# hist(train$data_channel_is_socmed, col = "blue") #dummy
# hist(train$data_channel_is_tech, col = "blue") #dummy
# hist(train$data_channel_is_world, col = "blue") #dummy
# hist(train$kw_min_min, col = "blue", breaks = 50) #right skewed
# hist(train$kw_max_min, col = "blue", breaks = 50) #single column & flat few
# hist(train$kw_avg_min, col = "blue", breaks = 50) #single column & flat few both sides
# hist(train$kw_min_max, col = "blue", breaks = 50) #right skewed
# hist(train$kw_max_max, col = "blue", breaks = 50) #left skewed
# hist(train$kw_avg_max, col = "blue", breaks = 50) #bimodal
# hist(train$kw_min_avg, col = "blue", breaks = 50) #single column and set of flats
# hist(train$kw_max_avg, col = "blue", breaks = 50) #right skewed
# hist(train$kw_avg_avg, col = "blue", breaks = 50) #normal
# hist(train$self_reference_min_shares, col = "blue", breaks = 50) #right skewed
# hist(train$self_reference_max_shares, col = "blue", breaks = 50) #right skewed
# hist(train$self_reference_avg_sharess, col = "blue", breaks = 50) #right skewed
# hist(train$is_weekend, col = "blue", breaks = 50) #right skewed
# hist(train$global_subjectivity, col = "blue", breaks = 50) #normal
# hist(train$global_sentiment_polarity, col = "blue", breaks = 50) #normal
# hist(train$global_rate_negative_words, col = "blue", breaks = 50) #right skewed
# hist(train$global_rate_positive_words, col = "blue", breaks = 50) #normal
# hist(train$rate_positive_words, col = "blue", breaks = 50) #left skewed
# hist(train$rate_negative_words, col = "blue", breaks = 50) #right skewed
# hist(train$avg_positive_polarity, col = "blue", breaks = 50) #normal
# hist(train$min_positive_polarity, col = "blue", breaks = 50) #right skewed
# hist(train$max_positive_polarity, col = "blue", breaks = 50) #left skewed
# hist(train$avg_negative_polarity, col = "blue", breaks = 50) #left skewed
# hist(train$min_negative_polarity, col = "blue", breaks = 50) #left skewed
# hist(train$max_negative_polarity, col = "blue", breaks = 50) #distributed
# hist(train$title_subjectivity, col = "blue", breaks = 50) #column and flats
# hist(train$title_sentiment_polarity, col = "blue", breaks = 50) #column and flats on both sides
# hist(train$abs_title_subjectivity, col = "blue", breaks = 50) #column and flats
# hist(train$abs_title_sentiment_polarity, col = "blue", breaks = 50) #column and flats




####################################################################################
# preparing train data to implement cost function and gradient descent

X_train <- train[,1:58]
y_train <- train[,59]

# X Intercept column
intercept <- rep(1, nrow(X_train))


############################################
# Function to normalize (bring to scale between 0,1)
##############################################

normal <- function(p){
  (p - min(p))/(max(p) - min(p))
}

###################################################################

# Normalize all X variables
X_norm <- apply(X_train, 2, normal)

X_full <- cbind(intercept, X_norm)

X_mat <- as.matrix(X_full)
y_mat <- as.matrix(y_train)

###################################################################################
# preparing validation data to implement cost function and gradient descent

X_valid <- validation[,1:58]
y_valid <- validation[,59]

# X Intercept column
intercept_valid <- rep(1, nrow(X_valid))


# Normalize all X variables in test set
X_norm_valid <- apply(X_valid, 2, normal)

X_full_valid <- cbind(intercept_valid, X_norm_valid)

X_mat_valid <- as.matrix(X_full_valid)
y_mat_valid <- as.matrix(y_valid)



###################################################################################
# preparing test data to implement cost function and gradient descent

X_test <- test[,1:58]
y_test <- test[,59]

# X Intercept column
intercept_test <- rep(1, nrow(X_test))


# Normalize all X variables in test set
X_norm_test <- apply(X_test, 2, normal)

X_full_test <- cbind(intercept_test, X_norm_test)

X_mat_test <- as.matrix(X_full_test)
y_mat_test <- as.matrix(y_test)



#############################################################################################################
#
# Linear Regression
#
#############################################################################################################


#########################################################################################################
# Functions 
############################################################################################################

# Cost function

cost <- function(X_mat, y_mat, theta){
  m <- nrow(y_mat)
  h <- X_mat %*% theta
  J = (1/(2*m))*sum((h - y_mat)^2)
  return(J)
}




# Gradient decent function

gradient_desc <- function(X_mat, y_mat, theta, alp, num_iter){
  m <- nrow(y_mat)
  J_history <- rep(0, num_iter)
  
  for(i in 1:num_iter){
    theta <- theta - (alp/m)*t(X_mat)%*%((X_mat %*% theta) - y_mat)
    J_history[i] <- cost(X_mat, y_mat, theta)
    
  }
  return(list(theta , J_history))
  
}


##################################################################################################################

# Define the variables
alp <- 0.2
# alp_range <- seq(from = 0.01, to = 0.12, by = 0.01)
# mse_train_range <- data.frame(val = 0)

# for(k in 1:12){

# alp <- k/100
num_iter <- 3000

theta_1 <- matrix(rep(0, ncol(X_full)))

# Run gradient decent and cost functions to return the cost history every iteration and theta at minimum cost
values <- gradient_desc(X_mat, y_mat, theta_1, alp,  num_iter)

# Assign the cost history to variable J_histoty
J_history <- values[[2]]
J_history

plot(1:num_iter, J_history, col = "blue", main = "Cost over iterations - alpha = 0.2", xlab = "Iterations", ylab = "Cost")

# Assign the theta values at lowest cost to variable theta_final
theta_final <- values[[1]]



####################################################################

# predict y in train data with gradient descent theta

y_gd_hat <- rep(0, nrow(train))

y_gd_hat <- X_mat %*% theta_final

# error_train <- (y_mat - y_gd_hat)
mse_train <- (sum((y_mat - y_gd_hat)^2))/nrow(train)
mape_train <- (sum((abs(y_mat - y_gd_hat))/y_mat))/nrow(train)

# mse_train_range <- rbind(mse_train_range, mse_train)

# }

# plot(1:24, mse_train_range$val, col = "red", main = "Error of the train data", xlab = "data points", ylab = "Error")

# mse_train_range

###################################################################
# Testing on validation data
#################################################################


# predict y in validation data

y_valid_hat <- rep(0, nrow(validation))

y_valid_hat <- X_mat_valid %*% theta_final

mse_valid <- (sum((y_mat_valid - y_valid_hat)^2))/nrow(validation)
mape_valid <- (sum((abs(y_mat_valid - y_valid_hat))/y_mat_valid))/nrow(validation)



###################################################################
# Testing on test data
#################################################################


# predict y in test data

y_test_hat <- rep(0, nrow(test))

y_test_hat <- X_mat_test %*% theta_final

mse_test <- (sum((y_mat_test - y_test_hat)^2))/nrow(test)
mape_test <- (sum((abs(y_mat_test - y_test_hat))/y_mat_test))/nrow(test)





########################################################################################################
#
# Logistic regression
#
##############################################################################################################

# Look at train data in shares to determine the boundary to seperate large and small shares
summary(y_train)
# The median is in 1,400. Will be a good place to split data
y_select <- y_train[which(y_train < 10000)]

histinfo <- hist(y_select, col = "blue", breaks = 100, xlab = "Number of shares", main = "Histogram of number of shares")
histinfo

# splitting data as large and small based on median

# Train data
y_log_train <- as.matrix(cut(y_mat, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_train <- mapply(y_log_train, FUN = as.numeric)

y_log_train <- matrix(data = y_log_train, ncol = 1, nrow = nrow(y_mat))

summary(y_log_train)


# Validation data

y_log_valid <- as.matrix(cut(y_mat_valid, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_valid <- mapply(y_log_valid, FUN = as.numeric)

y_log_valid <- matrix(data = y_log_valid, ncol = 1, nrow = nrow(y_mat_valid))



# Test data

y_log_test <- as.matrix(cut(y_mat_test, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_test <- mapply(y_log_test, FUN = as.numeric)

y_log_test <- matrix(data = y_log_test, ncol = 1, nrow = nrow(y_mat_test))



#########################################################################################################
# Functions 
############################################################################################################

# Cost function

sigmoid <- function(z){
  1/(1+exp(-z))
}


cost_log <- function(X_mat, y_mat, theta){
  m <- nrow(y_mat)
  h <- X_mat %*% theta
  J = (-1/m)*((t(y_mat) %*% log(sigmoid(h))) + (t(1 - y_mat) %*% log(1 - sigmoid(h))))
  return(J)
}




# Gradient decent function

gradient_desc_log <- function(X_mat, y_mat, theta, alp, num_iter){
  m <- nrow(y_mat)
  J_history <- rep(0, num_iter)
  
  for(i in 1:num_iter){
    h <- X_mat %*% theta
    theta <- theta - (alp/m)*t(X_mat)%*%(sigmoid(h) - y_mat)
    J_history[i] <- cost(X_mat, y_mat, theta)
    
  }
  return(list(theta , J_history))
  
}


##################################################################################################################

# Define the variables
alp_log <- 0.08
num_iter_log <- 3000

theta_log <- matrix(rep(0, ncol(X_full)))

# Run gradient decent and cost functions to return the cost history every iteration and theta at minimum cost
values_log <- gradient_desc(X_mat, y_log_train, theta_log, alp_log,  num_iter_log)

# Assign the cost history to variable J_histoty_log
J_history_log <- values_log[[2]]
J_history_log

plot(1:num_iter_log, J_history_log, col = "blue", main = "Cost over iterations - alpha = 0.08", xlab = "Iterations", ylab = "Cost")

# Assign the theta values at lowest cost to variable theta_final_log
theta_final_log <- values_log[[1]]



####################################################################

# predict y in train data with gradient descent theta

y_hat_log <- rep(0, nrow(train))

y_hat_log <- ifelse(X_mat %*% theta_final_log >= 0.5, 1, 0)

y_total_train <- as.data.frame(cbind(y_log_train, y_hat_log))
colnames(y_total_train) <- c("actual_y", "predicted_y")

table(y_total_train)

# model accuracy

# True negative
specificity <- 6504/(6504+3343)
false_positive <- 1 - specificity

# True positive
sensitivity <- 5988/(5988+3589)
false_negative <- 1 - sensitivity



###################################################################
# Testing on validation data
#################################################################


# predict y in validation data

y_valid_hat_log <- rep(0, nrow(validation))

y_valid_hat_log <- ifelse(X_mat_valid %*% theta_final_log >= 0.5, 1, 0)

y_total_valid <- as.data.frame(cbind(y_log_valid, y_valid_hat_log))
colnames(y_total_valid) <- c("actual_y", "predicted_y")

table(y_total_valid)

# model accuracy

# True negative
specificity_valid <- 2791/(2791+1469)
false_positive_valid <- 1 - specificity_valid

# True positive
sensitivity_valid <- 2520/(2520+1545)
false_negative_valid <- 1 - sensitivity_valid




###################################################################
# Testing on test data
#################################################################


# predict y in test data

y_test_hat_log <- rep(0, nrow(test))

y_test_hat_log <- ifelse(X_mat_test %*% theta_final_log >= 0.5, 1, 0)


y_total_test <- as.data.frame(cbind(y_log_test, y_test_hat_log))
colnames(y_total_test) <- c("actual_y", "predicted_y")

table(y_total_test)

# model accuracy

# True negative
specificity_test <- 3983/(3983+1992)
false_positive_test <- 1 - specificity_test

# True positive
sensitivity_test <- 3620/(3620+2299)
false_negative_test <- 1 - sensitivity_test


################################################################################################################
#
# Question 2
#
##############################################################################################################

# Select 10 features at random

# set seed to make sample reproducible
set.seed(9563)
train_ind_r10 <- sample(seq_len(NCOL(data1[,1:58])), size = 10)

# Divide data to test
test_r10 <- as.data.frame(cbind(test[,train_ind_r10], shares = test[,59]))

# Divide train data to train and validation
validation_r10 <- as.data.frame(cbind(validation[,train_ind_r10], shares = validation[,59]))
train_r10 <- as.data.frame(cbind(train[,train_ind_r10], shares = train[,59]))


####################################################################################
# preparing train data to implement cost function and gradient descent

X_train_r10 <- train_r10[,1:10]
y_train_r10 <- train_r10[,11]


# Normalize all X variables
X_norm_r10 <- apply(X_train_r10, 2, normal)

X_full_r10 <- cbind(intercept, X_norm_r10)

X_mat_r10 <- as.matrix(X_full_r10)
y_mat_r10 <- as.matrix(y_train_r10)

###################################################################################
# preparing validation data to implement cost function and gradient descent

X_valid_r10 <- validation_r10[,1:10]
y_valid_r10 <- validation_r10[,11]


# Normalize all X variables in test set
X_norm_valid_r10 <- apply(X_valid_r10, 2, normal)

X_full_valid_r10 <- cbind(intercept_valid, X_norm_valid_r10)

X_mat_valid_r10 <- as.matrix(X_full_valid_r10)
y_mat_valid_r10 <- as.matrix(y_valid_r10)




###################################################################################
# preparing test data to implement cost function and gradient descent

X_test_r10 <- test_r10[,1:10]
y_test_r10 <- test_r10[,11]


# Normalize all X variables in test set
X_norm_test_r10 <- apply(X_test_r10, 2, normal)

X_full_test_r10 <- cbind(intercept_test, X_norm_test_r10)

X_mat_test_r10 <- as.matrix(X_full_test_r10)
y_mat_test_r10 <- as.matrix(y_test_r10)



##################################################################################################################
# Retrain the model

# Define the variables
alp_r10 <- 0.35
num_iter_r10 <- 3000

theta_r10 <- matrix(rep(0, ncol(X_full_r10)))

# Run gradient decent and cost functions to return the cost history every iteration and theta at minimum cost
values_r10 <- gradient_desc(X_mat_r10, y_mat_r10, theta_r10, alp_r10,  num_iter_r10)

# Assign the cost history to variable J_histoty
J_history_r10 <- values_r10[[2]]
J_history_r10

plot(1:num_iter_r10, J_history_r10, col = "blue", main = "Cost over iterations - alpha = 0.35", xlab = "Iterations", ylab = "Cost")

# Assign the theta values at lowest cost to variable theta_final
theta_final_r10 <- values_r10[[1]]



####################################################################

# predict y in train data with gradient descent theta

y_hat_r10 <- rep(0, nrow(train_r10))

y_hat_r10 <- X_mat_r10 %*% theta_final_r10

mse_train_r10 <- (sum((y_mat_r10 - y_hat_r10)^2))/nrow(train_r10)
mape_train_r10 <- (sum((abs(y_mat_r10 - y_hat_r10))/y_mat_r10))/nrow(train_r10)

###################################################################
# Testing on validation data
#################################################################


# predict y in validation data

y_valid_hat_r10 <- rep(0, nrow(validation_r10))

y_valid_hat_r10 <- X_mat_valid_r10 %*% theta_final_r10

mse_valid_r10 <- (sum((y_mat_valid_r10 - y_valid_hat_r10)^2))/nrow(validation_r10)
mape_valid_r10 <- (sum((abs(y_mat_valid_r10 - y_valid_hat_r10))/y_mat_valid_r10))/nrow(validation_r10)




###################################################################
# Testing on test data
#################################################################


# predict y in test data

y_test_hat_r10 <- rep(0, nrow(test_r10))

y_test_hat_r10 <- X_mat_test_r10 %*% theta_final_r10

mse_test_r10 <- (sum((y_mat_test_r10 - y_test_hat_r10)^2))/nrow(test_r10)
mape_test_r10 <- (sum((abs(y_mat_test_r10 - y_test_hat_r10))/y_mat_test_r10))/nrow(test_r10)

########################################################################################################
#
# Logistic regression
#
##############################################################################################################


# Train data
y_log_train_r10 <- as.matrix(cut(y_mat_r10, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_train_r10 <- mapply(y_log_train_r10, FUN = as.numeric)

y_log_train_r10 <- matrix(data = y_log_train_r10, ncol = 1, nrow = nrow(y_mat_r10))

summary(y_log_train_r10)

# Validation data

y_log_valid_r10 <- as.matrix(cut(y_mat_valid_r10, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_valid_r10 <- mapply(y_log_valid_r10, FUN = as.numeric)

y_log_valid_r10 <- matrix(data = y_log_valid_r10, ncol = 1, nrow = nrow(y_mat_valid_r10))



# Test data

y_log_test_r10 <- as.matrix(cut(y_mat_test_r10, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_test_r10 <- mapply(y_log_test_r10, FUN = as.numeric)

y_log_test_r10 <- matrix(data = y_log_test_r10, ncol = 1, nrow = nrow(y_mat_test_r10))


##################################################################################################################

# Define the variables
alp_log_r10 <- 0.2
num_iter_log_r10 <- 3000

theta_log_r10 <- matrix(rep(0, ncol(X_full_r10)))

# Run gradient decent and cost functions to return the cost history every iteration and theta at minimum cost
values_log_r10 <- gradient_desc(X_mat_r10, y_log_train_r10, theta_log_r10, alp_log_r10,  num_iter_log_r10)

# Assign the cost history to variable J_histoty_log
J_history_log_r10 <- values_log_r10[[2]]
J_history_log_r10

plot(1:num_iter_log_r10, J_history_log_r10, col = "blue", main = "Cost over iterations - Alpha = 0.2", xlab = "Iterations", ylab = "Cost")

# Assign the theta values at lowest cost to variable theta_final_log
theta_final_log_r10 <- values_log_r10[[1]]



####################################################################

# predict y in train data with gradient descent theta

y_hat_log_r10 <- rep(0, nrow(train_r10))

y_hat_log_r10 <- ifelse(X_mat_r10 %*% theta_final_log_r10 >= 0.5, 1, 0)

y_total_train_r10 <- as.data.frame(cbind(y_log_train_r10, y_hat_log_r10))
colnames(y_total_train_r10) <- c("actual_y", "predicted_y")

table(y_total_train_r10)

# model accuracy

# True negative
specificity_r10 <- 5194/(5194+4653)
false_positive_r10 <- 1 - specificity_r10

# True positive
sensitivity_r10 <- 5694/(5694+3883)
false_negative_r10 <- 1 - sensitivity_r10



###################################################################
# Testing on validation data
#################################################################


# predict y in validation data

y_valid_hat_log_r10 <- rep(0, nrow(validation_r10))

y_valid_hat_log_r10 <- ifelse(X_mat_valid_r10 %*% theta_final_log_r10 >= 0.5, 1, 0)

y_total_valid_r10 <- as.data.frame(cbind(y_log_valid_r10, y_valid_hat_log_r10))
colnames(y_total_valid_r10) <- c("actual_y", "predicted_y")

table(y_total_valid_r10)

# model accuracy

# True negative
specificity_valid_r10 <- 2181/(2181+2079)
false_positive_valid_r10 <- 1 - specificity_valid_r10

# True positive
sensitivity_valid_r10 <- 2550/(2550+1515)
false_negative_valid_r10 <- 1 - sensitivity_valid_r10






###################################################################
# Testing on test data
#################################################################


# predict y in test data

y_test_hat_log_r10 <- rep(0, nrow(test_r10))

y_test_hat_log_r10 <- ifelse(X_mat_test_r10 %*% theta_final_log_r10 >= 0.5, 1, 0)


y_total_test_r10 <- as.data.frame(cbind(y_log_test_r10, y_test_hat_log_r10))
colnames(y_total_test_r10) <- c("actual_y", "predicted_y")

table(y_total_test_r10)

# model accuracy

# True negative
specificity_test_r10 <- 3155/(3155+2820)
false_positive_test_r10 <- 1 - specificity_test_r10

# True positive
sensitivity_test_r10 <- 3574/(3574+2345)
false_negative_test_r10 <- 1 - sensitivity_test_r10






################################################################################################################
#
# Question 3 - Pick 10 features - not random
#
##############################################################################################################


# Look at correlation matrix
cor_mat <- cor(train, use = "complete.obs", method = "pearson")

# write a csv file to look in excel
# write.csv(cor_mat, file = "correlation_matrix.csv", row.names = FALSE)

# data3 <- as.data.frame(c(num_hrefs = data1$num_hrefs, num_imgs = data1$num_imgs, num_videos = data1$num_videos, num_keywords = data1$num_keywords, kw_avg_avg = data1$kw_avg_avg, self_reference_avg_sharess = data1$self_reference_avg_sharess, is_weekend = data1$is_weekend, LDA_03 = data1$LDA_03, global_subjectivity = data1$global_subjectivity, abs_title_sentiment_polarity = data1$abs_title_sentiment_polarity, shares = data1$shares))


# train_ind_10 <- c(14, 15, 8, 6, 11, 37, 40, 41, 56, 45)
# #
# # # Divide data to test
# test_10 <- as.data.frame(cbind(test[,train_ind_10], shares = test[,59]))
# validation_10 <- as.data.frame(cbind(validation[,train_ind_10], shares = validation[,59]))
# train_10 <- as.data.frame(cbind(train[,train_ind_10], shares = train[,59]))


# Divide data to test
test_10 <- cbind(num_hrefs = test$num_hrefs, num_imgs = test$num_imgs, num_videos = test$num_videos, data_channel_is_world = test$data_channel_is_world, kw_max_min = test$kw_max_min, self_reference_avg_sharess = test$self_reference_avg_sharess, is_weekend = test$is_weekend, LDA_03 = test$LDA_03, global_subjectivity = test$global_subjectivity, kw_avg_avg = test$kw_avg_avg, shares = test$shares)


# Divide train data to train and validation
validation_10 <- cbind(num_hrefs = validation$num_hrefs, num_imgs = validation$num_imgs, num_videos = validation$num_videos, data_channel_is_world = validation$data_channel_is_world, kw_max_min = validation$kw_max_min, self_reference_avg_sharess = validation$self_reference_avg_sharess, is_weekend = validation$is_weekend, LDA_03 = validation$LDA_03, global_subjectivity = validation$global_subjectivity, kw_avg_avg = validation$kw_avg_avg, shares = validation$shares)
train_10 <- cbind(num_hrefs = train$num_hrefs, num_imgs = train$num_imgs, num_videos = train$num_videos, data_channel_is_world = train$data_channel_is_world, kw_max_min = train$kw_max_min, self_reference_avg_sharess = train$self_reference_avg_sharess, is_weekend = train$is_weekend, LDA_03 = train$LDA_03, global_subjectivity = train$global_subjectivity, kw_avg_avg = train$kw_avg_avg, shares = train$shares)



####################################################################################
# preparing train data to implement cost function and gradient descent

X_train_10 <- train_10[,1:10]
y_train_10 <- train_10[,11]


# Normalize all X variables
X_norm_10 <- apply(X_train_10, 2, normal)

X_full_10 <- cbind(intercept, X_norm_10)

X_mat_10 <- as.matrix(X_full_10)
y_mat_10 <- as.matrix(y_train_10)


###################################################################################
# preparing validation data to implement cost function and gradient descent

X_valid_10 <- validation_10[,1:10]
y_valid_10 <- validation_10[,11]


# Normalize all X variables in test set
X_norm_valid_10 <- apply(X_valid_10, 2, normal)

X_full_valid_10 <- cbind(intercept_valid, X_norm_valid_10)

X_mat_valid_10 <- as.matrix(X_full_valid_10)
y_mat_valid_10 <- as.matrix(y_valid_10)




###################################################################################
# preparing test data to implement cost function and gradient descent

X_test_10 <- test_10[,1:10]
y_test_10 <- test_10[,11]


# Normalize all X variables in test set
X_norm_test_10 <- apply(X_test_10, 2, normal)

X_full_test_10 <- cbind(intercept_test, X_norm_test_10)

X_mat_test_10 <- as.matrix(X_full_test_10)
y_mat_test_10 <- as.matrix(y_test_10)



##################################################################################################################
# Retrain the model

# Define the variables
alp_10 <- 0.2
num_iter_10 <- 3000

theta_10 <- matrix(rep(0, ncol(X_full_10)))

# Run gradient decent and cost functions to return the cost history every iteration and theta at minimum cost
values_10 <- gradient_desc(X_mat_10, y_mat_10, theta_10, alp_10,  num_iter_10)

# Assign the cost history to variable J_histoty
J_history_10 <- values_10[[2]]
J_history_10

plot(1:num_iter_10, J_history_10, col = "blue", main = "Cost over Iterations - Alpha = 0.2", xlab = "Iterations", ylab = "Cost")

# Assign the theta values at lowest cost to variable theta_final
theta_final_10 <- values_10[[1]]



####################################################################

# predict y in train data with gradient descent theta

y_hat_10 <- rep(0, nrow(train_10))

y_hat_10 <- X_mat_10 %*% theta_final_10

mse_train_10 <- (sum((y_mat_10 - y_hat_10)^2))/nrow(train_10)
mape_train_10 <- (sum((abs(y_mat_10 - y_hat_10))/y_mat_10))/nrow(train_10)

###################################################################
# Testing on validation data
#################################################################


# predict y in validation data

y_valid_hat_10 <- rep(0, nrow(validation_10))

y_valid_hat_10 <- X_mat_valid_10 %*% theta_final_10

mse_valid_10 <- (sum((y_mat_valid_10 - y_valid_hat_10)^2))/nrow(validation_10)
mape_valid_10 <- (sum((abs(y_mat_valid_10 - y_valid_hat_10))/y_mat_valid_10))/nrow(validation_10)



###################################################################
# Testing on test data
#################################################################


# predict y in test data

y_test_hat_10 <- rep(0, nrow(test_10))

y_test_hat_10 <- X_mat_test_10 %*% theta_final_10

mse_test_10 <- (sum((y_mat_test_10 - y_test_hat_10)^2))/nrow(test_10)
mape_test_10 <- (sum((abs(y_mat_test_10 - y_test_hat_10))/y_mat_test_10))/nrow(test_10)

########################################################################################################
#
# Logistic regression
#
##############################################################################################################


# Train data
y_log_train_10 <- as.matrix(cut(y_mat_10, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_train_10 <- mapply(y_log_train_10, FUN = as.numeric)

y_log_train_10 <- matrix(data = y_log_train_10, ncol = 1, nrow = nrow(y_mat_10))

summary(y_log_train_10)

# Validation data

y_log_valid_10 <- as.matrix(cut(y_mat_valid_10, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_valid_10 <- mapply(y_log_valid_10, FUN = as.numeric)

y_log_valid_10 <- matrix(data = y_log_valid_10, ncol = 1, nrow = nrow(y_mat_valid_10))



# Test data

y_log_test_10 <- as.matrix(cut(y_mat_test_10, c(0, median(y_train), max(y_train)), labels = c(0,1)))

y_log_test_10 <- mapply(y_log_test_10, FUN = as.numeric)

y_log_test_10 <- matrix(data = y_log_test_10, ncol = 1, nrow = nrow(y_mat_test_10))


##################################################################################################################

# Define the variables
alp_log_10 <- 0.2
num_iter_log_10 <- 3000

theta_log_10 <- matrix(rep(0, ncol(X_full_10)))

# Run gradient decent and cost functions to return the cost history every iteration and theta at minimum cost
values_log_10 <- gradient_desc(X_mat_10, y_log_train_10, theta_log_10, alp_log_10,  num_iter_log_10)

# Assign the cost history to variable J_histoty_log
J_history_log_10 <- values_log_10[[2]]
J_history_log_10

plot(1:num_iter_log_10, J_history_log_10, col = "blue", main = "Cost over Iterations - Aplha = 0.2", ylab = "Cost")

# Assign the theta values at lowest cost to variable theta_final_log
theta_final_log_10 <- values_log_10[[1]]



####################################################################

# predict y in train data with gradient descent theta

y_hat_log_10 <- rep(0, nrow(train_10))

y_hat_log_10 <- ifelse(X_mat_10 %*% theta_final_log_10 >= 0.5, 1, 0)

y_total_train_10 <- as.data.frame(cbind(y_log_train_10, y_hat_log_10))
colnames(y_total_train_10) <- c("actual_y", "predicted_y")

table(y_total_train_10)

# model accuracy

# True negative
specificity_10 <- 6597/(6597+3250)
false_positive_10 <- 1 - specificity_10

# True positive
sensitivity_10 <- 5019/(5019+4558)
false_negative_10 <- 1 - sensitivity_10



###################################################################
# Testing on validation data
#################################################################


# predict y in validation data

y_valid_hat_log_10 <- rep(0, nrow(validation_10))

y_valid_hat_log_10 <- ifelse(X_mat_valid_10 %*% theta_final_log_10 >= 0.5, 1, 0)

y_total_valid_10 <- as.data.frame(cbind(y_log_valid_10, y_valid_hat_log_10))
colnames(y_total_valid_10) <- c("actual_y", "predicted_y")

table(y_total_valid_10)

# model accuracy

# True negative
specificity_valid_10 <- 2862/(2862+1398)
false_positive_valid_10 <- 1 - specificity_valid_10

# True positive
sensitivity_valid_10 <- 2862/(2862+1398)
false_negative_valid_10 <- 1 - sensitivity_valid_10


###################################################################
# Testing on test data
#################################################################


# predict y in test data

y_test_hat_log_10 <- rep(0, nrow(test_10))

y_test_hat_log_10 <- ifelse(X_mat_test_10 %*% theta_final_log_10 >= 0.5, 1, 0)


y_total_test_10 <- as.data.frame(cbind(y_log_test_10, y_test_hat_log_10))
colnames(y_total_test_10) <- c("actual_y", "predicted_y")

table(y_total_test_10)

# model accuracy

# True negative
specificity_test_10 <- 4835/(4835+1140)
false_positive_test_10 <- 1 - specificity_test_10

# True positive
sensitivity_test_10 <- 2215/(2215+3704)
false_negative_test_10 <- 1 - sensitivity_test_10




