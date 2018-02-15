# Machine-Learning---Gradient-descent
Implement Linear and Logistic regression on a online news sharing data set in R.

The online news sharing data set was downloaded from the following link.
https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#

The objective of this is to implement following 2 algorithms:
1. Linear regression
2. Logistic regression

Tasks:
1. Divide the dataset into train and test sets sampling randomly using only predictive attributes and the target variable
2. Use linear regression to predict the number of shares. Report and compare the train and test error/accuracy metrics - mean squared error
3. Convert this problem into a binary classification problem. The target variable will have two values (large or small number of shares).
4. Implement logistic regression to carry out classification on this data set. Report accuracy/error metrics for train and test sets.

Experimentation:
1. Experiment with various model parameters (learning rate for gradient descent) for both linear and logistic regression and observe how the error varies for train and test sets with varying these parameters.
2. Pick ten features randomly and retrain the model only on these ten features. Compare train and test error results for the case of using all features to using ten random features.
3. Pick ten features that I think are best suited to predict the output, and retrain the model using these ten features. Compare to the case of using all features and to random features case.
