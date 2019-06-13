import numpy as np


def log_likelihood_grader(X, y, w):
    h_x = np.dot(X, w)
    log_likelihood = np.sum(y*h_x - np.log(1 + np.exp(h_x)))
    return log_likelihood


def sigmoid_grader(z):
    sgmd = 1 / (1 + np.exp(-z))
    return sgmd


def gradient_grader(X, y, w):
    h_x = np.dot(X, w)
    y_hat = sigmoid_grader(h_x)
    grad = np.dot(X.T, (y - y_hat))
    return grad


def logistic_regression_grader(X, y, max_iter, alpha):
    n, d = X.shape
    weights = np.zeros(d)
    losses = np.zeros(max_iter)    
    
    for step in range(max_iter):
        grad = gradient_grader(X, y, weights)
        weights += alpha * grad
        losses[step] = log_likelihood_grader(X, y, weights)
    
    return weights, losses