import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...C-1, for C classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    N, C = X.shape[1], W.shape[0]
    dW = np.zeros(W.shape)
    loss = 0.0
    score = W.dot(X)

    for i in xrange(N):
        for c in xrange(C):
            if c == y[i]:
                continue
            margin = score[c][i] - score[y[i]][i] + 1
            if margin > 0:
                loss += margin
                dW[c] += X[:,i].T
                dW[y[i]] -= X[:,i].T
    dW /= N
    loss /= N
    dW += reg * W
    loss += reg * np.sum(W ** 2)
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    N = X.shape[1]
    dW = np.zeros(W.shape)
    scores = W.dot(X)
    margins = np.maximum(0, scores - scores[y, np.arange(N)] + 1)
    margins[y, np.arange(N)] = 0
    loss = np.sum(margins) / N + reg * np.sum(W ** 2)
    
    margins[margins > 0] = 1
    margins[y, np.arange(N)] = -np.sum(margins, axis=0)
    dW = margins.dot(X.T) / N

    return float(loss), dW














