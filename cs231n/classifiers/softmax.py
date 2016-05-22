import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    score = W.dot(X)
    loss = 0.0
    C, D = W.shape
    N = X.shape[1]
    dW = np.zeros(W.shape)

    for i in xrange(N):
        loss += np.log(np.sum(np.exp(score[:, i]))) - score[y[i], i]
        sum_exp = np.sum(np.exp(score[:, i]))
        for j in xrange(C):
            dW[j, :] += 1.0 / sum_exp * np.exp(score[j, i]) * X[:, i]
        dW[y[i], :] -= X[:, i]
    
    dW /= N
    loss /= N
    dW += reg * W
    loss += reg * np.sum(W ** 2)
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    N, C = X.shape[1], W.shape[0]
    score = W.dot(X)
    exp_score = np.exp(score)
    loss = 0.0
    
    loss = np.sum(np.log(np.sum(exp_score, axis=0)) - score[y, np.arange(N)])
    M = exp_score  / (np.sum(exp_score, axis=0))
    M[y, np.arange(N)] -= 1
    dW = M.dot(X.T)
    
    dW /= N
    loss /= N
    dW += reg * W
    loss += reg * np.sum(W * W)
    loss = np.mean(loss) #почему-то уже в класификаторе эта штука выдавала array, поэтому возвращаю среднее
    loss = loss[0]
    return loss, dW
