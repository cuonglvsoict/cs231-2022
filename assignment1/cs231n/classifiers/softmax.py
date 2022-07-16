from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    z = X.dot(W) # n x C
    train_num = X.shape[0]
    classes_num = W.shape[1]

    for i in range(train_num):
      sum = 0
      for j in range(classes_num):
        sum += np.exp(z[i][j])

      loss += (-z[i][y[i]] + np.log(sum))

      for j in range(classes_num):
        if j==y[i]:
          dW[:,j] += X[i] * (np.exp(z[i][y[i]]) / sum - 1)
        else:
          dW[:,j] += X[i] * (np.exp(z[i][j]) / sum)

    loss /= train_num
    loss += reg * np.sum(W * W)

    dW /= train_num
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train_num = X.shape[0]
    classes_num = W.shape[1]

    z = X.dot(W)
    exp = np.exp(z)
    sum_exp = np.sum(exp, axis=1)
    sfm = exp / sum_exp.reshape(train_num, 1)

    loss = -np.sum(np.log(sfm[range(train_num), y]))
    loss /= train_num
    loss += reg * np.sum(W * W)

    sfm[range(train_num), y] -= 1
    dW = X.T.dot(sfm)
    dW /= train_num
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
