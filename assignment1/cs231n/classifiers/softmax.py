import numpy as np
from random import shuffle

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

  train_num = X.shape[0]
  class_num = W.shape[1]

  for i in range(train_num):
      scores = X[i].dot(W)
      scores -= np.max(scores)

      ## based on the calculus,
      ## \frac{\partial{L_i}} {\partial{x_i}} == \frac {e^{x_i}} {\sum{e^{x_j}}}

      p = lambda k: np.exp(scores[k]) / np.sum(np.exp(scores))
      loss += -1 * np.log(p(y[i]))

      ## get the deviation of class y[i], which means the right class.
      ## X[i] is a D-dimension vector, e^x / sum(e^x{j}) get the deviation vector
      ## of this class

      for c in range(class_num):
          p_c = p(c)   ## get the probability of nth
          dW[:, c] += (p_c - (c == y[i])) * X[i]
          ## if c is y[i], which is right class, (p - 1)* X
          ## else p * X

  loss /= train_num
  loss += 0.5 * reg * np.sum(W*W)

  dW /= train_num
  dW += reg * W  ## if W is huge, the derivation became huge.

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  train_num = X.shape[0]

  P = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  ## there are some probability is the product of the classes that not existed in the train labels
  ## and the weights, therefore, we need to delete these probabilities by filter.

  loss = np.sum(-1 * np.log(P[np.arange(train_num), y]))
  #loss = -1 * np.sum(np.log(np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)))

  loss /= train_num
  loss += 0.5 * reg * np.sum(W*W)


  index = np.zeros_like(P)
  index[np.arange(train_num), y] = 1
  dW = X.T.dot(P - index)
  dW /= train_num
  dW += reg * W
  ################# ############################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

