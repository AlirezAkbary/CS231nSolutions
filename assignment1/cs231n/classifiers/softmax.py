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
  num_train = X.shape[0]
  num_classes = W.shape[1]      
  for i in xrange(num_train):
      scores = np.dot(X[i], W)
      temp = np.sum(np.exp(scores))
      loss += (np.log(temp) - scores[y[i]])
      for j in xrange(num_classes):
          if j == y[i]:
              dW[:, j] += (np.exp(scores[y[i]])*X[i]/temp - X[i])
              continue
          dW[:, j] += X[i] * np.exp(scores[j]) / temp
      
  loss /= num_train
  dW /= num_train
  
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  

      
          
          
          
      
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  score_matrix = np.dot(X, W)
  ###normalizing for stability
  score_matrix -= np.max(score_matrix, axis=1, keepdims=True)
  #score_matrix -= score_matrix.max()
  score_matrix = np.exp(score_matrix)

  
  loss_vector = (score_matrix[np.arange(num_train),y])/(np.sum(score_matrix, axis=1))
  loss = -np.sum(np.log(loss_vector))/num_train + reg*np.sum(W*W)
  ### notice that you can't divide a matrix with a 1-D array that has a common row number
  ###possible if they had common column number
  score_matrix /= np.sum(score_matrix, axis=1, keepdims = True)
  
  score_matrix[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, score_matrix)
  
  dW += 2*reg*W
  
  
  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

