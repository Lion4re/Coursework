import numpy as np
from random import shuffle

def compute_gradient_and_loss(W:np.ndarray, X:np.ndarray, y:np.ndarray, reg:float, reg_type:int, opt:int):
      """
      loss and gradient function.

      Inputs have dimension D, there are C classes,
      and we operate on minibatches of N examples.

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - X: A numpy array of shape (N, D) containing a minibatch of data.
      - y: A numpy array of shape (N,) containing training labels;
            y[i] = c means that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength
      - reg_type: (int) regularization type (1: l1, 2: l2)
      - opt: (int) 0 for computing both loss and gradient, 1 for computing loss only

      Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
      """
      assert reg_type in [1, 2], "Invalid 'reg_type' (Valid types: 1 -> l1 | 2 -> l2)"
      assert opt in [0, 1], "Invalid 'opt' (Valid options: 0 -> loss & gradient | 1 -> loss only)"
      dW = np.zeros(W.shape) # initialize the gradient as zero

      # compute the loss and the gradient
      num_classes = W.shape[1]  # C
      num_train = X.shape[0]    # N
      loss = 0.0

      #############################################################################
      # TODO:                                                                     #
      # Implement the routine to compute the loss, storing the result in loss     #
      #############################################################################
      scores = X.dot(W)
      correct_class_scores = scores[np.arange(num_train), y]
      margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
      margins[np.arange(num_train), y] = 0
      loss += np.sum(margins)

      loss /= num_train

      if reg_type == 1:
            loss += reg * np.sum(np.abs(W))  # L1 regularization
      elif reg_type == 2:
            loss += reg * np.sum(W * W)  # L2 regularization

      if opt == 1:
            return loss, None

      #############################################################################
      # TODO:                                                                     #
      # Implement the gradient for the required loss, storing the result in dW.   #
      #                                                                           #
      # Hint: Instead of computing the gradient from scratch, it may be easier    #
      # to reuse some of the intermediate values that you used to compute the     #
      # loss.                                                                     #
      #############################################################################
      binary = margins
      binary[margins > 0] = 1
      row_sum = np.sum(binary, axis=1)
      binary[np.arange(num_train), y] = -row_sum
      dW += X.T.dot(binary)

      dW /= num_train

      if reg_type == 1:
            dW += reg * np.sign(W)  # L1 regularization
      elif reg_type == 2:
            dW += 2 * reg * W  # L2 regularization

      return loss, dW

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
