import math
import numpy as np


def hello_world(s):
    """Given an input string s, return a new string.  For example if the
    paramter s contains the string 'John', return the new string 'Hello John!'

    Arguments
    ---------
    s - A string is provided as an input parameter to this function.

    Returns
    -------
    string - returns a new string of the form 'Hello <s>!'
    """
    # remove the return and replace with your own code
    return "" 

def basic_sigmoid(x):
    """Compute sigmoid of the scalar (float/double usualy) input
    parameter and return.  This basic version should not be vectorized,
    and is required to use the `exp` function from the `math` library.

    Arguments
    ---------
    x - a scalar (float/double) value

    Returns
    -------
    scalar float/double - The sigmoid of the input scalar parameter x is calculated and
        returned.
    """
    # remove the return and replace with your own code
    return 0.0

def sigmoid(x):
    """Compute sigmoid of the input parameter x and return. In this version
    the input parameter might be a scalar, but it could be a list or
    a numpy array.  Your implementation should be vectorized and able to
    hanld all of these.

    Arguments
    ---------
    x - a scalar, python list or numpy array of real valued (float/double) numbers.

    Returns
    -------
    s - Result will be of the same shape as the input and will be the element wise
      calculation of the sigmoid for all values given as input.
    """
    # remove the return and replace with your own code
    return np.zeros(3)


def sigmoid_grad(x):
    """Compute the gradient (also called the slope or derivative) of the sigmoid function with respect
    to its input x.  You can store the output of the sigmoid function into a variable and then use it to
    calculate the gradient.  You are required to reuse your previous `sigmoid()` function in this function.

    Arguments
    ---------
    x - A scalar or numpy array of real valued numbers

    Returns
    -------
    ds - The computed gradients of the sigmoid of x with respect to the x inputs.
    
    """
    # remove the return and replace with your own code
    return np.zeros(3)

def image2vector(image):
    """Flatten a 3-D tensor into a 1-D.  You can assume that the given image
    is a 3-D tensor (though it wouldn't hurt to check this with asserts). The
    result of flattening an image of shape (length, width, channels) will be a new
    row vector of shape (length * width * channels, 1).

    Arguments
    ----------
    image - a 3-D tensor of shape (length, width, channels)

    Returns
    -------
    vector - Technically you should return a 2-D row tensor, of shape (length * width * channels, 1)
    """
    # remove the return and replace with your own code
    return np.zeros(3)

def normalize_rows(x):
    """Implement a function that normalizes each row of the matrix x (to have unit length).
    This function should not modify the passed in matrix x, so you will need to make
    a copy of x to manipulate and return from this function.

    Arguments
    ---------
    x - A numpy matrix (2-D tensor) of shape (n, m), e.g. n rows x m columns

    Returns
    -------
    x - The normalized (by row) numpy matrix.  Each row will be a vecotr that has unit length.
    """
    # remove the return and replace with your own code
    return np.zeros(3)

def softmax(x):
    """Calculates the softmax for each row of the input 2-D tensor matrix x.
    Your code should work for a row vector of shape (1, n) but also for general
    matrices of shape (m, n).  You shouldn't have to do anything special to handl
    a row vector, numpy broadcasting should work in either case.

    Arguments
    ---------
    x - A numpy matrix (2-D tensor) of shape (m, n)

    Returns
    -------
    s - A numpy matrix with the same shape (m, n) as the input argument x, with the computed softmax of x
    """
    # remove the return and replace with your own code
    return np.zeros(3)

def l2_loss(y_pred, y_true):
    """This function computes the mean squared error / L2 loss function.  We expect 2 vectors of the
    same size as input to this function.

    Arguments
    ---------
    y_pred - a 1-d tensor / vector of size m, the predicted labels
    y_true - a 1-d tensor / vector of the same size m, the true labels

    Returns
    -------
    loss - the value of the L2 / MSE loss function for y_pred and y_true
    """
    # remove the return and replace with your own code
    return 0.0

def l1_loss(y_pred, y_true):
    """This function computes the L1 loss.  We expect 2 vectors of the
    same size as input to this function.

    Arguments
    ---------
    y_pred - a 1-d tensor / vector of size m, the predicted labels
    y_true - a 1-d tensor / vector of the same size m, the true labels

    Returns
    -------
    loss - the value of the L1 (absolute value) loss function for y_pred and y_true
    """
    # remove the return and replace with your own code
    return 0.0
