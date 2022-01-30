# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  loss.py
# Author:  Billy Carson
# Date written:  01-28-2022
# Last modified:  01-30-2022

r"""
Description:  Beta-divergence loss NumPy implementations functions definition file. Code modified from scikit-learn
implementation of beta-divergence.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT STATEMENTS
# ----------------------------------------------------------------------------------------------------------------------

# Import statements
import numpy as np
from scipy.sparse import issparse

# Define constants
EPSILON = float(np.finfo(np.float32).eps)


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Calculates beta-divergence between two numpy.ndarrays
def beta_div(a: np.ndarray, b: np.ndarray, beta: float, reduction='mean', square_root=False) -> float:
    r"""
    Beta-divergence loss function. Code modified from scikit-learn implementation of beta-divergence.
    
    Parameters
    ----------
    beta : float or int
        Beta value for beta-divergence loss. Default is 0 (Itakura-Saito divergence).
    reduction : str
        Loss reduction type. Default is 'mean' (average over batch size and number of features).
    square_root : bool
        Indicates whether to take square root of final loss value. Default is False.
    
    Returns
    -------
    loss_val : float
        Beta-divergence of arrays a and b (target).
    """
    
    # Check input and target tensor dimensions
    if len(a.shape) != len(b.shape):
        raise ValueError('Input and target matrices must have same number of dimensions.')
    
    # Check input and target tensor shapes
    if a.shape != b.shape:
        raise ValueError('Input and target matrices must have same shape.')
    
    # Check for proper type/value of beta-divergence beta value
    if (not isinstance(beta, float)) & (not isinstance(beta, int)):
        raise TypeError('Beta must be an integer or float value greater than or equal to 0.')
    elif beta < 0.0:
        raise ValueError('Beta must be a value greater than or equal to 0.')
    
    # Check for proper type/value of reduction type
    if not isinstance(reduction, str):
        raise TypeError('Reduction type must be a string.')
    elif (reduction  != 'mean') & (reduction  != 'batchmean') & (reduction  != 'sum'):
        raise ValueError('Reduction type not recognized. Accepted reductions are \"mean\", \"batchmean\", ' + \
                         'and \"sum\".')
    
    # Check for proper type/value of square root option
    if not isinstance(square_root, bool):
        raise TypeError('Square root option must be of type bool.')
    
    # Get number of samples and features from shape of data
    b_shape = np.array(b.shape)
    n_samp = b_shape[0]
    n_feat = b_shape[1]
    
    # Flatten input and target matrices
    a_flat = a.ravel()
    b_flat = b.ravel()
    
    # Do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    eps_idx = b_flat > EPSILON
    a_flat = a_flat[eps_idx]
    b_flat = b_flat[eps_idx]
    
    # Used to avoid division by zero
    # a_flat[a_flat == 0] = EPSILON
    a_flat[a_flat < EPSILON] = EPSILON
    
    # Generalized Kullback-Leibler divergence
    if beta == 1:            
        # Computes sum of target * log(target / input) only where target is non-zero
        div = b_flat / a_flat
        loss_val = np.dot(b_flat, np.log(div))
        
        # Add difference between full sum of input matrix and full sum of target matrix
        loss_val += np.sum(a_flat) - np.sum(b_flat)
    
    # Itakura-Saito divergence
    elif beta == 0:
        div = b_flat / a_flat
        loss_val = np.sum(div) - np.prod(b_shape) - np.sum(np.log(div))
    
    # Calculate beta-divergence when beta not equal to 0, 1, or 2
    else:
        a_beta_sum = np.sum(a ** beta)
        b_input_sum = np.dot(b_flat, a_flat ** (beta - 1.0))
        loss_val = np.sum(b_flat ** beta) - (beta * b_input_sum)
        loss_val += a_beta_sum * (beta - 1.0)
        loss_val /= beta * (beta - 1.0)
    
    # Mean reduction
    if reduction == 'mean':
        loss_val /= (n_samp * n_feat)
    
    # Batch-wise mean reduction
    elif reduction == 'batchmean':
        loss_val /= n_samp
    
    # Square root of beta-divergence loss
    if square_root:
        loss_val /= np.sqrt(2.0 * loss_val)
    
    # Return beta-divergence loss
    return loss_val


# Calculates beta-divergence between numpy.ndarray of data and numpy.ndarray of NMF approximation of data
def nmf_beta_div(X: np.ndarray, H: np.ndarray, W: np.ndarray, beta: float, reduction='mean',
                 square_root=False) -> float:
    r"""
    NMF beta-divergence loss function. Code modified from scikit-learn implementation of beta-divergence.
    
    Parameters
    ----------
    beta : float or int
        Beta value for beta-divergence loss. Default is 0 (Itakura-Saito divergence).
    reduction : str
        Loss reduction type. Default is 'mean' (average over batch size and number of features).
    square_root : bool
        Indicates whether to take square root of final loss value. Default is False.
    
    Returns
    -------
    loss_val : float
        Beta-divergence of X and product of matrices H and W.
    """
    
    # Check data and reconstruction tensor dimensions
    if len(X.shape) != len(H.shape[:-1] + W.shape[1:]):
        raise ValueError('Data matrix and data reconstruction matrix must have same number of dimensions.')
    
    # Check input and target tensor shapes
    if X.shape != H.shape[:-1] + W.shape[1:]:
        raise ValueError('Data matrix and data reconstruction matrix must have same shape.')
    
    # Get number of samples and features from shape of data
    X_shape = np.array(X.shape)
    n_samp = X_shape[0]
    n_feat = X_shape[1]
    
    # Frobenius norm
    if beta == 2:
        # Avoid creation of dense matrix multiplication of H and W if X is sparse
        if issparse(X):
            X_norm = X @ X.T
            X_hat_norm = _trace_dot(a=((H.T @ H) @ W), b=W)
            cross_prod = _trace_dot(a=(X * W.T), b=H)
            loss_val = (X_norm + X_hat_norm - (2.0 * cross_prod)) / 2.0
        else:
            loss_val = _squared_norm(a=X - (H @ W)) / 2.0
    
    # Compute X_hat where X is not equal to zero
    if issparse(X):
        X_hat_flat = _special_sparse_mm(X=X, H=H, W=W).ravel()
        X_flat = X.ravel()
    
    # Compute X_hat, X is not sparse
    else:
        X_hat = H @ W
        X_hat_flat = X_hat.ravel()
        X_flat = X.ravel()
    
    # Do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    eps_idx = X_flat > EPSILON
    X_hat_flat = X_hat_flat[eps_idx]
    X_flat = X_flat[eps_idx]
    
    # Used to avoid division by zero
    # X_hat_flat[X_hat_flat == 0] = EPSILON
    X_hat_flat[X_hat_flat < EPSILON] = EPSILON
    
    # Generalized Kullback-Leibler divergence
    if beta == 1:
        # Fast and memory efficient computation of sum of elements of matrix multiplication of H and W
        X_hat_sum = np.dot(np.sum(H, axis=0), np.sum(W, axis=1))
        
        # Computes sum of X * log(X / HW) only where X is non-zero
        div = X_flat / X_hat_flat
        loss_val = np.dot(X_flat, np.log(div))
        
        # Add difference between full sum of matrix multiplication of H and W and full sum of X
        loss_val += X_hat_sum - np.sum(X_flat)
    
    # Itakura-Saito divergence
    elif beta == 0:
        div = X_flat / X_hat_flat
        loss_val = np.sum(div) - np.prod(X_shape) - np.sum(np.log(div))
    
    # Calculate beta-divergence when beta not equal to 0, 1, or 2
    else:
        if issparse(X):
            X_hat_beta_sum = np.sum((H @ W) ** beta)
        else:
            X_hat_beta_sum = np.sum(X_hat ** beta)
        X_X_hat_sum = np.dot(X_flat, X_hat_flat ** (beta - 1.0))
        loss_val = np.sum(X_flat ** beta) - (beta * X_X_hat_sum)
        loss_val += X_hat_beta_sum * (beta - 1.0)
        loss_val /= beta * (beta - 1.0)
    
    # Mean reduction
    if reduction == 'mean':
        loss_val /= (n_samp * n_feat)
    
    # Batch-wise mean reduction
    elif reduction == 'batchmean':
        loss_val /= n_samp
    
    # Square root of beta-divergence loss
    if square_root:
        loss_val /= np.sqrt(2.0 * loss_val)
    
    # Return beta-divergence loss
    return loss_val


# Trace dot product method
def _trace_dot(a: np.ndarray, b: np.ndarray) -> float:
    r"""
    Trace of dot product between arrays a and b.
    
    Parameters
    ----------
    a : numpy.ndarray
        First array.
    b : numpy.ndarray
        Second array.
    
    Returns
    -------
    a_b_trace_dot : float
        Trace of dot product between tensors a and b.
    """
    
    # Compute trace dot
    a_b_trace_dot = np.dot(a.ravel(), b.ravel())
    
    # Return computed trace dot
    return a_b_trace_dot


# Special sparse dot product
def _special_sparse_mm(X: np.ndarray, H: np.ndarray, W: np.ndarray) -> np.ndarray:
    r"""
    Computes product of matrices of H and W where X is non-zero.
    
    Parameters
    ----------
    X : numpy.ndarray
        Data array; shape (n_samples, n_features).
    H : numpy.ndarray
        Scores/encodings array; shape (n_samples, n_components).
    W : numpy.ndarray
        Components array; shape (n_components, n_features).
    
    Returns
    -------
    X_hat : numpy.ndarray
        Produc of matrices of H and W.
    """
    
    # Product of matrices H and W where X is non-zero
    if issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = H.shape[1]
        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.sum(np.multiply(H[ii[batch], :], W.T[jj[batch], :]), axis=1)
        X_hat = dot_vals[ii, :]
        X_hat = X_hat[:, jj]
    
    # For non-sparse X, compute product of matrices H and W
    else:
        X_hat = H @ W
    
    # Return product of matrices H and W
    return X_hat


# Squared norm method
def _squared_norm(a: np.ndarray) -> float:
    r"""
    Squared Euclidean or Frobenius norm of a.
    
    Parameters
    ----------
    a : numpy.ndarray
        Array of which to compute squared norm.
    
    Returns
    -------
    float
        The Euclidean norm when a is a vector, Frobenius norm when a is a matrix (2D array).
    """
    
    # Flatten array a
    a_flat = a.ravel()
    
    # Return dot product
    return np.dot(a_flat, a_flat)

