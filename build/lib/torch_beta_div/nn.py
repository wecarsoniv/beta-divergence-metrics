# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  nn.py
# Author:  Billy Carson
# Date written:  10-19-2021
# Last modified:  01-05-2022

"""
Description:  Beta-divergence PyTorch custom loss class definition file. Code modified from scikit-learn implementation
of beta-divergence.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT STATEMENTS
# ----------------------------------------------------------------------------------------------------------------------

# Import statements
import numpy as np
from scipy.sparse import issparse
import torch
import torch.nn as nn

# Define constants
EPSILON = float(np.finfo(np.float32).eps)


# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Beta-divergence loss class
class BetaDivLoss(nn.modules.loss._Loss):
    r"""
    Beta divergence loss class. Code modified from scikit-learn implementation of beta-divergence to work with PyTorch.
    
    Parameters
    ----------
    beta : float or int
        Beta value for beta divergence loss. Default is 0 (Itakura-Saito divergence).
    reduction : str
        Loss reduction type. Default is 'mean' (average over batch size and number of features).
    
    Attributes
    ----------
    beta : float or int
        Beta value for beta divergence loss. Default is 0 (Itakura-Saito divergence).
    reduction : str
        Loss reduction type. Default is 'mean' (average over batch size and number of features).
    
    Methods
    -------
    forward(X, H, W)
        Beta-divergence loss forward method.
    _trace_dot(a, b)
        Trace of dot product between tensors a and b.
    _special_sparse_mm(X, H, W)
        Computes product of matrices H and W, only where X is non zero.
    _squared_norm(a)
        Squared Euclidean or Frobenius norm of a.
    """
    
    # Beta-divergence loss instantiation method
    def __init__(self, beta, reduction='mean'):
        r"""
        Beta divergence loss class instantiation method.
        
        Parameters
        ----------
        beta : float or int
            Beta value for beta divergence loss. Default is 0 (Itakura-Saito divergence).
        reduction : str
            Loss reduction type. Default is 'mean' (average over batch size and number of features).
        
        Returns
        -------
        none
        """
        
        # Inherit from loss base class
        super().__init__()
        
        # Check for proper type/value and assign beta divergence beta value
        if (not isinstance(beta, float)) & (not isinstance(beta, int)):
            raise TypeError('Beta must be an integer or float value greater than or equal to 0.')
        elif beta < 0.0:
            raise ValueError('Beta must be a value greater than or equal to 0.')
        self.beta = beta
        
        # Check for proper type/value and assign reduction type
        if not isinstance(reduction, str):
            raise TypeError('Reduction type must be a string.')
        elif (reduction  != 'mean') & (reduction  != 'batchmean') & (reduction  != 'sum'):
            raise ValueError('Reduction type not recognized. Accepted reductions are \"mean\", \"batchmean\", ' + \
                             'and \"sum\".')
        self.reduction = reduction
    
    # Beta-divergence loss forward method
    def forward(self, X, H, W):
        r"""
        Beta-divergence loss forward method.
        
        Parameters
        ----------
        X : torch.Tensor
            Data tensor; shape (n_samples, n_features).
        H : torch.Tensor
            Scores/encodings tensor; shape (n_samples, n_components).
        W : torch.Tensor
            Components tensor; shape (n_components, n_features).
        
        Returns
        -------
        beta_div_loss : torch.Tensor
            Beta divergence of X and product of matrices H and W.
        """
        
        # Get number of samples and features from shape of data
        X_shape = X.shape
        n_samp = X_shape[0]
        n_feat = X_shape[1]
        
        # Frobenius norm
        if self.beta == 2:
            # Avoid creation of dense matrix multiplication of H and W if X is sparse
            if issparse(X.detach().cpu().numpy()):
                X_norm = torch.mm(X, X.T)
                X_hat_norm = self._trace_dot(a=torch.mm(torch.mm(H.T, H), W), b=W)
                cross_prod = self._trace_dot(a=(X * W.T), b=H)
                beta_div_loss = (X_norm + X_hat_norm - (2.0 * cross_prod)) / 2.0
            else:
                beta_div_loss = self._squared_norm(a=X - torch.mm(H, W)) / 2.0

        # Compute X_hat where X is not equal to zero
        if issparse(X.detach().cpu().numpy()):
            X_hat_flattened = _special_sparse_mm(X=X, H=H, W=W).flatten()
            X_flattened = X.flatten()
        
        # Compute X_hat, X is not sparse
        else:
            X_hat = torch.mm(H, W)
            X_hat_flattened = X_hat.flatten()
            X_flattened = X.flatten()

        # Do not affect the zeros: here 0 ** (-1) = 0 and not infinity
        eps_idx = X_flattened > EPSILON
        X_hat_flattened = X_hat_flattened[eps_idx]
        X_flattened = X_flattened[eps_idx]

        # Used to avoid division by zero
        X_hat[X_hat == 0] = EPSILON

        # Generalized Kullback-Leibler divergence
        if self.beta == 1:
            # Fast and memory efficient computation of sum of elements of matrix multiplication of H and W
            X_hat_sum = torch.dot(torch.sum(H, dim=0), torch.sum(W, dim=1))
            
            # Computes sum of X * log(X / HW) only where X is non-zero
            div = X_flattened / X_hat_flattened
            beta_div_loss = torch.dot(X_flattened, torch.log(div))
            
            # Add difference between full sum of matrix multiplication of H and W and full sum of X
            beta_div_loss += X_hat_sum - torch.sum(X_flattened)

        # Itakura-Saito divergence
        elif self.beta == 0:
            div = X_flattened / X_hat_flattened
            beta_div_loss = torch.sum(div) - torch.prod(torch.Tensor(X.shape)) - torch.sum(torch.log(div))

        # Calculate beta-divergence when beta not equal to 0, 1, or 2
        else:
            if issparse(X.detach().cpu().numpy()):
                X_hat_beta_sum = torch.sum(torch.mm(H, W) ** self.beta)
            else:
                X_hat_beta_sum = torch.sum(X_hat ** self.beta)
            X_X_hat_sum = torch.dot(X_flattened, X_hat_flattened ** (self.beta - 1.0))
            beta_div_loss = torch.sum(X_flattened ** self.beta) - (self.beta * X_X_hat_sum)
            beta_div_loss += X_hat_beta_sum * (self.beta - 1.0)
            beta_div_loss /= self.beta * (self.beta - 1.0)
            
        # Mean reduction
        if self.reduction == 'mean':
            beta_div_loss /= (n_samp * n_feat)
            
        # Batch-wise mean reduction
        elif self.reduction == 'batchmean':
            beta_div_loss /= n_samp
        
        # Return beta-divergence loss
        return beta_div_loss
    
    # Trace dot product method
    @staticmethod
    def _trace_dot(a, b):
        r"""
        Trace of dot product between tensors a and b.
        
        Parameters
        ----------
        a : torch.Tensor
            First tensor.
        b : torch.Tensor
            Second tensor.
            
        Returns
        -------
        a_b_trace_dot : torch.Tensor
            Trace of dot product between tensors a and b.
        """
        
        # Compute trace dot
        a_b_trace_dot = torch.dot(a.flatten(), b.flatten())
                         
        # Return computed trace dot
        return a_b_trace_dot
    
    # Special sparse dot product
    @staticmethod
    def _special_sparse_mm(X, H, W):
        r"""
        Computes product of matrices of H and W where X is non-zero.
        
        Parameters
        ----------
        X : torch.Tensor
            Data tensor; shape (n_samples, n_features).
        H : torch.Tensor
            Scores/encodings tensor; shape (n_samples, n_components).
        W : torch.Tensor
            Components tensor; shape (n_components, n_features).
            
        Returns
        -------
        X_hat : torch.Tensor
            Produc of matrices of H and W.
        """
        
        # Product of matrices H and W where X is non-zero
        if issparse(X.detach().cpu().numpy()):
            ii, jj = X.nonzero()
            n_vals = ii.shape[0]
            dot_vals = torch.empty(n_vals)
            n_components = H.shape[1]
            batch_size = max(n_components, n_vals // n_components)
            for start in range(0, n_vals, batch_size):
                batch = slice(start, start + batch_size)
                dot_vals[batch] = torch.sum(torch.multiply(H[ii[batch], :], W.T[jj[batch], :]), dim=1)
            X_hat = dot_vals[ii, :]
            X_hat = X_hat[:, jj]
        
        # For non-sparse X, compute product of matrices H and W
        else:
            X_hat = torch.mm(H, W)
        
        # Return product of matrices H and W
        return X_hat

    # Squared norm method
    @staticmethod
    def _squared_norm(a):
        r"""
        Squared Euclidean or Frobenius norm of a.
        
        Parameters
        ----------
        a : torch.Tensor
            Tensor of which to compute squared norm.
        
        Returns
        -------
        float
            The Euclidean norm when a is a vector, Frobenius norm when a is a matrix (2D array).
        """
        
        # Flatten tensor a
        a_flattened = a.flatten()
        
        # Return dot product
        return torch.dot(a_flattened, a_flattened)

