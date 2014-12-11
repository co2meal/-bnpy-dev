import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import nnls

def nnls_activeset(X, Y, W_init=None):
  n_samples, n_features = X.shape
  n_targets = Y.shape[1]
  W = np.zeros((n_features, n_targets), dtype=np.float64)
  allres = 0
  for colID in xrange(Y.shape[1]):
    W[:,colID], res = nnls(X, Y[:,colID])
    allres += res
  return W, res

def nnls_lbfgs(X, Y, Winit=None, factr=1e12, max_iter=50, tol=1e-8):
    """Non-negative least squares solver using L-BFGS-B.
        
    Solves for W in
    min 0.5 ||Y - XW||^2_F + + l1_reg * sum(W) + 0.5 * l2_reg * ||W||^2_F
    
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    G = np.dot( X.T, X)
    flatXy = np.dot(X.T, Y).ravel()

    def f(w, *args):
        W = w.reshape((n_features, n_targets))
        diff = np.dot(X, W) - Y
        res = 0.5 * np.sum(np.square(diff))
        return res

    def fprime(w, *args):
        W = w.reshape((n_features, n_targets))
        grad = (np.dot(G, W)).ravel() - flatXy
        return grad

    if Winit is None:
      Winit = 1.0/n_features * np.ones((n_features * n_targets,), dtype=np.float64)
    else:
      Winit = np.reshape( Winit, (n_features * n_targets))

    W, residual, Info = fmin_l_bfgs_b(
                f, x0=Winit, fprime=fprime, pgtol=tol, factr=factr,
                bounds=[(0, None)] * n_features * n_targets,
                maxiter=max_iter)
    
    # testing reveals that sometimes, very small negative values occur
    W[W < 0] = 0
    residual = np.sqrt(2 * residual)
    if Info['warnflag'] > 0:
        print("L-BFGS-B failed to converge")
    
    return W.reshape((n_features, n_targets)), residual, Info
