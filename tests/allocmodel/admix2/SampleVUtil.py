import numpy as np
import warnings

def sampleVd(u, nDoc=100, alpha=0.5, PRNG=np.random.RandomState(0)):
  K = u.size
  cumprod1mu = np.ones(K)
  cumprod1mu[1:] *= np.cumprod(1 - u[:-1])

  Vd = np.zeros((nDoc, K))
  for k in xrange(K):
    Vd[:,k] = PRNG.beta( alpha * cumprod1mu[k] * u[k],
                         alpha * cumprod1mu[k] * (1. - u[k]),
                         size=nDoc)
    ## Warning: beta rand generator can fail when both params
    ## are very small (~1e-8). This will yield NaN values.
    ## To fix, we use fact that Beta(eps, eps) will always yield a 0 or 1.
    badIDs = np.flatnonzero(np.isnan(Vd[:,k]))
    if len(badIDs) > 0:
      p = np.asarray( [1. - u[k], u[k]] )
      Vd[badIDs, k] = PRNG.choice([1e-12, 1-1e-12], len(badIDs), p=p, replace=True)
  assert not np.any(np.isnan(Vd))
  assert np.all(np.isfinite(Vd))
  return Vd

def summarizeVd(Vd):
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message='divide by zero')
    logVd = np.log(Vd)
    log1mVd = np.log(1-Vd)

  assert not np.any(np.isnan(logVd))
  logVd = replaceInfVals(logVd)
  log1mVd = replaceInfVals(log1mVd)
  return np.sum(logVd, axis=0), np.sum(log1mVd, axis=0)

def summarizeVdToPi(Vd):
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                               message='divide by zero')
    logVd = np.log(Vd)
    log1mVd = np.log(1-Vd)
    mask = Vd < 1e-15
    log1mVd[mask] = np.log1p( -1*Vd[mask] )

  assert not np.any(np.isnan(logVd))
  logVd = replaceInfVals(logVd)
  log1mVd = replaceInfVals(log1mVd)
  ElogVd = np.sum(logVd, axis=0)
  Elog1mVd = np.sum(log1mVd, axis=0)
  ElogPi = np.hstack([ElogVd, 0])
  ElogPi[1:] += np.cumsum(Elog1mVd)
  return ElogPi

def replaceInfVals( logX, replaceVal=-100):
  infmask = np.isinf(logX)
  logX[infmask] = replaceVal
  return logX

