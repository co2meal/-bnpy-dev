'''
TargetPlanner.py

Handles advanced selection of a plan-of-attack for improving a current model via a birthmove.  

Key methods
--------
  * select_target_comp
'''
import numpy as np

import BirthProposalError

EPS = 1e-14
MIN_SIZE = 25

def select_target_comp(K, SS=None, model=None, LP=None, Data=None,
                           lapsSinceLastBirth=defaultdict(int),
                           excludeList=list(), doVerbose=False, 
                           **kwargs):
  ''' Choose a single component among possible choices {0,2,3, ... K-2, K-1}
      to target with a birth proposal.

      Keyword Args
      -------
      randstate : numpy RandomState object, allows random choice of ktarget
      targetSelectName : string, identifies procedure for selecting ktarget
                          {'uniform','sizebiased', 'delaybiased',
                           'delayandsizebiased'}

      Returns
      -------
      ktarget : int, range: 0, 1, ... K-1
                identifies single component in the current model to target

      Raises
      -------
      BirthProposalError, if cannot select a valid choice
  '''
  if SS is None:
    targetSelectName = 'uniform'
    assert K is not None
  else:
    assert K == SS.K
  
  if len(excludeList) >= K:
    msg = 'BIRTH not possible. All K=%d targets used or excluded.' % (K)
    raise BirthProposalError(msg)

  ######## Build vector ps: gives probability of each choice
  ########
  targetSelectName = kwargs['targetSelectName']
  ps = np.zeros(K)
  if targetSelectName == 'uniform':
    ps = np.ones(K)
  elif targetSelectName == 'sizebiased':
    ps = SS.N.copy()
    ps[SS.N < MIN_SIZE] = 0
  elif targetSelectName == 'delaybiased':
    # Bias choice towards components that have not been selected in a long time
    lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
    ps = np.maximum(lapDist + 1e-5, 0)
    ps = ps * ps
  elif targetSelectName == 'delayandsizebiased':
    # Bias choice towards components that have not been selected in a long time
    #  *and* which have many members
    lapDist = np.asarray([lapsSinceLastBirth[kk] for kk in range(K)])
    ps = np.maximum(lapDist + 1e-5, 0)
    ps = ps * ps * SS.N
    ps[SS.N < MIN_SIZE] = 0
  else:
    raise NotImplementedError('Unrecognized procedure: ' + targetSelectName)

  ######## Make a choice using vector ps, if possible. Otherwise, raise error.
  ########
  ps[excludeList] = 0
  if np.sum(ps) < EPS:
    msg = 'BIRTH not possible. All K=%d targets have zero probability.' % (K)
    raise BirthProposalError(msg)
  ps = ps / np.sum(ps)  
  assert np.allclose( np.sum(ps), 1.0)

  ktarget = kwargs['randstate'].choice(K, p=ps)
  return ktarget

'''
  if doVerbose:
    sortIDs = np.argsort(-1.0 * ps) # sort in decreasing order
    for kk in sortIDs[:6]:
      msg = "comp %3d : %.2f prob | %3d delay | %8d size"
      print msg % (kk, ps[kk], lapsSinceLastBirth[kk], SS.N[kk])
'''




