'''
HMTUtil.py

Provides sum-product algorithm for HMTs
'''
import numpy as np
import math


def SumProductAlg_QuadTree(PiInit, PiMat, logSoftEv):
	'''Execute sum-product algorithm given HMT state
	   transition params and log likelihoods of each observation

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 4 2D arrays, size KxK
            piMat[i, j] is the transition distribution on branch i from state j to all
            K states. each row must be probability vector (positive entries, sums to one)
     logSoftEv : 2D array, size NxK
            logSoftEv[n] := log p( x[n] | z[nk] = 1)
                         log likelihood of observation n under state k
     Returns
     -------
     resp : 2D array, size N x K
            resp[n,k] = marg. prob. that step t assigned to state K
                        p( z[n,k] = 1 | x[1], x[2], ... x[N])
     respPair : 2D array, size N x K x K
            respPair[n,j,k] = marg. prob. that both
                              * node n assigned to state k
                              * parent of node n assigned to state j
                        p( z[pa(n),j] = 1, z[n,k] = 1 | x[1], x[2], ... x[N])
            respPair[0,:,:] is undefined, but kept to match indexing consistent. 

  	'''
	PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
	logSoftEv = _parseInput_SoftEv(logSoftEv, K)
	N = logSoftEv.shape[0]

	SoftEv, lognormC = expLogLik(logSoftEv)
	umsg = UpwardPass(PiInit, PiMat, SoftEv)
	dmsg, margPrObs = DownwardPass(PiInit, PiMat, SoftEv, umsg)

	respPair = np.zeros((N,K,K))

	for n in xrange( 1, N ):
		parent = get_parent_index(n)
		siblings = get_children_indices(parent,N)
		siblings.remove(n)
		message = 1
		message *= dmsg[parent]
		for s in siblings:
			branch = get_branch(s)
			message *= np.dot(PiMat[branch,:,:], umsg[s] * SoftEv[s])
		respPair[n] = PiMat[get_branch(n),:,:] * np.outer(message, umsg[n] * SoftEv[n])
		respPair[n] /= np.sum(respPair[n])

	#logMargPrSeq = np.log(margPrObs[N-1]) + lognormC.sum()
	resp = dmsg * umsg
	logMargPrSeq = np.log(resp[N-1].sum()) + lognormC.sum()
	resp = resp / resp.sum(axis=1)[:,np.newaxis]
	return resp, respPair, logMargPrSeq


def UpwardPass(PiInit, PiMat, SoftEv):
	'''Propagate messages upwards along the tree, starting from the leaves
    Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 4 2D array, size 4xKxK
            piMat[i,j] is transition distribution on branch ifrom state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size NxK
            SoftEv[n] := p( x[n] | z[nk] = 1)
                         likelihood of observation n under state k
                         given up to an additive constant for each n
     Returns
     -------
        umsg : 2D array, size N x K
                  probability of state k on latent variable n, given all 
                  observations from its predecessors and its observation
                  umsg[n,k] = p( z[n,k] = 1 | x[c(c(n))]...x[c(n)] ... x[n] )
	'''
	N = SoftEv.shape[0]
	K = PiInit.size
	umsg = np.ones( (N, K) )
	start = find_last_nonleaf_node(N)
	for n in xrange(start-1, -1, -1):
		children = get_children_indices(n, N)
		for child in children:
			branch = get_branch(child)
			umsg[n] = umsg[n] * np.dot(PiMat[branch,:,:], umsg[child]*SoftEv[child])
		normalization_const = np.sum(umsg[n])
		#umsg[n] /= normalization_const
	return umsg


def DownwardPass(PiInit, PiMat, SoftEv, umsg):
	'''Propagate messages downwards along the tree, starting from the root

    Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 4 2D array, size 4xKxK
            piMat[i,j] is transition distribution on branch ifrom state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size NxK
            SoftEv[n] := p( x[n] | z[nk] = 1)
                         likelihood of observation n under state k
                         given up to an additive constant for each n
     Returns
     -------
        dmsg : 2D array, size N x K
                  dmsg[n,k] = p( x[p(n)], x[p(p(n))], ... x[1] |  z[n,k] = 1 )
	'''
	N = SoftEv.shape[0]
	K = PiInit.size
	PiTMat = np.empty( (4,K,K) )
	for d in xrange(0, 4):
		PiTMat[d,:,:] = PiMat[d,:,:].T

	margPrObs = np.empty( N )
	dmsg = np.empty( (N,K) )
	for n in xrange( 0, N ):
		if n == 0:
			dmsg[n] = PiInit * SoftEv[0]
			margPrObs[n] = np.sum(dmsg[n])
			#dmsg[n] /= margPrObs[n]
		else:
			parent_index = get_parent_index(n)
			siblings = get_children_indices(parent_index, N)
			siblings.remove(n)
			message = 1
			message *= dmsg[parent_index]
			for s in siblings:
				branch = get_branch(s)
				message *= np.dot(PiMat[branch,:,:], umsg[s]* SoftEv[s])
			branch = get_branch(n)
			dmsg[n] = np.dot(PiTMat[branch,:,:], message) * SoftEv[n]
		margPrObs[n] = np.sum(dmsg[n])
		#dmsg[n] /= margPrObs[n]
	return dmsg, margPrObs

def get_parent_index(child_index):
	if child_index == 0:
		return None #it is a root
	elif child_index%4 == 0:
		return (child_index-1)/4
	else:
		return child_index/4

def get_children_indices(parent, N):
	if 4*parent+1 > N:
		return []
	else:
		myList = [4*parent+j+1 for j in range(4)]
		return myList

def get_branch(child_index):
	'''Find on which branch of its parent this child lies
	'''
	if (child_index%4 == 0):
		return 3
	else:
		return (child_index%4 - 1)

def find_last_nonleaf_node(N):
	'''Get the index of last nonleaf node in the data
	'''
	if N == 1:
		return None
	else:
		height = 1
		total = 1
		while (total + 4**height) < N:
			total += 4**height
			height += 1
		return total

########################################################### expLogLik
###########################################################
def expLogLik(logSoftEv, axis=1):
	''' Return element-wise exp of input log likelihood
        guaranteed not to underflow
    
      Returns
      --------
      SoftEv : 2D array, size NxK
                equal to exp(logSoftEv), up to prop constant for each row
      lognormC : 1D array, size N
                gives log of the prop constant for each row
	'''
	lognormC = np.max(logSoftEv, axis)
	if axis==0:
		logSoftEv = logSoftEv - lognormC[np.newaxis,:]
	elif axis==1:
		logSoftEv = logSoftEv - lognormC[:,np.newaxis]
	SoftEv = np.exp(logSoftEv)
	return SoftEv, lognormC

########################################################### Parse input
###########################################################
def _parseInput_TransParams(PiInit, PiMat):
	PiInit = np.asarray(PiInit, dtype=np.float64)
	PiMat = np.asarray(PiMat, dtype=np.float64)
	assert PiInit.ndim == 1
	K0 = PiInit.shape[0]
	assert PiMat.ndim == 3
	I, J, K = PiMat.shape
	assert J == K
	assert K0 == K
	assert I == 4
	return PiInit, PiMat, K

def _parseInput_SoftEv(logSoftEv, K):
	logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
	Nl, Kl = logSoftEv.shape
	assert Kl == K
	return logSoftEv
