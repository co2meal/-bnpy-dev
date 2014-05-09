'''
QuadTreeUtil.py 

A module for comparing sum-product algorithm with brute force approach on quad trees
'''
import numpy as np
import math


def calcRespBySumProduct(PiInit, PiMat, logSoftEv):
	'''Execute sum-product algorithm for binary trees
	'''
	PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
	logSoftEv = _parseInput_SoftEv(logSoftEv, K)
	N = logSoftEv.shape[0]

	SoftEv, lognormC = expLogLik(logSoftEv)
	'''umsg = UpwardPass(PiInit, PiMat, SoftEv)
	dmsg, margPrObs = DownwardPass(PiInit, PiMat, SoftEv, umsg)

	respPair = np.zeros((N,K,K))
	for n in xrange( 1, N ):
		parent = get_parent_index(n)
		siblings = get_children_indices(parent,N)
		siblings.remove(n)
		message = 1
		message *= dmsg[parent]
		for s in siblings:
			message *= np.dot(PiMat, umsg[s] * SoftEv[s])
		respPair[n,:,:] = PiMat * np.outer(message, umsg[n] * SoftEv[n])
		respPair[n,:,:] /= np.sum(respPair[n,:,:])

	#logMargPrSeq = np.log(np.sum(umsg[4]*margPrObs[N-1])) + lognormC.sum()
	resp = dmsg * umsg
	logMargPrSeq = np.log(resp[N-1].sum()) + lognormC.sum()
	resp = resp / resp.sum(axis=1)[:,np.newaxis]
	return resp, respPair, logMargPrSeq'''
	umsg, umsg_helper, margStateDist, margPrObs = UpwardPass_Smoothed(PiInit, PiMat, SoftEv)
	dmsg = DownwardPass_Smoothed(PiInit, PiMat, SoftEv, umsg, umsg_helper, margStateDist)
	resp = umsg*dmsg
	respPair = np.zeros((N,K,K))
	for n in xrange( 1, N ):
		parent = get_parent_index(n)
		respPair[n] = PiMat * np.outer(resp[parent]/umsg_helper[n], umsg[n]/margStateDist[n])
	logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()
	return resp, respPair, logMargPrSeq


def UpwardPass(PiInit, PiMat, SoftEv):
	'''Propagate messages upwards along the tree, starting from the leaves
	'''
	N = SoftEv.shape[0]
	K = PiInit.size

	umsg = np.ones( (N, K) )
	start = find_last_nonleaf_node(N)
	if start is None:
		# Base case N=1. Only the root exists.
		return umsg
	for n in xrange(start-1, -1, -1):
		children = get_children_indices(n, N)
		for child in children:
			umsg[n] = umsg[n] * np.dot(PiMat, umsg[child] * SoftEv[child])
		normalization_const = np.sum(umsg[n])
		#umsg[n] /= normalization_const
	return umsg


def DownwardPass(PiInit, PiMat, SoftEv, umsg):
	'''Propagate messages downwards along the tree, starting from the root
	'''
	N = SoftEv.shape[0]
	K = PiInit.size
	PiT = PiMat.T

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
				message *= np.dot(PiMat, umsg[s]* SoftEv[s])
			dmsg[n] = np.dot(PiT, message) * SoftEv[n]
			margPrObs[n] = np.sum(dmsg[n])
			#dmsg[n] /= margPrObs[n]
	return dmsg, margPrObs

def UpwardPass_Smoothed(PiInit, PiMat, SoftEv):
	'''Upward pass with normalized messages
	'''
	N = SoftEv.shape[0]
	K = PiInit.size
	PiT = PiMat.T
	start = find_last_nonleaf_node(N)
	umsg = np.ones( (N,K) )
	margStateDist = np.empty( (N,K) )
	umsg_helper = np.empty( (N,K) )
	margStateDist[0] = PiInit
	margPrObs = np.empty( N )
	for n in xrange(1, N):
		margStateDist[n] = np.dot(PiT, margStateDist[get_parent_index(n)])
	for n in xrange(N-1, -1, -1):
		if(n > start-1):
			umsg[n] = SoftEv[n] * margStateDist[n]
			margPrObs[n] = np.sum(umsg[n])
			umsg[n] /= margPrObs[n]
			umsg_helper[n] = np.dot(PiMat, umsg[n]/margStateDist[n])
		else:
			children = get_children_indices(n, N)
			for c in children:
				umsg[n] = umsg[n] * umsg_helper[c]
			umsg[n] = umsg[n] * (SoftEv[n] * margStateDist[n])
			margPrObs[n] = np.sum(umsg[n])
			umsg[n] /= margPrObs[n]
			umsg_helper[n] = np.dot(PiMat, umsg[n]/margStateDist[n])
	return umsg, umsg_helper, margStateDist, margPrObs

def DownwardPass_Smoothed(PiInit, PiMat, SoftEv, umsg, umsg_helper, margStateDist):
	'''Downward pass with normalized messages
	'''
	N = SoftEv.shape[0]
	K = PiInit.size
	PiT = PiMat.T
	dmsg = np.ones( (N,K) )
	for n in xrange(1, N):
		parent = get_parent_index(n)
		dmsg[n] = np.dot(PiT, (dmsg[parent]*umsg[parent])/umsg_helper[n]) / margStateDist[n]
	return dmsg

########################################################### Brute Force
###########################################################
def calcRespByBruteForce(PiInit, PiMat, logSoftEv):
	''' Calculate marginal state-assignments for binary trees via brute force.

		Returns
		-------
		resp : 2D array, size N x K
		respPair : 3D array, size N x K x K
		logPrObs : scalar log probability
	'''
	N = logSoftEv.shape[0]
	PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
	logSoftEv = _parseInput_SoftEv(logSoftEv, K)
	SoftEv, lognormC = expLogLik(logSoftEv)
	if N > 21:
		raise ValueError("Brute force is too expensive for N=%d!" % (N))
	resp = np.zeros((N,K))
	respPair = np.zeros((N,K,K))
	margPrObs = 0
	for configID in xrange(K ** N):
		Ztree = makeZTree(configID, N, K)
		prTree = calcProbOfTree(Ztree, PiInit, PiMat, SoftEv)
		margPrObs += prTree
		for n in range(N):
			resp[n, Ztree[n]] += prTree
			if n > 0:
				pa = get_parent_index(n)
				respPair[n, Ztree[pa], Ztree[n]] += prTree
	resp /= resp.sum(axis=1)[:,np.newaxis]
	for n in range(1, N):
		respPair[n] /= respPair[n].sum()

	return resp, respPair, np.log(margPrObs) + np.sum(lognormC)

def makeZTree(configID, N, K):
	'''Create configuration of hidden state variables for all nodes in tree.

		Examples
		--------
		>>> makeZTree(0, 3, 2)
		[0, 0, 0]
		>>> makeZTree(7, 3, 2)
		[1, 1, 1]
		>>> makeZTree(7, 4, 2)
		[0, 1, 1, 1]
		>>> makeZTree(7, 4, 3)
		[0, 0, 2, 1]
	'''
	Ztree = np.zeros( N, dtype=np.int32)
	for n in range(N-1, -1, -1):
		posID = N - n - 1
		Ztree[posID] = configID / (K**n)   
		configID = configID - Ztree[posID] * (K**n)
	return Ztree

def calcProbOfTree(Ztree, PiInit, PiMat, SoftEv):
	''' Calculate joint prob of assignments and observations for entire tree.
	'''
	N = SoftEv.shape[0]
	prTree = PiInit[Ztree[0]] * SoftEv[0,Ztree[0]]
	for n in xrange(1, N):
		parent = get_parent_index(n)
		prTree *= PiMat[Ztree[parent], Ztree[n]] * SoftEv[n,Ztree[n]]
	return prTree

########################################################### tree utilities
###########################################################
def get_parent_index(n):
	if n == 0:
		return None # it is a root
	elif n % 4 == 0:
		return (n-1)/4
	else:
		return n/4

def get_children_indices(n, N):
	if 4 * n + 1 >= N:
		return [] # it is a leaf
	else:
		return [4 * n + j + 1 for j in range(4)]

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
	assert PiMat.ndim == 2
	J, K = PiMat.shape
	assert J == K
	return PiInit, PiMat, K

def _parseInput_SoftEv(logSoftEv, K):
	logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
	Nl, Kl = logSoftEv.shape
	assert Kl == K
	return logSoftEv
