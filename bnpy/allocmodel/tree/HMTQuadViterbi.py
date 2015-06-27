'''
HMTViterbi.py 

Implements a variation of Viterbi algorithm based on Durand et al.: Computational Methods for Hidden Markov Tree Models, 2004
'''
import numpy as np

def ViterbiAlg(PiInit, PiMat, logSoftEv):

	PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
	logSoftEv = _parseInput_SoftEv(logSoftEv, K)
	N = logSoftEv.shape[0]

	SoftEv, lognormC = expLogLik(logSoftEv)
	last = find_last_nonleaf_node(N)
	message = np.empty( (N,K) )
	encoding = np.empty( N )
	backtrack = np.empty( (last,4,K) )
	for n in xrange(N-1, -1, -1):
		message[n] = SoftEv[n]
		if n < last:
			children = get_children_indices(n, N)
			for c in children:
				branch = get_branch(c)
				temp = PiMat * message[c]
				maxValues = temp.max(1)
				message[n] *=maxValues
				indices = np.argmax(temp, axis=1)
				backtrack[n,branch,:] = indices
	encoding[0] = np.argmax(message[0]*PiInit)
	for n in xrange(0, last):
		children = get_children_indices(n, N)
		for c in children:
			branch = get_branch(c)
			encoding[c] = backtrack[n, branch,encoding[n]]
	return encoding

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

def get_parent_index(child_index):
	if child_index == 0:
		return None #it is a root
	elif child_index%4 == 0:
		return (child_index-1)/4
	else:
		return child_index/4

########################################################### Brute Force
###########################################################
def findEncodingByBruteForce(PiInit, PiMat, logSoftEv):
	N = logSoftEv.shape[0]
	PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
	logSoftEv = _parseInput_SoftEv(logSoftEv, K)
	SoftEv, lognormC = expLogLik(logSoftEv)
	if N > 21:
		raise ValueError("Brute force is too expensive for N=%d!" % (N))
	resp = np.zeros((N,K))
	respPair = np.zeros((N,K,K))
	margPrObs = 0
	maxPr = np.finfo(np.double).min
	encoding = np.zeros(N)
	for configID in xrange(K ** N):
		Ztree = makeZTree(configID, N, K)
		prTree = calcProbOfTree(Ztree, PiInit, PiMat, SoftEv)
		if prTree > maxPr:
			encoding = Ztree
			maxPr = prTree
	return encoding

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