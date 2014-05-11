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
				temp = PiMat[branch] * message[c]
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