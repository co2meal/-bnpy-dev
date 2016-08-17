import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import inv
from scipy.special import digamma, gammaln, psi

import math

from bnpy.allocmodel.topics import ELBO
from bnpy.viz import BarsViz


def SLDA_VariationalBayes(Data,resp,theta,eta,Lik,tokens,delta=1.0,alpha=1.0,Niters=50,figkey=None):

	nDoc = Data.nDoc
	#nUniqueTokens = max(Data.word_id) + 1
	response = Data.response
	K,V = Lik.shape
	elbos = []

	#DTC_true = np.zeros((nDoc,resp.shape[1]))
	#DTC_inferred = np.zeros((nDoc,resp.shape[1]))
	#rand_doc_sample = np.random.randint(0,nDoc,10)
	
	for Niter in xrange(Niters):	
		#print Niter
		print 'Local Step'
		# Update local parameters resp and theta
		resp_idx_start = 0 
		for d in xrange(nDoc):
			#print d
			start = Data.doc_range[d] 
			stop = Data.doc_range[d+1] 
			wid_d = Data.word_id[start:stop]
			wc_d = Data.word_count[start:stop]
			N_d = int(sum(wc_d))
			
			response_d = Data.response[d]
			theta_d = theta[d]
			resp_d = resp[resp_idx_start:resp_idx_start+N_d,:].copy()
			
			tokens_d = tokens[resp_idx_start:resp_idx_start+N_d]
			#print tokens_d
			# Local updates
			resp_d_update, theta_d_update = local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,tokens_d,delta,alpha)
			resp[resp_idx_start:resp_idx_start+N_d,:] = resp_d_update
			theta[d] = theta_d_update
			
			#DTC_inferred[d,:] = np.sum(resp_d_update,axis=0)
			
			resp_idx_start += N_d
		
		# Global updates, eta and topics/Lik
		print 'Global Step'
		eta, Lik, delta = global_step(Data,resp,theta,response,V)
		
		print 'token / topic prob for 1st topic after iter %d: ' % Niter
		print Lik[0]

		print 'eta after iter %d: ' % Niter
		print np.round(eta,0)
		#print 'delta after iter %d: ' % Niter
		#print np.round(delta,4)
		
		if figkey is not None:
			h = BarsViz.showTopicsAsSquareImages(Lik)
			#h.savefig('/Users/leah/bnpy_repo/bnpy-dev/bnpy/allocmodel/topics/%s_after_iter_%d.png' % (figkey,Niter))
			h.savefig('%s_after_iter_%d.png' % (figkey,Niter))
		
		elbo = ELBO.calc_elbo(Data,resp,theta,eta,Lik)
		print 'elbo:', elbo
		elbos.append(elbo)
		print 'delta: ', delta
		
	return resp,theta,eta,Lik,delta,elbos


      
    
        
def local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,tokens_d,delta,alpha):
	# Each row of resp in one unique token in the document

	convThrLP = 0.001
	N_d,K = resp_d.shape	

	'''
	tokens = []
	for i in range(len(wid_d)):
		tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
	'''
	converged = False
	#iterNo = 0
	while converged == False:
		#iterNo += 1
		for i,v in enumerate(tokens_d):
			
			#t1 = digamma(theta_d)
			t1 = digamma(theta_d) - digamma(sum(theta_d))
			
			t2 = np.inner(eta, response_d / float(N_d * delta))
			
			R = np.sum(resp_d,axis=0) - resp_d[i,:]
			t3 = np.inner(eta,R)
			t3 = t3 * eta /  float(delta * N_d * N_d)
			
			t4 = (eta * eta) /  float(2 * delta * N_d * N_d)

			resp_d[i,:] = (Lik[:,v] * np.exp(t1 + t2 - t3 - t4)).transpose()
			
			# Normalize
			rsum = resp_d[i,:].sum()
			resp_d[i,:] = resp_d[i,:] / rsum
	

		#theta_d = theta_d
		prev_theta_d = theta_d.copy()
		theta_d = alpha + np.sum(resp_d, axis=0)			
		
		# Check for convergence
		maxDiff = np.max(np.abs(theta_d - prev_theta_d))
		if maxDiff < convThrLP:
			converged = True
			return resp_d, theta_d
		
	#return resp_d_out ,theta_d
	


def global_step(Data,resp,theta,response,V):
	
	nDoc = Data.nDoc
	nUniqueTokens = max(Data.word_id) + 1
	K = resp.shape[1]
	
	
	# Update token / topic distributions (Lik)
	'''
	Lik = np.zeros((K,V))
	for k in xrange(K):
		#print k
		resp_idx_start = 0
		for d in xrange(nDoc):
			start = Data.doc_range[d] 
			stop = Data.doc_range[d+1] 

			wid_d = Data.word_id[start:stop]
			wc_d = Data.word_count[start:stop]
			N_d = int(sum(wc_d))

			resp_d = resp[resp_idx_start:resp_idx_start + N_d,:]

			tokens = []
			for i in range(len(wid_d)):
				tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
			
			for i,v in enumerate(tokens):
				Lik[k,v] += resp_d[i,k]
			
			resp_idx_start += N_d

	# Normalize Lik
	Lsum = Lik.sum(axis=1)
	Lik_update = Lik / Lsum[:,np.newaxis]
	'''
	Lik_update = np.zeros((K,V))

	for k in xrange(K):
		resp_idx_start = 0
		for d in xrange(nDoc):
			start = Data.doc_range[d] 
			stop = Data.doc_range[d+1] 
			wid_d = Data.word_id[start:stop]
			wc_d = Data.word_count[start:stop]
			N_d = int(sum(wc_d))
			resp_d = resp[resp_idx_start:resp_idx_start + N_d,:]

			tokens = []
			for i in range(len(wid_d)):
				tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])

			#for i,v in enumerate(tokens):
			#	Lik_from_resp_true[k,v] += resp_d[i,k]

			for v in range(V):
				for i,n in enumerate(tokens):
					if v == n:
						Lik_update[k,v] += resp_d[i,k]

			resp_idx_start += N_d

	# Normalize Lik
	Lsum = Lik_update.sum(axis=1)
	Lik_update = Lik_update / Lsum[:,np.newaxis]
	
	
	'''
	Lik_update = np.zeros((K,V))

	for k in range(K):
		for v in range(V):
			resp_idx_start = 0
			for d in xrange(nDoc):
				start = Data.doc_range[d] 
				stop = Data.doc_range[d+1] 
				wid_d = Data.word_id[start:stop]
				wc_d = Data.word_count[start:stop]
				N_d = int(sum(wc_d))
				resp_d = resp[resp_idx_start:resp_idx_start + N_d,:]

				tokens = []
				for i in range(len(wid_d)):
					tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
			
				for i,n in enumerate(tokens):
					if n == v:
						Lik_update[k,v] += resp_d[i,k]
				
				resp_idx_start += N_d
			
	# Normalize Lik
	Lsum = Lik_update.sum(axis=1)
	Lik_update = Lik_update / Lsum[:,np.newaxis]
	'''
	# Calculate terms for eta and delta update
	
	EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K

	EXTX = np.zeros((K,K)) #E[X^T X], KxK
	
	resp_idx_start = 0
	for d in xrange(nDoc):
		#EXTX_d = 0
		#print d
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp[resp_idx_start:resp_idx_start+N_d,:]

		EX[d,:] = (1 / float(N_d)) * np.sum(resp_d,axis=0)
		
		EXTX_d = np.zeros((K,K))
		for n in range(N_d):
			for m in range(N_d):
				if m != n:
					EXTX_d += np.outer(resp_d[n,:],resp_d[m,:])
		
		for n in range(N_d):
			EXTX_d += np.diag(resp_d[n,:])
			
		
		EXTX += (1/ float(N_d * N_d)) * EXTX_d	
		#EXTX += EXTX_d
			
		'''
		R2 = np.sum(resp_d,axis=0)
		R2 = np.outer(R2,R2)
		
		EXTX += (1/ float(N_d * N_d)) * R2
		'''
		resp_idx_start += N_d

	EXTXinv = inv(EXTX)  
	
	# Update eta
	eta_update = np.dot(EX.transpose(), response)
	eta_update = np.dot(EXTXinv,eta_update)


	# Update delta	
	'''
	A = np.dot(response,response)
	B = np.dot(response,EX)
	C = np.dot(B,EXTXinv)
	D = np.dot(C,EX.transpose())
	E = np.dot(D,response)
	'''
	
	delta_update = np.dot(response,EX)
	delta_update = np.dot(response,response) - np.dot(delta_update, eta_update)
	delta_update = delta_update / float(nDoc)
	
	#delta_update = 1.0

	return eta_update, Lik_update, delta_update
