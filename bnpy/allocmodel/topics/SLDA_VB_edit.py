import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import inv
from scipy.special import digamma, gammaln, psi

import math

from bnpy.allocmodel.topics import ELBO

def SLDA_VariationalBayes(Data,resp,theta,eta,Lik,delta=1.0,alpha=1.0,Niters=50):
	nDoc = Data.nDoc
	nUniqueTokens = max(Data.word_id) + 1
	response = Data.response
	K,V = Lik.shape
	elbos = []
	for Niter in xrange(Niters):	
		#print Niter
		print 'Local Step'
		# Update local parameters, resp and theta
		resp_idx_start = 0 
		for d in xrange(nDoc):
			#if d % 100 == 0:
			#	print d
			start = Data.doc_range[d] 
			stop = Data.doc_range[d+1] 

			wid_d = Data.word_id[start:stop]
			wc_d = Data.word_count[start:stop]
			N_d = int(sum(wc_d))
			
			response_d = Data.response[d]
			theta_d = theta[d]
			resp_d = resp[resp_idx_start:resp_idx_start+N_d,:] 
			
			# Local updates
			resp_d_update, theta_d_update = local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,delta,alpha)
			resp[resp_idx_start:resp_idx_start+N_d,:] = resp_d_update
			theta[d] = theta_d_update
			
			resp_idx_start += N_d
		
		# Global updates, eta and topics/Lik
		print 'Global Step'
		#eta, Lik = global_step_extended(Data,resp,theta,response,eta,V,delta=1.0,alpha=1.0)
		eta, Lik_update = global_step(Data,resp,theta,response,eta,V,delta=1.0,alpha=1.0)
		print eta
		#if Lik_update is not None:
		Lik = Lik_update
		
		#if Niter % 10 == 0:
		#	elbo = ELBO.calc_elbo(Data,resp,theta,eta)
		#	print elbo
		#	elbos.append(elbo)
		
	return resp,theta,eta,Lik,elbos


      
    
        
def local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,delta,alpha):
	# Each row of resp in one unique token in the document
	convThrLP = 0.001
	N_d,K = resp_d.shape	

	tokens = []
	for i in range(len(wid_d)):
		tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
	
	converged = False
	iterNo = 0
	while converged == False:
		#if iterNo % 100 == 0:
		#	print 'Iter No: ', iterNo
		#iterNo += 1
		for i,v in enumerate(tokens):
			t1 = digamma(theta_d)
			t2 = (response_d * eta) / (N_d * delta)
			
			R = np.sum(resp_d,axis=0) - resp_d[i]
			
			t3 = np.outer(eta.transpose(),R)
			t3 = np.dot(t3,eta) /  (delta * N_d * N_d)
			
			t4 = (eta * eta) /  (2 * delta * N_d * N_d)


			resp_d[i,:] = Lik[:,v] * np.exp(t1 + t2 - t3 - t4)
		
			rsum = resp_d[i,:].sum()
			resp_d[i,:] = resp_d[i,:] / rsum
			
		#theta_d = theta_d
		prev_theta_d = theta_d.copy()
		theta_d = alpha + np.sum(resp_d, axis=0)	
		#Niter += 1

		# Check for convergence
		maxDiff = np.max(np.abs(theta_d - prev_theta_d))
		#print maxDiff
		if maxDiff < convThrLP:
			return resp_d, theta_d

	


def global_step(Data,resp,theta,response,eta,V,delta=1.0,alpha=1.0):
	
	nDoc = Data.nDoc
	nUniqueTokens = max(Data.word_id) + 1
	
	# Update token / topic distributions (Lik)
	
	K = resp.shape[1]
	#Lik_update = None
	
	Lik = np.zeros((K,V))
	for k in xrange(K):
		#print k
		resp_idx_start = 0
		for d in xrange(nDoc):
			start = Data.doc_range[d] 
			stop = Data.doc_range[d+1] 
					
			#resp_d = resp[start:stop,:]
			wid_d = Data.word_id[start:stop]
			wc_d = Data.word_count[start:stop]
			N_d = int(sum(wc_d))
			resp_d = resp[resp_idx_start:resp_idx_start + N_d,:]
			#N_resp = resp_d.shape[0]
			
			tokens = []
			for i in range(len(wid_d)):
				tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
			
			for i,v in enumerate(tokens):
				Lik[k,v] += resp_d[i,k]
				
			resp_idx_start += N_d

	# Normalize Lik
	Lsum = Lik.sum(axis=1)
	Lik_update = Lik / Lsum[:,np.newaxis]

	print'update eta'
	# Update eta
	EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K

	EXTX = np.zeros((K,K)) #E[X^T X], KxK
	
	resp_idx_start = 0
	for d in xrange(nDoc):
		#print d
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp[resp_idx_start:resp_idx_start+N_d,:]

		R1 = np.sum(resp_d,axis=0)
		EX[d] = (1 / float(N_d)) * R1

		temp = np.multiply.outer(resp_d, resp_d)
		C = np.swapaxes(temp, 1, 2)
		R2 = np.zeros((K,K))
		for i in range(C.shape[0]):
			for k in range(C.shape[2]):
				R2 += C[i][j]
		'''
		R2 = np.zeros((K,K)) 
		for n in xrange(N_d):
			for m in xrange(N_d):
				R2 += np.outer(resp_d[n,:], resp_d[m,:])
		'''
		EXTX += (1/ float(N_d * N_d)) * R2
		
		resp_idx_start += N_d
	
	'''
	EXTX = np.zeros((K,K)) #E[X^T X], KxK
	resp_idx_start = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp[resp_idx_start:resp_idx_start+N_d,:]

		R = np.zeros((K,K)) 
		for n in xrange(N_d):
			for m in xrange(N_d):
				R += np.outer(resp_d[n,:], resp_d[m,:])

		EXTX += (1/ float(N_d * N_d)) * R
		resp_idx_start += N_d
	'''	
	EXTXinv = inv(EXTX)  

	eta_update = np.dot(EXTXinv,EX.transpose())
	eta_update = np.dot(eta_update, response)

	return eta_update, Lik_update
