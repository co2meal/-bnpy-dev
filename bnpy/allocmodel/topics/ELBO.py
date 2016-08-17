import math
import numpy as np
from scipy.special import digamma, gammaln
from bnpy.allocmodel.topics import slda_helper

def calc_elbo(Data,resp_extended,theta,eta,Lik):

	nDoc = Data.nDoc
	#digammaSumTheta = digamma(theta.sum(axis=1))
	#ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]

	response = Data.response
	
	elbo = 0
	resp_start_idx = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		#resp_d = resp[start:stop,:]
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		
		theta_d = theta[d]
		response_d = response[d]
		
		resp_d = resp_extended[resp_start_idx:resp_start_idx + N_d,:]
		#resp_d_extended = slda_helper.expand_resp_single_doc(resp_d,wid_d,wc_d)
		
		elbo_d = calc_elbo_single_doc(resp_d,theta_d,response_d,N_d,wid_d,wc_d,eta,Lik)
		elbo += elbo_d
		
		resp_start_idx += N_d

	return elbo
	
	
def calc_elbo_single_doc(resp_d,theta_d,response_d,N_d,wid_d,wc_d,eta,Lik,delta=1.0,alpha=1.0):
	
	elbo_alloc = calc_elbo_alloc(resp_d,theta_d)
	elbo_obsv = calc_elbo_obsv(resp_d,wid_d,wc_d,Lik)
	
	elbo_SLDA = calc_elbo_SLDA(response_d,resp_d,N_d,wc_d,eta,delta)
	
	elbo_entropy = calc_elbo_entropy(resp_d,theta_d)
	
	#print elbo_alloc, elbo_obsv, elbo_SLDA, elbo_entropy
	return elbo_alloc + elbo_obsv + elbo_SLDA - elbo_entropy
	

			
def calc_elbo_alloc(resp_d,theta_d,alpha=1.0):
	# Calculate E[log p(Pi)] + E[ log p(z)]
	
	#E[log p(Pi)]
	N,K = resp_d.shape
	avec = alpha * np.ones(K)
	ElogPi_d = gammaln(np.sum(avec)) - np.sum(gammaln(avec))
	
	sum_theta_d = theta_d.sum()
	
	if alpha != 1:
		ElogPi_d += np.dot((avec - 1), (digamma(theta_d) - digamma(sum_theta_d)))

	# E[ log p(z)]
	
	Elogz_d = 0
	'''
	for n in range(N):
		for k in range(K):
			Elogz_d += resp_d[n,k] * (digamma(theta_d[k]) - digamma(sum_theta_d))
	'''
	
	digamma_diff = (digamma(theta_d) - digamma(sum_theta_d))
	for n in range(N):
		Elogz_d += np.dot(resp_d[n,:],digamma_diff)
		#Elogz_d += np.dot(resp_d[n,:],(digamma(theta_d) - digamma(sum_theta_d)))

	return ElogPi_d + Elogz_d

	
def calc_elbo_obsv(resp_d,wid_d,wc_d,Lik):
	# Calculate E[log p(w) ] 	
	K = resp_d.shape[1]
	
	tokens = []
	for i in range(len(wid_d)):
		tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
	
	Lik_d = 0
	for i,v in enumerate(tokens):
		Lik_d += np.dot(resp_d[i,:], np.log(Lik[:,v]))
		
	return Lik_d
	

def calc_elbo_entropy(resp_d,theta_d,alpha=1.0):
	# Calculate E[log q(z)] + E[log(Pi)]

	qElogz_d = np.sum(resp_d * np.log(resp_d))
	
	#qElogz_d = np.sum(np.dot(resp_d,np.log(resp_d)))
	'''
	N,K = resp.shape
	qElogz_d = 0
	for n in range(N):
		for k in range(K):
			qElogz_d += resp[n,k] * np.log(resp[n,k])
	'''
	qElogPi_d = gammaln(np.sum(theta_d)) - np.sum(gammaln(theta_d))
	
	qElogPi_d += np.dot((theta_d - 1), (digamma(theta_d) - digamma(theta_d.sum())))
	
	return qElogz_d + qElogPi_d
	
	

def calc_elbo_SLDA(response_d,resp_d,N_d,wc_d,eta,delta=1.0):
	#Calculate E [log p(y)]
	
	N_d, K = resp_d.shape
	'''	
	# eta (\sum rdn) (\sum rdn) eta
	N_resp, K = resp_d.shape	
	
	R = np.zeros(K) # \sum_{n=1}^{N_d} resp_d[n,:]
	for n in xrange(N_resp):
		R += resp_d[n,:] #* wc_d[n]
	
	sterm = np.outer(R, R)
	sterm = np.dot(eta,sterm)
	sterm = np.dot(sterm,eta)
	
	
	sterm = np.dot(eta,R)
	sterm = sterm * sterm
	'''
	EZ_d = (1.0 / N_d) * np.sum(resp_d,axis=0)
	
	#sterm = np.dot(eta,EZ_d)
	#sterm = sterm * sterm
	
	
	EZTZ_d = np.zeros((K,K))
	for n in range(N_d):
		for m in range(N_d):
			if m != n:
				EZTZ_d += np.outer(resp_d[n,:],resp_d[m,:])
	
	for n in range(N_d):
		EZTZ_d += np.diag(resp_d[n,:])
		
	
	EZTZ_d = (1/ float(N_d * N_d)) * EZTZ_d	
	'''	
	sterm = 0
	for n in range(N_d):
		for m in range(N_d):
			if m != n:
				sterm += np.outer(resp_d[n,:],resp_d[m,:])
			else:
				sterm += np.diag(resp_d[n])

	
	sterm = np.dot(eta,sterm)
	sterm = np.dot(sterm,eta)
	'''
	
	sterm = np.dot(eta,EZTZ_d)
	sterm = np.dot(sterm,eta)
			
	slda_elbo = np.log( 1 / np.sqrt(2 * math.pi * delta))
	#slda_elbo = - np.log(2 * math.pi * delta) / 2
	slda_elbo -= ((response_d * response_d )/ (2 * delta))
	#slda_elbo += (response_d / (delta * N_d)) * np.dot(eta, EZ_d) # np.dot(eta,np.sum(resp_d,axis=0))
	slda_elbo += (response_d / delta) * np.dot(eta, EZ_d)
	slda_elbo -= (1 / (2 * delta * N_d * N_d)) * sterm

	return slda_elbo
	
	
	
	
	