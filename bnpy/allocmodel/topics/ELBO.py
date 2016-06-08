import math
import numpy as np
from scipy.special import digamma, gammaln

def calc_elbo(Data,resp,theta,eta):

	nDoc = Data.nDoc
	digammaSumTheta = digamma(theta.sum(axis=1))
	ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]

	response = Data.response

	elbo = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		resp_d = resp[start:stop,:]
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = sum(wc_d)
		
		
		theta_d = theta[d]
		response_d = response[d]
		
		elbo += calc_elbo_single_doc(resp_d,theta_d,response_d,N_d,wid_d,wc_d,eta)
		
	return elbo
	
	
def calc_elbo_single_doc(resp_d,theta_d,response_d,N_d,wid_d,wc_d,eta,delta=1.0,alpha=1.0):
	
	elbo_alloc = calc_elbo_alloc(resp_d,theta_d)
	elbo_obsv = calc_elbo_obsv(resp_d,wid_d,wc_d)
	
	elbo_SLDA = calc_elbo_SLDA(response_d,resp_d,N_d,wc_d,eta,delta)
	
	elbo_entropy = calc_elbo_entropy(resp_d,theta_d)
	
	return elbo_alloc + elbo_obsv + elbo_SLDA - elbo_entropy
	
	
def calc_elbo_alloc(resp_d,theta_d,alpha=1.0):
	# Calculate E[log p(Pi)] + E[ log p(z)]
	
	#E[log p(Pi)]
	K = resp_d.shape[1]
	avec = alpha * np.ones(K)
	ElogPi_d = gammaln(np.sum(avec)) - np.sum(gammaln(avec))
	
	if alpha != 1:
		ElogPi_d += np.dot((avec - 1), digamma(theta_d))

	# E[ log p(z)]
	
	Elogz_d = sum(sum(resp_d * digamma(theta_d)))
	
	return ElogPi_d + Elogz_d
	
	
def calc_elbo_obsv(resp_d,wid_d,wc_d):
	# Calculate E[log p(w) ] 
	Lik_d = 0
	for i,v in enumerate(wid_d):
		Lik_d += sum(wc_d[i] * resp_d[i])
		#Lik[:,v] += wc_d[i] * resp_d[i]
	
	return Lik_d
	

def calc_elbo_entropy(resp_d,theta_d,alpha=1.0):
	# Calculate E[log q(z)] + E[log(Pi)]

	qElogz_d = sum(sum(resp_d * np.log(resp_d)))

	qElogPi_d = gammaln(np.sum(theta_d)) - np.sum(gammaln(theta_d))
	
	if alpha != 1.0:
		qElogPi_d += np.dot((theta_d - 1), digamma(theta_d))
	
	return qElogz_d + qElogPi_d
	
	

def calc_elbo_SLDA(response_d,resp_d,N_d,wc_d,eta,delta=1.0):
	#Calculate E [log p(y)]
	
	
	# eta (\sum rdn) (\sum rdn) eta
	N_resp, K = resp_d.shape
	R = np.zeros(K) # \sum_{n=1}^{N_d} resp_d[n,:]
	for n in xrange(N_resp):
		R += resp_d[n,:] * wc_d[n]
		
	sterm = np.outer(R, R)
	sterm = np.dot(eta,sterm)
	sterm = np.dot(sterm,eta)
			
	slda_elbo = np.log( 1 / np.sqrt(2 * math.pi * delta))
	slda_elbo -= (response_d / (2 * delta))
	slda_elbo += (response_d / (delta * N_d)) * np.dot(eta, np.sum(resp_d,axis=0))
	slda_elbo -= (1 / (2 * delta * N_d * N_d)) * sterm
	
	return slda_elbo