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
		print Niter
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
		eta, Lik_update = global_step_extended(Data,resp,theta,response,eta,V,delta=1.0,alpha=1.0)
		print eta
		#if Lik_update is not None:
		Lik = Lik_update
		
		#if Niter % 10 == 0:
		#	elbo = ELBO.calc_elbo(Data,resp,theta,eta)
		#	print elbo
		#	elbos.append(elbo)
		
	return resp,theta,eta,Lik,elbos


      
        
        
def local_step_single_doc_slow(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,delta,alpha):
	# Each row of resp in one unique token in the document
	convThrLP = 0.001
	N_d,K = resp_d.shape	

	tokens = []
	for i in range(len(wid_d)):
		tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
	
	converged = False
	while converged == False:
		for i,v in enumerate(tokens):
			for k in xrange(K):	
				#t1 = digamma(theta_d[k])
				#t1 = np.log(theta_d[k])
				#resp_d[i,k] = Lik[k,v] * np.exp(t1)
		
				t1 = digamma(theta_d[k])
				t2 = float((response_d * eta[k])) / (N_d * delta)
		
				S = 0
				for j in xrange(K):
					S += eta[j] * (np.sum(resp_d[:,j],axis=0) - resp_d[i,j])

				t3 = (eta[k] * S) / (N_d * N_d * delta)
				t4 = (eta[k] * eta[k]) / (2 * delta * N_d * N_d)
		
				resp_d[i,k] = Lik[k,v] * np.exp(t1 + t2 - t3 - t4)
		
				#resp_d[i,k] = Lik[k,v] * np.exp(t1)
				#resp_d_nk_update = np.exp(np.log(Lik[k,v]) + t1)
				#resp_d[i,k] = resp_d_nk_update
		
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

	
	'''
	niter = 0
	while niter < 2: # fix at 20 iterations for now....
		t1 = digamma(theta_d)
		t2 = (float(response_d) / (N_d * delta)) * eta
		t3 = (1 / float(2 * N_d * N_d * delta)) * eta * eta

		for i,v in enumerate(tokens):			
			R = np.sum(resp_d,axis=0)
			resp_d_without_i = R - resp_d[i,:]
			
			curTerm = np.outer(eta,resp_d_without_i)
			curTerm = np.dot(curTerm,eta)
			curTerm *= 1 / float(N_d * N_d * delta)
			
			resp_d_update_i = Lik[:,v] * np.exp(t1 + t2 - t3 - curTerm)
			rsum = resp_d_update_i.sum()
			resp_d_update_i = resp_d_update_i / rsum
			resp_d[i,:] = resp_d_update_i
		
		theta_d = alpha + np.sum(resp_d, axis=0)	
		#theta_d = theta_d
		niter += 1
		'''
	#return resp_d, theta_d
	
        
def local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,delta,alpha):
	# Each row of resp in one unique token in the document
	convThrLP = 0.001
	N_d,K = resp_d.shape	

	tokens = []
	for i in range(len(wid_d)):
		tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])
	
	converged = False
	while converged == False:
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

	
	'''
	niter = 0
	while niter < 2: # fix at 20 iterations for now....
		t1 = digamma(theta_d)
		t2 = (float(response_d) / (N_d * delta)) * eta
		t3 = (1 / float(2 * N_d * N_d * delta)) * eta * eta

		for i,v in enumerate(tokens):			
			R = np.sum(resp_d,axis=0)
			resp_d_without_i = R - resp_d[i,:]
			
			curTerm = np.outer(eta,resp_d_without_i)
			curTerm = np.dot(curTerm,eta)
			curTerm *= 1 / float(N_d * N_d * delta)
			
			resp_d_update_i = Lik[:,v] * np.exp(t1 + t2 - t3 - curTerm)
			rsum = resp_d_update_i.sum()
			resp_d_update_i = resp_d_update_i / rsum
			resp_d[i,:] = resp_d_update_i
		
		theta_d = alpha + np.sum(resp_d, axis=0)	
		#theta_d = theta_d
		niter += 1
		'''
	#return resp_d, theta_d

def global_step_extended(Data,resp,theta,response,eta,V,delta=1.0,alpha=1.0):
	
	nDoc = Data.nDoc
	nUniqueTokens = max(Data.word_id) + 1
	
	# Update token / topic distributions (Lik)
	
	K = resp.shape[1]
	#Lik_update = None
	
	Lik = np.zeros((K,V))
	for k in xrange(K):
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
				tokens.extend( [wid_d[i] for j in range(int(wc_d[i]))] )
			
			for i,v in enumerate(tokens):
				Lik[k,v] += resp_d[i,k]
				
			resp_idx_start += N_d

	# Normalize Lik
	Lsum = Lik.sum(axis=1)
	Lik_update = Lik / Lsum[:,np.newaxis]
	'''
	Lsum = np.sum(Lik,axis=1)
	Lik_update = np.zeros(Lik.shape)
	for i in range(Lik.shape[0]):
		Lik_update[i] = Lik[i] / Lsum[i]
	'''
	#Lik_update = Lik / Lsum
	#Lik_update = Lik_update.transpose()
	
	
	# Update eta
	EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K
	resp_idx_start = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp[resp_idx_start:resp_idx_start+N_d,:]

		R = np.sum(resp_d,axis=0)

		EX[d] = (1 / float(N_d)) * R
		
		resp_idx_start += N_d

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
		
	EXTXinv = inv(EXTX)  

	eta_update = np.dot(EXTXinv,EX.transpose())
	eta_update = np.dot(eta_update, response)

	return eta_update, Lik_update
	
'''	
	
def global_step_compact(Data,resp,theta,response,eta,Lik,delta=1.0,alpha=1.0):
	

	nDoc = Data.nDoc
	nUniqueTokens = max(Data.word_id) + 1
	# Update token / topic distributions
	Lik_update = None
	K = resp.shape[1]
	for k in xrange(K):
		for d in xrange(nDoc):
			start = Data.doc_range[d] 
			stop = Data.doc_range[d+1] 
					
			resp_d = resp[start:stop,:]
			wid_d = Data.word_id[start:stop]
			wc_d = Data.word_count[start:stop]
			#N_d = int(sum(wc_d))
			#N_resp = resp_d.shape[0]
			
			for i,v in enumerate(wid_d):
				Lik[:,v] += wc_d[i] * resp_d[i]

	# Normalize
	Lsum = np.sum(Lik,axis=1)
	Lik_update = Lik.transpose() / Lsum
	Lik_update = Lik_update.transpose()
	
	# Update eta
	EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp[start:stop,:]
		N_resp = resp_d.shape[0]

		R = np.zeros(K)
		for n in xrange(N_resp):
			R += resp_d[n] * wc_d[n]

		EX[d] = (1 / float(N_d)) * R


	EXTX = np.zeros((K,K)) #E[X^T X], KxK
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp[start:stop,:]
		
		N_resp = resp_d.shape[0]

		R = np.zeros(K) # \sum_{n=1}^{N_d} resp_d[n,:]
		for n in xrange(N_resp):
			R += resp_d[n,:] * wc_d[n]

		EXTX += (1/ float(N_d * N_d)) * np.outer(R,R.transpose())

	EXTXinv = inv(EXTX)  

	eta_update = np.dot(EXTXinv,EX.transpose())
	eta_update = np.dot(eta_update, response)

	return eta_update, Lik_update
'''
'''
def calc_elbo(Data,resp,theta,eta,K,Lik,response,delta=1.0,alpha=1.0):
	nDoc = Data.nDoc

	digammaSumTheta = digamma(theta.sum(axis=1))
	ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]

	elbo = 0
	for d in xrange(nDoc):
		elbo_d = calc_elbo_singledoc(d,Data,resp,theta,eta,K,ElogPi,Lik,response,delta=1.0,alpha=1.0)
		elbo += elbo_d

	return elbo
		
	
def calc_elbo_singledoc(d,Data,resp,theta,eta,K,ElogPi,Lik,response,delta=1.0,alpha=1.0):
	start = Data.doc_range[d] 
	stop = Data.doc_range[d+1] 

	resp_d = resp[start:stop,:]
	wid_d = Data.word_id[start:stop]
	wc_d = Data.word_count[start:stop]
	N_d = int(sum(wc_d))
	N_resp = resp_d.shape[0]

	Lik_d = Lik[:,wid_d]
	theta_d = theta[d]
	response_d = response[d]

	# E[ log p(\theta | \alpha)] = 
	t1 = c_Func(alpha, K)

	# E[ log p(z) ] = 
	t2 = 0
	#c_Func_theta_d = c_Func(theta_d)
	for n in xrange(resp_d.shape[0]):
		for k in xrange(K):
			t2 += resp_d[n,k] * ElogPi[d,k]
		
	# E[ log p(w) ] = 
	t3 = sum(sum(Lik_d))
	
	# E[ log p(y) ] = 
	
	R = np.zeros(K) # \sum_{n=1}^{N_d} resp_d[n,:]
	for n in xrange(N_resp):
		R += resp_d[n,:] * wc_d[n]
		
	sterm = np.outer(R, R)
	sterm = np.dot(eta,sterm)
	sterm = np.dot(sterm,eta)
			
	t4 = np.log( 1 / np.sqrt(2 * math.pi * delta))
	t4 -= (response_d / (2 * delta))
	t4 += (response_d / (delta * N_d)) * np.dot(eta, np.sum(resp_d,axis=0))
	t4 -= (1 / (2 * delta * N_d * N_d)) * sterm
	
	# E[ log q(z ) ] = 
	t5 = 0
	for n in xrange(N_resp):
		for k in xrange(K):
			t5 += resp_d[n,k] * np.log(resp_d[n,k])
			
	# E[ log q(pi) ] 
	t6 = ElogPi[d][k] #c_Func(theta)
	
	return t1 + t2 + t3 + t4 -t5 - t6
'''	

def c_Func(avec, K=0):
    ''' Evaluate cumulant function of the Dirichlet distribution

    Returns
    -------
    c : scalar real
    '''
    if isinstance(avec, float) or avec.ndim == 0:
        assert K > 0
        avec = avec * np.ones(K)
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    elif avec.ndim == 1:
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    else:
        return np.sum(gammaln(np.sum(avec, axis=1))) - np.sum(gammaln(avec))

	
	
'''
def local_step_all_doc(Data,K,eta,delta=1.0,alpha=1.0):
	
	nDoc = Data.nDoc
	
	for d in xrange(nDoc):
		start = Data.doc_range[d] 
		stop = Data.doc_range[d+1] 

		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))

		response_d = Data.response[d]
		theta_d = theta[d]
		resp_d = resp[resp_doc_idx:resp_doc_idx + N_d,:]

		#resp_d_update = resp_d
	
		tokens = []
		for i in range(len(wid_d)):
			tokens.extend( [wid_d[i] for j in range(int(wc_d[i]))] )
		
		
		resp_d_update, theta_d_update = SLDA_VB.local_step_single_doc(resp_d,theta_d,response_d,N_d,tokens,K,eta,delta,Lik,alpha=1.0)
		resp[resp_doc_idx:resp_doc_idx + N_d,:] = resp_d_update
		theta[d] = theta_d_update
		
	return resp
'''
			

'''

def SLDA_VB(Data,K,alpha=1.0):

	# Initialize parameters
	
	nDoc = Data.nDoc
	
	# Likelihood (beta)
	nUniqueTokens = max(Data.word_id) + 1
	
	#TODO INITIALIZE LIKELIHOOD
	Lik = np.zeros((K,nUniqueTokens))
                                             
                                             
	resp = []
	theta = np.zeros((nDoc,K))
	
	for d in xrange(nDoc):
		
		start = Data.doc_range[d] 
		stop = Data.doc_range[d+1] 
		wc_d = Data.word_count[start:stop]
		N_d = sum(wc_d) # Number of tokens in doc d
		
		# Initialize resp[d,n,k] = 1 / K
		resp_d = np.ones((N_d,K)) * (1 / float(K))
		if d == 0:
			resp = resp_d
		else:
			resp = np.concatenate((resp,resp_d))
		#resp[start:stop,:] = resp_d
		
		# Initialize theta[d,k] = alpha + N_d / K
		#alphaK = alpha * np.ones((1,K))
		theta[d] = np.ones((1,K)) * (alpha + float(N_d) / float(K))
	


	
										 
	#Lik = topics									 
	###  E step  ###
	resp_doc_idx = 0
	for d in xrange(nDoc):
		print d
		start = Data.doc_range[d] 
		stop = Data.doc_range[d+1] 
	
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
	
		response_d = Data.response[d]
		theta_d = theta[d]
		resp_d = resp[resp_doc_idx:resp_doc_idx + N_d,:]

		resp_d_update = resp_d
		
		tokens = []
		for i in range(len(wid_d)):
			tokens.extend( [wid_d[i] for j in range(int(wc_d[i]))] )
	
		niter = 0
		while niter < 20: # 100 iterations for now....
			# Just LDA update for now...
			# Expand wid and wc
	
			
			
			t1 = digamma(theta_d)
			t2 = (float(response_d) / (N_d * delta)) * eta
			t3 = (1 / float(2 * N_d * N_d * delta)) * eta * eta
		
			for n in xrange(N_d):
				wid = tokens[n]
				resp_d_without_n = np.sum(resp_d, axis=0) - resp_d[n]
				
				curTerm = np.outer(eta,resp_d_without_n)
				curTerm = np.dot(curTerm,eta)
				curTerm *= 1 / float(N_d * N_d * delta)
			
				resp_d_update_n = Lik[:,wid] * np.exp(t1 + t2 - t3 - curTerm)
				resp_d[n,:] = normalize(resp_d_update_n,norm='l1')
				#resp_d[n,:] = Lik[:,wid] * np.exp(t1 + t2 - t3 - curTerm)
			
			
			#resp_d_update = normalize(resp_d, norm='l1', axis=1)
		
			resp_d = resp_d_update		
			theta[d] = alpha + np.sum(resp_d, axis=0)
		
			resp[resp_doc_idx:resp_doc_idx + N_d,:] = resp_d	
		
			niter += 1
		
		resp_doc_idx = resp_doc_idx + N_d
		

		
	###  M Step  ###
	
	# Update likelihood / beta
        
	for k in xrange(K):
		print k
		for v in xrange(nUniqueTokens):
			for d in xrange(nDoc):
				start = Data.doc_range[d] 
				stop = Data.doc_range[d+1] 
				N_d = stop - start
				
				resp_d = resp[start:stop,:]
				wid_d = Data.word_id[start:stop]
				wc_d = Data.word_count[start:stop]
				
				tokens = []
				for i in range(len(wid_d)):
					tokens.extend( [wid_d[i] for j in range(int(wc_d[i]))] )
				
				
				if v in tokens:
					n = np.where(wid_d==v)[0][0]
					Lik[k,v] += wc_d[n] 
	
	Lik = normalize(Lik, norm='l1', axis=1)


	# Update eta
	response = Data.response

	EX = np.zeros((nDoc,K)) # E[X], X[d] = \bar{Z}_d, D X K
	for d in xrange(nDoc):
		starti = Data.doc_range[d]
		endi = Data.doc_range[d+1]
		resp_d = resp[starti:endi,:]
		N_d, _ = resp_d.shape
		EX[d] = (1 / float(N_d)) * np.sum(resp_d,axis=0)


	EXTX = np.zeros((K,K)) #E[X^T X], KxK
	for d in xrange(nDoc):
		starti = Data.doc_range[d]
		endi = Data.doc_range[d+1]
		resp_d = resp[starti:endi,:]
		N_d, _ = resp_d.shape

		t1 = np.zeros(K)
		for n in xrange(N_d):
			t1+= resp_d[n]
		
		EXTX += (1/ float(N_d * N_d)) * np.outer(t1,t1.transpose())


    EXTXinv = inv(EXTX)  
    
    eta_update = np.dot(EXTXinv,EX.transpose())
    eta_update = np.dot(eta_update, response)

	
'''			
