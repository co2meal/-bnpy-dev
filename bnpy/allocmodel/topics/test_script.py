import bnpy
from bnpy.allocmodel.topics import SLDA_VB
from bnpy.allocmodel.topics import slda_helper

## Make SLDA toy bars data set

import numpy as np
from bnpy.data import WordsData_slda
import Bars2D

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 8  # Number of topics
V = 100  # Vocabulary Size
gamma = 0.5  # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 500
Defaults['nWordsPerDoc'] = 500 #200

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'], Defaults['eta'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG,slda=True)

kwargs = dict()
for key in Defaults:
	if key not in kwargs:
		kwargs[key] = Defaults[key]
		
kwargs['delta'] = 1.0

Data = WordsData_slda.WordsData.CreateToyDataFromSLDAModel(**kwargs)


## Test SLDA inference
response = Data.response

eta_true = Data.TrueParams['eta']
resp_true = Data.TrueParams['resp']
theta_true = Data.TrueParams['Pi'] *  Defaults['nWordsPerDoc']

topics = Data.TrueParams['topics']
topic_prior = Data.TrueParams['topic_prior']

Lik_true = topics

alpha = 1.0
delta = 1.0

nDoc = Data.nDoc


# Get random initial values for parameters
eta_init = eta_true + np.random.normal(-10,10,eta_true.shape[0])
Lik_init = np.random.rand(K,V)

theta_init = np.zeros((nDoc,K))
resp_init = np.zeros(resp_true.shape)

for d in xrange(nDoc):
	start = Data.doc_range[d] 
	stop = Data.doc_range[d+1] 
	wc_d = Data.word_count[start:stop]
	N_d = int(sum(wc_d))
	N_resp = stop - start
	resp_d = np.ones((N_resp,K)) * (1/float(K))
	resp_init[start:stop,:] = resp_d
	theta_init[d] = alpha * np.ones(K) + (float(N_d) / float(K))


#resp_init_copy = resp_init.copy()
#theta_init_copy = theta_init.copy()

resp_init_extended = slda_helper.expand_resp(Data,resp_init)
resp_true_extended = slda_helper.expand_resp(Data,resp_true)

# Test Local Step on extended resp

# Give true parameters, ensure resp doesn't wander too far
resp = resp_true_extended.copy()
theta = theta_true.copy()
eta = eta_true.copy()
Lik = Lik_true.copy()

resp_update = np.zeros(resp.shape)

# Update local parameters, resp and theta
resp_start_idx = 0
for d in xrange(nDoc):
	print d
	start = Data.doc_range[d] 
	stop = Data.doc_range[d+1] 

	wid_d = Data.word_id[start:stop]
	wc_d = Data.word_count[start:stop]
	N_d = int(sum(wc_d))

	response_d = Data.response[d]
	theta_d = theta[d]
	resp_d = resp[resp_start_idx:resp_start_idx + N_d].copy()

	# Local updates
	resp_d_update, theta_d_update = SLDA_VB.local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,delta,alpha)
	resp_update[resp_start_idx:resp_start_idx + N_d,:] = resp_d_update

	#theta[d] = theta_d_update
	
	resp_start_idx += N_d

# Analyze the difference between true resp and inferred resp
n = 0
for i in xrange(500): #xrange(resp.shape[0]):
	diff = max(abs(resp_update[i] - resp_true_extended[i]))
	if diff > 0.01:
		n += 1
		print i, diff


# Test Global Step on extended resp
# Given true resp, theta, try to converge on true eta
resp = resp_true_extended.copy()
theta = theta_true.copy()
Lik = Lik_true.copy()

eta = eta_init.copy()

eta_update, Lik_update = SLDA_VB_edit.global_step_extended(Data,resp,theta,response,eta,V,alpha,delta)

diff = max(abs(eta_update - eta_true))

n = 0
for i in xrange(Lik_update.shape[0]):
	diff = max(abs(Lik_update[0] - Lik_true[0]))
	if diff > 0.01:
		n += 1
		print i, diff


# Alternate Local and Global step

resp = resp_init_extended.copy()
theta = theta_init.copy()

eta = eta_init.copy()
#Lik = Lik_init.copy()
Lik = Lik_true.copy()

resp_vb,theta_vb,eta_vb,Lik_vb = SLDA_VB_edit.SLDA_VariationalBayes(Data,resp,theta,eta,Lik,delta=1.0,alpha=1.0,Niters=4)

n = 0
for i in xrange(resp.shape[0]):
	diff = max(abs(resp_vb[i] - resp_true_extended[i]))
	if diff > 0.1:
		n += 1
		print i, diff



####
import scipy
Rinit1 = scipy.io.loadmat('/Users/leah/bnpy_repo/bnpy-dev/out/slda_updates_INITS_niters=100.mat')
Rvb1 = scipy.io.loadmat('/Users/leah/bnpy_repo/bnpy-dev/out/slda_updates_VB_niters=100.mat')

resp_init1 = Rinit1['resp']
eta_init1 = Rinit1['eta']
Lik_init1 = Rinit1['Lik']

resp_vb1 = Rvb1['resp']
eta_vb1 = Rvb1['eta']
Lik_vb1 = Rvb1['Lik']
elbos_vb1 = Rvb1['elbos']

print abs(eta_vb1 - eta_true)

n = 0
for i in xrange(resp_vb1.shape[0]):
	diff = max(abs(resp_vb1[i] - resp_true_extended[i]))
	if diff > 0.5:
		n += 1
		print i, diff


R2 = scipy.io.loadmat('/Users/leah/bnpy_repo/bnpy-dev/out/slda_updates_50iters.mat')

resp_vb2 = R2['resp']
eta_vb2 = R2['eta']
theta_vb2 = R2['theta']
Lik_vb2 = R2['Lik']

print abs(eta_vb2 - eta_true)

n = 0
for i in xrange(resp.shape[0]):
	diff = max(abs(resp_vb2[i] - resp_true_extended[i]))
	if diff > 0.05:
		n += 1
		print i, diff





