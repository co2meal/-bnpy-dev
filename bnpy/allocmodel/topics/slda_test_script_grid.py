#!/contrib/projects/EnthoughtPython/epd64/bin/python

import bnpy
from bnpy.allocmodel.topics import SLDA_VB_edit
from bnpy.allocmodel.topics import slda_helper

## Make SLDA toy bars data set
import scipy
import numpy as np
from bnpy.data import WordsData_slda
import Bars2D

import sys

Niters = int(sys.argv[1])
print Niters

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 8  # Number of topics
V = 100  # Vocabulary Size
gamma = 0.5  # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 500 #500
Defaults['nWordsPerDoc'] = 200 #200

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
#eta_init = eta_true + np.random.normal(-10,10,eta_true.shape[0])
eta_init = np.random.rand(eta_true.shape[0])
Lik_init_ = np.random.rand(K,V)
Lsum = np.sum(Lik_init_,axis=1)
Lik_init = np.zeros(Lik_init_.shape)
for i in range(Lik_init.shape[0]):
	Lik_init[i] = Lik_init_[i] / Lsum[i]
	
#Lsum = np.sum(Lik_init,axis=1)
#Lik_init = Lik_init.transpose() / Lsum
#Lik_init = Lik_init.transpose()

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


# Alternate Local and Global step

resp = resp_init_extended.copy()
theta = theta_init.copy()

eta = eta_init.copy()
#Lik = Lik_init.copy()
Lik = Lik_true.copy()

print 'Doing VB...'
#SLDA_VB_edit is faster
resp_vb,theta_vb,eta_vb,Lik_vb, elbos = SLDA_VB_edit.SLDA_VariationalBayes(Data,resp,theta,eta,Lik,delta=1.0,alpha=1.0,Niters=Niters)

scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_trueLik_INITS_'+str(Niters)+'=iters_nwpd=200_june6.mat', {'resp':resp, 'eta': eta, 'theta':theta, 'Lik':Lik})
scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_trueLik_VB_'+str(Niters)+'=iters_nwpd=200_jun6.mat', {'resp':resp_vb, 'eta': eta_vb, 'theta':theta_vb, 'Lik':Lik_vb, 'elbos': elbos})

#scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_INITS_niters='+str(Niters)+'_e.mat', {'resp':resp, 'eta': eta, 'theta':theta, 'Lik':Lik})
#scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_VB_niters='+str(Niters)+'_e.mat', {'resp':resp_vb, 'eta': eta_vb, 'theta':theta_vb, 'Lik':Lik_vb, 'elbos': elbos})
try:
	scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_DEFAULTS_niters='+str(Niters)+'_nwpd=200_june6.mat', Defaults)
except:
	pass
	

'''

resp = resp_init_extended.copy()
theta = theta_init.copy()

eta = eta_init.copy()
Lik = Lik_init.copy()
#Lik = Lik_true.copy()

resp_vb,theta_vb,eta_vb,Lik_vb = SLDA_VB.SLDA_VariationalBayes(Data,resp,theta,eta,Lik,delta=1.0,alpha=1.0,Niters=40)

scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_trueLik.mat', {'resp':resp_vb, 'eta': eta_vb, 'theta':theta_vb, 'Lik':Lik_vb})

#scipy.io.savemat('/home/lweiner/bnpy_repo/bnpy-dev/out/slda_updates_trueLik.mat', {'resp':resp_vb, 'eta': eta_vb, 'theta':theta_vb, 'Lik':Lik_vb})

'''



