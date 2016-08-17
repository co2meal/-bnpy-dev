#!/contrib/projects/EnthoughtPython/epd64/bin/python

import bnpy
from bnpy.viz import BarsViz
from bnpy.allocmodel.topics import SLDA_VB_edit
from bnpy.allocmodel.topics import slda_helper

## Make SLDA toy bars data set

import numpy as np
import os
from bnpy.data import WordsData_slda
from bnpy.data import WordsData
import Bars2D

import sys

try:
	fkey = sys.argv[1]
except:
	fkey = ''
	
SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 8  # Number of topics
V = 100  # Vocabulary Size
#gamma = 0.5  # hyperparameter over doc-topic distribution
gamma = 25.0

Defaults = dict()
Defaults['nDocTotal'] = 50
Defaults['nWordsPerDoc'] = 400

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

bnpyoutdir = os.environ['BNPYOUTDIR']
jobname = 'slda_test_%s' % fkey

if not os.path.exists(os.path.join(bnpyoutdir,jobname)):
	os.mkdir(os.path.join(bnpyoutdir,jobname))

f = open(os.path.join(bnpyoutdir,jobname,'test_script_out.txt'),'w')


Data = WordsData_slda.WordsData.CreateToyDataFromSLDAModel(**kwargs)

## Viz true topics
#h = BarsViz.showTopicsAsSquareImages(Defaults['topics'])
#h.show()

## Test SLDA inference
response = Data.response
eta_true = Data.TrueParams['eta']
resp_true = Data.TrueParams['resp']
theta_true = Data.TrueParams['Pi'] *  Defaults['nWordsPerDoc']

topics = Data.TrueParams['topics']
topic_prior = Data.TrueParams['topic_prior']

Lik_true = topics

alpha = 1.0
delta_init = 1.0

nDoc = Data.nDoc

# Get initial values for parameters by doing LDA inference

word_id = Data.word_id
word_count = Data.word_count
doc_range = Data.doc_range

ldaData = WordsData(word_id, word_count, doc_range, V)
ldaData.name = 'slda_to_lda_test_%s' % key
R, hmodel = bnpy.run(ldaData,'FiniteTopicModel','Mult','VB',K=K,nLap=300)

# Grab init values according to LDA output
LP = hmodel['LP']
resp_init = LP['resp']
theta_init = LP['theta']

resp_init_extended,_ = slda_helper.expand_resp(Data,resp_init)
resp_true_extended, tokens_true = slda_helper.expand_resp(Data,resp_true)

eta_init = np.random.rand(eta_true.shape[0])

# Calculate Lik init based on resp init
Lik_init = np.zeros((K,V))

for k in xrange(K):
	resp_start_idx = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d] 
		stop = Data.doc_range[d+1] 
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp_init_extended[resp_start_idx:resp_start_idx + N_d,:]

		tokens = tokens_true[resp_start_idx:resp_start_idx+N_d]
		
		for v in range(V):
			for i,n in enumerate(tokens):
				if v == n:
					Lik_init[k,v] += resp_d[i,k]

		resp_start_idx += N_d

# Normalize Lik
Lsum = Lik_init.sum(axis=1)
Lik_init = Lik_init / Lsum[:,np.newaxis]

## Get true doc / topic counts
resp = resp_true_extended.copy()
DTC_true = np.zeros((nDoc,resp.shape[1]))

resp_start_idx = 0
for d in xrange(nDoc):
	start = Data.doc_range[d] 
	stop = Data.doc_range[d+1] 
	N_d = int(sum(wc_d))
	DTC_true[d,:] = np.sum(resp[resp_start_idx:resp_start_idx + N_d],axis=0)
	resp_start_idx += N_d



# Get "true" topic / token counts based off true resp
Lik_from_resp_true = np.zeros((K,V))

for k in xrange(K):
	resp_idx_start = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d] 
		stop = Data.doc_range[d+1] 
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp_true_extended[resp_idx_start:resp_idx_start + N_d,:]

		tokens_d = tokens_true[resp_idx_start:resp_idx_start + N_d]

		for v in range(V):
			for i,n in enumerate(tokens_d):
				if v == n:
					Lik_from_resp_true[k,v] += resp_d[i,k]

		resp_idx_start += N_d

# Normalize Lik
Lsum = Lik_from_resp_true.sum(axis=1)
Lik_from_resp_true = Lik_from_resp_true / Lsum[:,np.newaxis]

# h = BarsViz.showTopicsAsSquareImages(Lik_update)
#h.show()


# Test Local Step 
f.write('TESTING LOCAL STEP....\n\n')

# Given true parameters, ensure resp doesn't wander too far
resp = resp_true_extended.copy()
theta = theta_true.copy()
eta = eta_true.copy()
Lik = Lik_true.copy()

resp_update = np.zeros(resp.shape)
theta_update = np.zeros(theta.shape)
tokens =  np.zeros(resp.shape[0])
#DTC_true = np.zeros((nDoc,resp.shape[1]))
DTC_inferred = np.zeros((nDoc,resp.shape[1]))


resp_start_idx = 0
for d in xrange(nDoc):

	start = Data.doc_range[d] 
	stop = Data.doc_range[d+1] 
	wid_d = Data.word_id[start:stop]
	wc_d = Data.word_count[start:stop]
	N_d = int(sum(wc_d))

	response_d = Data.response[d].copy()
	theta_d = theta[d].copy()
	resp_d = resp[resp_start_idx:resp_start_idx + N_d].copy()
	
	tokens_d = tokens_true[resp_start_idx:resp_start_idx+N_d]

	resp_d_update, theta_d_update = SLDA_VB_edit.local_step_single_doc(resp_d,theta_d,response_d,wid_d,wc_d,eta,Lik,tokens_d,delta_init,alpha)

	resp_update[resp_start_idx:resp_start_idx + N_d,:] = resp_d_update.copy()
	theta_update[d] = theta_d_update.copy()

	#DTC_true[d,:] = np.sum(resp[resp_start_idx:resp_start_idx + N_d],axis=0)
	DTC_inferred[d,:] = np.sum(resp_d_update,axis=0)

	resp_start_idx += N_d


for i in range(nDoc):
	print 'Doc %d:' % i
	print np.round(DTC_true[i])
	print np.round(DTC_inferred[i])
	print '--------------------------'

# Analyze doc topic counts
f.write('Doc Topic Counts (true then inferred) after local step for random sample of docs \n\n ')
rand_doc_sample = np.random.randint(0,nDoc,10)
for i in rand_doc_sample:
	f.write(', '.join(map(str,np.round(DTC_true[i]))))
	f.write('\n')
	f.write(', '.join(map(str,np.round(DTC_inferred[i]))))
	f.write('\n------------------------------------\n')


# Calculate Lik based on new resp
Lik_from_resp_update = np.zeros((K,V))

for k in xrange(K):
	resp_idx_start = 0
	for d in xrange(nDoc):
		start = Data.doc_range[d] 
		stop = Data.doc_range[d+1] 
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		N_d = int(sum(wc_d))
		resp_d = resp_update[resp_idx_start:resp_idx_start + N_d,:]

		tokens = []
		for i in range(len(wid_d)):
			tokens.extend([wid_d[i] for j in range(int(wc_d[i]))])


		for v in range(V):
			for i,n in enumerate(tokens):
				if v == n:
					Lik_from_resp_update[k,v] += resp_d[i,k]

		resp_idx_start += N_d

# Normalize Lik
Lsum = Lik_from_resp_update.sum(axis=1)
Lik_from_resp_update = Lik_from_resp_update / Lsum[:,np.newaxis]


#h = BarsViz.showTopicsAsSquareImages(Lik_from_resp_update)
#h.show()

# Test Global Step on extended resp
f.write('\n\n\n')
f.write('TESTING GLOBAL STEP....\n\n')

# Given true resp, theta, try to converge on true eta
resp = resp_true_extended.copy()
theta = theta_true.copy()

f.write('starting with init')
Lik = Lik_init.copy()
#Lik = Lik_true.copy()
eta = eta_init.copy()

#eta_update, Lik_update, delta_update = SLDA_VB_edit.global_step(Data,resp,theta,response,eta,V,alpha,delta)
eta_update, Lik_update, delta_update = SLDA_VB_edit.global_step(Data,resp,theta,response,V)

# How close is calculated Lik to true Lik (shouldn't be close)
f.write('true beta (token / topic probabilities) then updated beta for each topic \n')
for i in range(0,K):
	f.write('Topic %d \n' % i)
	f.write(', '.join(str(elem) for elem in np.argwhere(Lik_true[i] > 0.01).flatten().tolist()))
	f.write('\n')
	f.write(', '.join(str(elem) for elem in np.argwhere(Lik_update[i] > 0.01).flatten().tolist()))
	f.write('------------------------------------\n\n\n')


f.write('eta true then eta inferred (rounded) after global step \n')
f.write(', '.join(map(str,np.round(eta_true,0))))
f.write('\n')
f.write(', '.join(map(str,np.round(eta_update,0))))
f.write('\n')

# Alternate Local and Global step

f.write('\n\n\n')
f.write('TESTING GLOBAL AND LOCAL STEP.....\n')

###
#resp_init2_extended = np.ones(resp_init_extended.shape) * 1 / K

###



resp = resp_init_extended.copy()
#resp = resp_true_extended.copy()
theta = theta_init.copy()
eta = eta_init.copy()
Lik = Lik_init.copy()
#Lik = Lik_true.copy()

resp_vb,theta_vb,eta_vb,Lik_vb,delta_vb,elbos = SLDA_VB_edit.SLDA_VariationalBayes(Data,resp,theta,eta,Lik,tokens_true,delta=delta_init,alpha=1.0,Niters=50)

DTC_vb = np.zeros((nDoc,resp.shape[1]))

resp_start_idx = 0
for d in xrange(nDoc):
	start = Data.doc_range[d] 
	stop = Data.doc_range[d+1] 
	wid_d = Data.word_id[start:stop]
	wc_d = Data.word_count[start:stop]
	N_d = int(sum(wc_d))
	resp_d_vb = resp_vb[resp_start_idx:resp_start_idx + N_d].copy()
	resp_start_idx += N_d
	DTC_vb[d,:] = np.sum(resp_d_vb, axis=0)				
	

f.write('Doc Topic Counts (true then inferred) after alternating local and global step \n ')
for i in rand_doc_sample:
	f.write(', '.join(map(str,np.round(DTC_true[i]))))
	f.write('\n')
	f.write(', '.join(map(str,np.round(DTC_vb[i]))))
	f.write('\n------------------------------------\n')



f.close()

import scipy.io

scipy.io.savemat(os.path.join(bnpyoutdir,jobname,'bnpy_output.mat'),
		{'resp': resp_vb, 'theta': theta_vb, 'eta': eta_vb, 'elbos': elbos})



