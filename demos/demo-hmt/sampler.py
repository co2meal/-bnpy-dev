import bnpy
import numpy as np
from collections import deque
from scipy.io import savemat
from pylab import *
from matplotlib.colors import LogNorm

tmodel, Info = bnpy.run('pepperHMT', 'FiniteHMT', 'ZeroMeanGauss', 'VB', jobname='pepper-demo-vb-5', K=5, nLap=1000, convergeThr=0.001, doWriteStdOut=True)

Sigma = tmodel.obsModel.Post.B / (tmodel.obsModel.Post.nu - tmodel.obsModel.Post.D - 1)[:,np.newaxis, np.newaxis]
mu = np.zeros([8])
pi_init = np.random.dirichlet(tmodel.allocModel.initTheta)
k = pi_init.shape[0]
pi_trans = np.empty([4, k, k])
for b in xrange(4):
    for i in xrange(k):
        pi_trans[b, i, :] = np.random.dirichlet(tmodel.allocModel.transTheta[b,i,:])

datass_graph = np.empty([8, 8400])
q = deque()
for i in xrange(400):
    init_state = np.argmax(np.random.multinomial(1, pi_init))
    init_obs = np.random.multivariate_normal(mu, Sigma[init_state,:,:])
    datass_graph[:, i] = init_obs
    q.append((init_state, init_obs))
idx = 400
for i in xrange(2000):
    curr = q.popleft()
    for br in xrange(4):
        ch = np.argmax(np.random.multinomial(1, pi_trans[br,curr[0],:]))
        obs = np.random.multivariate_normal(mu, Sigma[ch,:,:])
        datass_graph[:, idx] = obs
        idx += 1
        q.append((ch, obs))

# Save datass_graph to be used for Scale, Position (Far), and Position (Near) graphs
savemat('~/Desktop/generated.mat', {'datass_graph': datass_graph})

# Generate graph for Orientation

Sigma = tmodel.obsModel.Post.B / (tmodel.obsModel.Post.nu - tmodel.obsModel.Post.D - 1)[:,np.newaxis, np.newaxis]
mu = np.zeros([8])
pi_init = np.random.dirichlet(tmodel.allocModel.initTheta)
k = pi_init.shape[0]
pi_trans = np.empty([4, k, k])
D = np.empty([3200, 2])
for b in xrange(4):
    for i in xrange(k):
        pi_trans[b, i, :] = np.random.dirichlet(tmodel.allocModel.transTheta[b,i,:])
idx = 0
for i in xrange(400):
    init_state = np.argmax(np.random.multinomial(1, pi_init))
    q = deque()
    init_obs = np.random.multivariate_normal(mu, Sigma[init_state,:,:])
    q.append((init_state, init_obs))
    #D[idx, :] = init_obs[0:1]
    #D[idx+1, 0] = init_obs[1]
    #D[idx+1, 1] = init_obs[0]
    #idx += 2
    for j in xrange(1):
        curr = q.popleft()
        for br in xrange(4):
            ch = np.argmax(np.random.multinomial(1, pi_trans[br,curr[0],:]))
            obs = np.random.multivariate_normal(mu, Sigma[ch,:,:])
            D[idx, :] = obs[0:1]
            D[idx+1, 0] = obs[1]
            D[idx+1, 1] = obs[0]
            idx += 2
            q.append((ch, obs))

# Display Orientation Graph

hist2d(D[:,0], D[:,1], bins=128, norm=LogNorm())
show()