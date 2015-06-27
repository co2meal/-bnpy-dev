from pylab import *
from matplotlib.colors import LogNorm
from scipy.io import loadmat

res = loadmat('~/Desktop/gen_scale128.mat')
D = res['D']
hist2d(D[:,0], D[:,1], bins=128, norm=LogNorm())
show()