'''
SequentialViz.py

Vizualizing sequential data

Usage (command-line)
-------
python -m bnpy.viz.SeqViz dataName aModelName obsModelName algName [kwargs]
'''


import numpy as np

#import sys
#import imp
from munkres import Munkres
import imp
#munkres = imp.load_source('munkres', '$BNPYthird-party/munkres.py')
from bnpy.viz import PlotELBO

def main():
    args = PlotELBO.parse_args()
    for (jobpath, jobname, color) in PlotELBO.job_to_plot_generator(args):
        print jobpath
        print jobname
        print color

def calcOptPermutation(estZ, trueZ):
    '''
    Takes in estimated Z and true Z values.  Be sure that both contain the same
    values (i.e. both are or are not zero-indexed)

    Returns a length(Z) x 2 array that contains the least costly permutation of
    the estZ values to match the trueZ values, as well as the total cost
    required to do so.
    '''

    
    estK = int(np.max(estZ)) + 1
    trueK = int(np.max(trueZ)) + 1
    c = np.zeros((estK, trueK))

    for k in xrange(estK):
        for l in xrange(trueK):
            for t in xrange(len(estZ)):
                if estZ[t] == k and trueZ[t] != l:
                    c[k,l] += 1
    m = Munkres()
    cCopy = c.copy() #current munkres implementation modifies c
    inds = m.compute(c)

    cost = 0
    for row, col in inds:
        cost += cCopy[row,col]

    return inds, cost


#TODO : custom implementation of munkres?  Currently a third party library
#def munkres():


if __name__ == '__main__':
    main()
