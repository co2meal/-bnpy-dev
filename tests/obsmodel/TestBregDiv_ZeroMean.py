import numpy as np
from bnpy.viz.PlotUtil import pylab

def calcBregDiv_ZeroMean(x, mu1):
    ''' Calculate breg divergence between data and mean parameter

    Returns
    -------
    div : ND array, same size as x
    '''
    xsq = np.square(x, dtype=np.float64)
    div = -0.5 - 0.5 * np.log(xsq+1e-10) + 0.5 * np.log(mu1) + 0.5 * (xsq)/mu1
    return div

def makePlot(muVals=[0.01, 0.1, 1, 10]):
    xgrid = np.linspace(0, 8, 2000)
    pylab.hold('on')
    for mu in muVals:
        ygrid = calcBregDiv_ZeroMean(xgrid, mu)
        pylab.plot(xgrid, ygrid, label='\mu=%6.2f' % (mu))
    pylab.legend(loc='upper right')
    pylab.xlim([-0.1, xgrid.max()])
    pylab.ylim([-0.1, xgrid.max()])
    pylab.xlabel('x')
    pylab.ylabel('D(x, \mu)')
    pylab.show(block=True)    

if __name__ == "__main__":
    makePlot()
