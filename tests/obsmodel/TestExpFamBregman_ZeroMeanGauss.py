import numpy as np

from scipy.special import gammaln, digamma
from bnpy.viz import PlotUtil

pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(
	pylab, 
	**{'figure.subplot.left':0.23,
	   'figure.subplot.bottom':0.23})

def make_phi_grid(ngrid=1000, min_phi=-20, max_phi=-1e-10):
	phi_grid = np.linspace(min_phi, max_phi, ngrid)
	return phi_grid

def make_mu_grid(ngrid=1000, min_mu=1e-10, max_mu=20):
	mu_grid = np.linspace(min_mu, max_mu, ngrid)
	return mu_grid

def phi2mu(phi_grid):
	mu_grid = -0.5 * 1.0/phi_grid
	return mu_grid

def mu2phi(mu_grid):
	phi_grid = -0.5 * 1.0/mu_grid
	return phi_grid

def c_Phi(phi_grid):
	c_grid = - 0.5 * np.log(- phi_grid) # - 0.5 * np.log(2)
	return c_grid

def c_Mu(mu_grid):
	c_grid = - 0.5 * np.log(mu_grid)
	return c_grid

def bregmanDiv(mu_grid, mu):
	'''

	Returns
	-------
	div : 1D array, size ngrid
	'''
	div = c_Mu(mu_grid) - c_Mu(mu) - (mu_grid - mu) * mu2phi(mu)
	return div

def makePlot_pdf_Phi(
		nu=0, tau=0, phi_grid=None,
		ngrid=1000, min_phi=-100, max_phi=100):
	label = 'nu=%7.2f' % (nu)
	cPrior = - gammaln(nu) + gammaln(nu-tau) + gammaln(tau)
	if phi_grid is None:
		phi_grid = np.linspace(min_phi, max_phi, ngrid)
	logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
	pdf_grid = np.exp(logpdf_grid)
	IntegralVal = np.trapz(pdf_grid, phi_grid)
	mu_grid = phi2mu(phi_grid)
	ExpectedPhiVal = np.trapz(pdf_grid * phi_grid, phi_grid)
	ExpectedMuVal = np.trapz(pdf_grid * mu_grid, phi_grid)
	print '%s Integral=%.4f E[phi]=%6.3f E[mu]=%.4f' % (
		label, IntegralVal, ExpectedPhiVal, ExpectedMuVal)
	pylab.plot(phi_grid, pdf_grid, '-', label=label)
	pylab.xlabel('phi (log odds ratio)')
	pylab.ylabel('density p(phi)')

def makePlot_pdf_Mu(
		nu=0, tau=0, phi_grid=None,
		ngrid=1000, min_phi=-100, max_phi=100):
	label = 'nu=%7.2f' % (nu,)
	cPrior = - gammaln(nu) + gammaln(nu-tau) + gammaln(tau)

	mu_grid = np.linspace(1e-15, 1-1e-15, ngrid)
	phi_grid = mu2phi(mu_grid)
	logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
	logJacobian_grid = -1.0 * np.log(mu_grid) - np.log(1-mu_grid)
	pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
	IntegralVal = np.trapz(pdf_grid, mu_grid)
	ExpectedMuVal = np.trapz(pdf_grid * mu_grid, mu_grid)
	ExpectedPhiVal = np.trapz(pdf_grid * phi_grid, mu_grid)
	print '%s Integral=%.4f E[phi]=%6.3f E[mu]=%.4f' % (
		label, IntegralVal, ExpectedPhiVal, ExpectedMuVal)
	pylab.plot(mu_grid, pdf_grid, '-', label=label)
	pylab.xlabel('mu')
	pylab.ylabel('density p(mu)')


def makePlot_cumulant_Phi(
		phi_grid=None, **kwargs):
	if phi_grid is None:
		phi_grid = make_phi_grid(**kwargs)
	c_grid = c_Phi(phi_grid)
	pylab.plot(phi_grid, c_grid, 'k-')
	pylab.xlabel('phi')
	pylab.ylabel('c(phi)')

def makePlot_cumulant_Mu(
		mu_grid=None, **kwargs):
	if mu_grid is None:
		mu_grid = make_mu_grid(**kwargs)
	c_grid = c_Mu(mu_grid)
	pylab.plot(mu_grid, c_grid, 'r-')
	pylab.xlabel('mu')
	pylab.ylabel('gamma(mu)')


def makePlot_bregmanDiv_Mu(
		mu_grid=None, mu=0.5, **kwargs):
	label = 'mu=%.2f' % (mu)
	if mu_grid is None:
		mu_grid = make_mu_grid(**kwargs)
	div_grid = bregmanDiv(mu_grid, mu)
	pylab.plot(mu_grid, div_grid, '--', linewidth=2, label=label)
	pylab.xlabel('x')
	pylab.ylabel('Bregman div.  D(x, mu)')

if __name__ == '__main__':
	pylab.figure()
	makePlot_cumulant_Phi(ngrid=1000)
	pylab.xlim([-20, 0.5])
	pylab.savefig('ZMG_cumulantPhi.eps')

	pylab.figure()
	makePlot_cumulant_Mu(ngrid=1000)
	pylab.xlim([-0.5, 20])
	pylab.savefig('ZMG_cumulantMu.eps')

	
	pylab.figure()
	for mu in [0.1, 1, 5]:
		makePlot_bregmanDiv_Mu(mu=mu, ngrid=5000)
	pylab.legend(loc='upper right', fontsize=13)
	pylab.ylim([-0.05, 3])
	pylab.xlim([-0.5, 10])
	pylab.savefig('ZMG_bregmanDiv.eps')

	'''
	mu_Phi = 0.7
	print "Mode: ", mu2phi(mu_Phi)
	nuRange = [1/2.0, 1, 2.0, 8, 32, 128]

	pylab.figure()
	for nu in nuRange[::-1]:
		tau = mu_Phi * nu
		makePlot_pdf_Phi(nu=nu, tau=tau, 
			ngrid=500000)
	pylab.legend(loc='upper left', fontsize=13)
	pylab.ylim([0, 2.0]); pylab.yticks([0, 1, 2])
	pylab.xlim([-10, 6])
	pylab.savefig('BetaBern_densityPhi.eps')

	pylab.figure()
	for nu in nuRange[::-1]:
		tau = mu_Phi * nu
		makePlot_pdf_Mu(nu=nu, tau=tau, 
			ngrid=500000)
	pylab.legend(loc='upper left', fontsize=13)
	pylab.ylim([0, 10])
	pylab.savefig('BetaBern_densityMu.eps')
	'''

	pylab.show()