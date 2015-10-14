import numpy as np

from scipy.special import gammaln, digamma
from bnpy.viz import PlotUtil

pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(
    pylab, 
    **{'figure.subplot.left':0.23,
       'figure.subplot.bottom':0.23})

def make_phi1_grid(ngrid=1000, min_phi=-20, max_phi=-1e-10):
    phi1_grid = np.linspace(min_phi, max_phi, ngrid)
    return phi1_grid

def make_phi2_grid(ngrid=1000, min_phi2=-20, max_phi2=20):
    phi2_grid = np.linspace(min_phi2, max_phi2, ngrid)
    return phi2_grid

def make_mu1_grid(ngrid=1000, min_mu=1e-10, max_mu=30):
    mu1_grid = np.linspace(min_mu, max_mu, ngrid)
    return mu1_grid

def make_mu2_grid(ngrid=1000, min_mu=-30, max_mu=30):
    mu2_grid = np.linspace(min_mu, max_mu, ngrid)
    return mu2_grid

def phi2mu(phi1_grid, phi2_grid):
    inv_phi1 = 1.0 / phi2_grid
    mu1_grid = -0.5 * inv_phi1 + \
        0.25 * np.square(inv_phi1) * np.square(phi2_grid)
    mu2_grid = -0.5 * inv_phi1 * phi2_grid
    return mu1_grid, mu2_grid

def mu2phi(mu1_grid, mu2grid):
    inv_mu = 1.0/(mu1_grid - np.square(mu2_grid))
    phi1_grid = -0.5 * inv_mu
    phi2_grid = mu2_grid * inv_mu
    return phi1_grid, phi2_grid

def c_Phi(phi1_grid, phi2_grid):
    c_grid = - 0.5 * np.log(- phi1_grid) - 0.25 * np.square(phi2_grid)/phi1_grid
    return c_grid

def c_Mu(mu1_grid, mu2_grid):
    c_grid = - 0.5 * np.log(mu1_grid - np.square(mu2_grid))
    return c_grid

def E_phi1(nu=0, tau=0):
    return - (0.5 * nu + 1.0) / tau

def bregmanDiv(mu1_grid, mu2_grid, mu1, mu2):
    '''

    Returns
    -------
    div : 1D array, size ngrid
    '''
    phi1, phi2 = mu2phi(mu)
    div = c_Mu(mu1_grid, mu2_grid) - c_Mu(mu1, mu2) - \
        (mu1_grid - mu1) * phi1 - \
        (mu2_grid - mu2) * phi2
    return div

def c_Prior(nu=0, tau=0):
    scaled_nu = 0.5 * nu + 1
    return gammaln(scaled_nu) - scaled_nu * np.log(tau)

def logJacobian_mu(mu_grid):
    '''
    Returns
    -------
    J : 1D array, size ngrid
        J[n] = log Jacobian(mu[n], mu2phi)
        Jacobian(mu, mu2phi) = d/dmu phi(mu)
    '''
    pass

def checkFacts_pdf_Phi(
        nu=0, tau=0, phi_grid=None, **kwargs):
    cPrior = c_Prior(nu=nu, tau=tau)
    if phi_grid is None:
        phi_grid = make_phi_grid(**kwargs)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    pdf_grid = np.exp(logpdf_grid)
    mu_grid = phi2mu(phi_grid)

    IntegralVal = np.trapz(pdf_grid, phi_grid)
    E_phi_numeric = np.trapz(pdf_grid * phi_grid, phi_grid)
    E_phi_formula = -(0.5 * nu + 1) / tau
    E_mu_numeric = np.trapz(pdf_grid * mu_grid, phi_grid)
    E_mu_formula = tau/nu
    mode_phi_numeric = phi_grid[np.argmax(pdf_grid)]
    mode_phi_formula = mu2phi(tau/nu)

    E_c_numeric = np.trapz(pdf_grid * c_Phi(phi_grid), phi_grid)
    E_c_formula = - 0.5 * digamma(0.5 * nu + 1) + 0.5 * np.log(tau)

    print "nu=%7.3f tau=%7.3f" % (nu, tau)
    print "     Integral=% 7.3f   should be % 7.3f" % (IntegralVal, 1.0)
    print "        E[mu]=% 7.3f   should be % 7.3f" % (E_mu_numeric, E_mu_formula)
    print "       E[phi]=% 7.3f   should be % 7.3f" % (E_phi_numeric, E_phi_formula)
    print "    E[c(phi)]=% 7.3f   should be % 7.3f" % (E_c_numeric, E_c_formula)
    print "    mode[phi]=% 7.3f   should be % 7.3f" % (
        mode_phi_numeric, mode_phi_formula)

def makePlot_pdf_Phi(
        nu=0, tau=0, phi_grid=None, **kwargs):
    label = 'nu=%7.2f' % (nu)
    cPrior = c_Prior(nu=nu, tau=tau)
    if phi_grid is None:
        phi_grid = make_phi_grid(**kwargs)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    pdf_grid = np.exp(logpdf_grid)
    pylab.plot(phi_grid, pdf_grid, '-', label=label)
    pylab.xlabel('phi')
    pylab.ylabel('density p(phi)')

def makePlot_pdf_Mu(
        nu=0, tau=0, mu_grid=None, **kwargs):
    label = 'nu=%7.2f' % (nu,)
    cPrior = c_Prior(nu=nu, tau=tau)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    phi_grid = mu2phi(mu_grid)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    logJacobian_grid = logJacobian_mu(mu_grid)
    pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
    pylab.plot(mu_grid, pdf_grid, '-', label=label)
    pylab.xlabel('mu')
    pylab.ylabel('density p(mu)')

def checkFacts_pdf_Mu(nu=0, tau=0, mu_grid=None, **kwargs):
    cPrior = c_Prior(nu=nu, tau=tau)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    phi_grid = mu2phi(mu_grid)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    logJacobian_grid = logJacobian_mu(mu_grid)
    pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
    Integral = np.trapz(pdf_grid, mu_grid)
    E_mu_numeric = np.trapz(pdf_grid * mu_grid, mu_grid)
    E_mu_formula = tau/nu
    E_phi_numeric = np.trapz(pdf_grid * phi_grid, mu_grid)
    E_phi_formula = E_phi(nu,tau)
    print "nu=%7.3f tau=%7.3f" % (nu, tau)
    print "     Integral=% 7.3f   should be % 7.3f" % (Integral, 1.0)
    print "        E[mu]=% 7.3f   should be % 7.3f" % (E_mu_numeric, E_mu_formula)
    print "       E[phi]=% 7.3f   should be % 7.3f" % (E_phi_numeric, E_phi_formula)

def makePlot_cumulant_Phi(
        phi1=None, phi2=None, **kwargs):
    if phi1 is None:
        phi1 = make_phi1_grid(**kwargs)
        grid = phi1
        label = 'phi2 = %7.3f' % (phi2)
        xlabel = 'phi_1'
    if phi2 is None:
        phi2 = make_phi2_grid(**kwargs)
        grid = phi2
        label = 'phi1 = %7.3f' % (phi1)
        xlabel = 'phi_2'
    c_grid = c_Phi(phi1, phi2)
    pylab.plot(grid, c_grid, '-', label=label)
    pylab.xlabel(xlabel)
    pylab.ylabel('c(phi)')

def makePlot_cumulant_Mu(
        mu1=None, mu2=None, **kwargs):
    if mu1 is None:
        mu1 = make_mu1_grid(**kwargs)
        grid = mu1
        label = 'mu2 = %7.3f' % (mu2)
        xlabel = 'mu1'
    if mu2 is None:
        mu2 = make_mu2_grid(**kwargs)
        grid = mu2
        label = 'mu1 = %7.3f' % (mu1)
        xlabel = 'mu2'
    c_grid = c_Mu(mu1, mu2)
    pylab.plot(grid, c_grid, '-', label=label)
    pylab.xlabel(xlabel)
    pylab.ylabel('gamma(mu)')

def makePlot_cumulant_Prior(
        nu_grid=None, tau=5.0, ngrid=10000, **kwargs):
    if nu_grid is None:
        nu_grid = np.linspace(1e-10, 100, ngrid)
    c_grid = c_Prior(nu_grid, tau)
    pylab.plot(nu_grid, c_grid, 'k:')
    pylab.xlabel('nu')
    pylab.ylabel('prior cumulant(nu, tau)')

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
    for mu2 in [-2.1, -1.1, 0, 1, 2, 4][::-1]:
        makePlot_cumulant_Mu(mu2=mu2, min_mu=np.square(mu2)+1e-10, max_mu=100)
    pylab.legend(loc='upper right', fontsize=13)
    pylab.xlim([-2, 100])

    '''
    pylab.figure()
    for phi2 in [-10.5, -5.5, 0, 5, 10, 20][::-1]:
        makePlot_cumulant_Phi(phi2=phi2, min_phi=-30)
    pylab.ylim([-5, 30])
    pylab.xlim([-20, 0.5])
    pylab.legend(loc='upper left', fontsize=13)


    nuRange = [1/2.0, 1, 2.0, 8, 32, 128]
    mu_Phi = 0.5 * 1.0/3.0
    print "mu(Mode[phi]): ", mu_Phi
    print "Mode[phi]: ", mu2phi(mu_Phi)

    pylab.figure()
    for nu in nuRange[::-1]:
        tau = mu_Phi * nu
        checkFacts_pdf_Phi(nu=nu, tau=tau, min_phi=-50, ngrid=5e6)
        makePlot_pdf_Phi(nu=nu, tau=tau, min_phi=4*mu2phi(mu_Phi), ngrid=2e4)
    pylab.legend(loc='upper left', fontsize=13)
    pylab.xticks([-12, -9, -6, -3, 0])
    pylab.savefig('ZMG_densityPhi.eps')
    
    pylab.figure()
    for nu in nuRange[::-1]:
        tau = mu_Phi * nu
        checkFacts_pdf_Mu(nu=nu, tau=tau, min_mu=1e-10, max_mu=20, ngrid=2e6)
        makePlot_pdf_Mu(nu=nu, tau=tau, min_mu=1e-10, max_mu=3 * mu_Phi, ngrid=2e4)
    pylab.legend(loc='upper right', fontsize=13)
    pylab.savefig('ZMG_densityMu.eps')

    
    pylab.figure()
    makePlot_cumulant_Prior(tau=5.0, ngrid=5000)

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
    pylab.show()
