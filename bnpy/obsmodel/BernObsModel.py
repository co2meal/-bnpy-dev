'''
ObsCompSet.py

Generic object for managing a prior and set of K components
'''
from IPython import embed

from ..distr import BernDistr
from ..distr import BetaDistr
import numpy as np
from bnpy.util import gammaln, digamma, EPS

import copy
from ObsCompSet import ObsCompSet

class BernObsModel( ObsCompSet ):
    def __init__(self, inferType, obsPrior=None):
        self.inferType = inferType
        self.comp = list()
        self.obsPrior = obsPrior # specify prior distribution which is a beta
        self.bounds = dict()
    def reset(self):
        pass  
    
    def set_inferType( self, inferType):
        self.inferType = inferType
  
############################################################## set prior parameters  ##############################################################    
    @classmethod
    def InitFromData(cls, inferType, priorArgDict, Data):
        if inferType == 'EM':
            obsPrior = None
            return cls(inferType, obsPrior)
        else:
            obsPrior = BetaDistr.InitFromData(priorArgDict, Data)
            return cls(inferType, obsPrior)
    
############################################################## human readable I/O  ##############################################################  
    def get_info_string(self):
        return 'BernObs'
        
    def get_info_string_prior(self):
        if self.obsPrior is None:
            return 'None'
        else:
            return 'Beta-Bernoulli Prior'
        
    def get_human_global_param_string(self):
        ''' Returns human-readable numerical repr. of parameters,
          for quick inspection of correctness'''
        pass

############################################################## MAT file I/O  ##############################################################  
    def to_dict_essential(self):
        PDict = dict(name=self.__class__.__name__, inferType=self.inferType)
        if hasattr(self,"K"):
            PDict['K']=self.K
        return PDict
      
    def from_dict(self):
        pass
    
    def get_prior_dict( self ):
        pass
        
#########################################################  Suff Stat Calc #########################################################   
    def get_global_suff_stats(self, Data, SS, LP, **kwargs):
            # sufficient statistic for lambda weights
        phi = LP['phi']
        #rhos = LP['rhos']
        K, _ = phi.shape 
        oa = np.zeros( (K,K) )
        ob = np.zeros( (K,K) )
        Y = Data.X
        E = Data.nObs
        for e in xrange(E):
            i = Y[e,0]
            j = Y[e,1]
            if Y[e,2] == 1:
                #oa += np.outer(phi[:,i], rhos[:,j])
                oa += np.outer(phi[:,i], phi[:,j])
            else:
                #ob += np.outer(phi[:,i], rhos[:,j])
                ob += np.outer(phi[:,i], phi[:,j])
        SS['oa'] = oa
        SS['ob'] = ob
    
        return SS 

    
#########################################################  Param Update Calc ######################################################### 
    def update_obs_params_EM(self):
        pass
    
    def update_obs_params_VB(self, SS, Krange):
        # update lambda parameters
        # update comp[k,l].update(SS)
        for k in xrange(self.K):
            for l in xrange(self.K):
                self.comp[k,l] = self.obsPrior.get_post_distr( self, SS )
        '''
        self.lambda_a = SS.oa + self.obsPrior.a
        self.lambda_b = SS.ob + self.obsPrior.b
        self.ElogW1 = digamma(self.lambda_a) - digamma(self.lambda_a + self.lambda_b)
        self.ElogW2 = digamma(self.lambda_b) - digamma(self.lambda_a + self.lambda_b)
        '''
    def update_obs_params_VB_soVB(self):
        pass

#########################################################  Evidence Calc #########################################################     
    def calc_evidence(self, Data, SS, LP):        
        ha = self.obsPrior.a
        hb = self.obsPrior.b
        
        phi = LP["phi"]
        #rhos = LP["rhos"]
        X = Data.X
        logY = LP["E_log_soft_ev"]
        
        # Calculate observation likelihoods
        py = 0
        for e in xrange(Data.nObs):
            i = X[e,0]
            j = X[e,1]
            #temp = np.outer(phi[:,i], rhos[:,j]) * logY[e,:,:]
            temp = np.outer(phi[:,i], phi[:,j]) * logY[e,:,:]
            py += temp.sum()
        
        # entropy of local parameters
        qz = (phi * np.log(phi+1e-10)).sum()
        #qR = rhos * np.log(rhos)
        
        # stochastic block matrix
        po = ((gammaln(ha+hb)-gammaln(ha)-gammaln(hb)) + ((ha-1)*self.ElogW1) + ((hb-1)*self.ElogW2)).sum()
        qo = (gammaln(self.lambda_a + self.lambda_b) - gammaln(self.lambda_a) - gammaln(self.lambda_b) + (self.lambda_a - 1)*self.ElogW1 + (self.lambda_b - 1) * self.ElogW2).sum()
        #elbo_obs = pY + pO.sum() - qS.sum() - qR.sum() - qO.sum()
        elbo_obs = py + po - qz - qo
        
        if 'pY' not in self.bounds:
            self.bounds['pY'] = [py]
            self.bounds['qZ'] = [qz]
            self.bounds['pO'] = [po]
            self.bounds['qO'] = [qo]
        else:
            self.bounds['pY'].append(py)
            self.bounds['qZ'].append(qz)
            self.bounds['pO'].append(po)
            self.bounds['qO'].append(qo)
        
        return elbo_obs
        
     
    def update_global_params( self, SS, rho=None, Krange=None):
        ''' M-step update'''
        self.K = SS.K
        # have something special for relational models
        if len(self.comp) != self.K:
            self.comp = np.zeros( (self.K, self.K), dtype=object)
            for k in xrange(self.K):
                for l in xrange(self.K):
                    self.comp[k,l] = copy.deepcopy(self.obsPrior)
        if Krange is None:
            Krange = xrange(self.K)
        if self.inferType == 'EM':
            self.update_obs_params_EM( SS, Krange )
        elif self.inferType.count('VB')>0:
            if rho is None or rho == 1.0:
                self.update_obs_params_VB( SS, Krange )
            else:
                self.update_obs_params_soVB( SS, rho, Krange )
  
######################################################### Local Param updates  

    def calc_local_params( self, Data, LP=dict()):
        if self.inferType == 'EM':
            LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
        elif self.inferType.count('VB') >0:
            # calculate f(y_ij,\omega)
            LP['E_log_soft_ev'] = self.E_log_pdf( Data )
        return LP

    def log_pdf( self, Data, Krange=None):
        E = Data.nObs
        K = self.K
        E_log_pX = np.zeros( (E, K, K) ) 
        for k in xrange(K):
            for l in xrange(K):
                E_log_pX[:,k,l] = self.obsPrior.ElogA[k,l] * Data[:,2] + self.obsPrior.ElogB[k,l] * (1-Data[:,2])
        return E_log_pX
    
    def E_log_pdf( self, Data):
        E = Data.nObs
        K = self.K
        X = Data.X
        E_log_pX = np.zeros( (E, K, K) ) 
        for k in xrange(K):
            for l in xrange(K):
                E_log_pX[:,k,l] = self.ElogW1[k,l] * X[:,2] + self.ElogW2[k,l] * (1-X[:,2])
        
        return E_log_pX
    

#########################################################  Comp List add/remove
    def add_empty_component( self ):
        self.K = self.K+1
        self.comp.append( copy.deepcopy(self.obsPrior) )

    def add_component( self, c=None ):
        self.K = self.K+1
        if c is None:
            self.comp.append( copy.deepcopy(self.obsPrior) )
        else:
            self.comp.append( c )
  
    def remove_component( self, delID):
        self.K = self.K - 1
        comp = [ self.comp[kk] for kk in range(self.K) if kk is not delID ]
        self.comp = comp    
    
    def delete_components( self, keepIDs ):
        if type(keepIDs) is not list:
            keepIDs = [keepIDs]
        comp = [ self.comp[kk] for kk in range(self.K) if kk in keepIDs ]
        self.comp = comp
        self.K

