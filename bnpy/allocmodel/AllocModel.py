''' AllocModel.py
'''
from __future__ import division


class AllocModel(object):

    def __init__(self, inferType):
        self.inferType = inferType

    def set_prior(self, **kwargs):
        pass

    def get_keys_for_memoized_local_params(self):
        ''' Return LP field names required for warm starts of local step
        '''
        return list()

    # ----    Local step update
    def calc_local_params(self, Data, LP):
        ''' Compute local parameters for each data item and component.
        '''
        pass

    def sample_local_params(self, obsModel, Data, SS, LP):
        ''' Sample local assignments for each data item.
        '''
        pass

    # ----    Summary step update
    def get_global_suff_stats(self, Data, SS, LP):
        ''' Calculate sufficient statistics for each component.
        '''
        pass

    # ----    Global step update
    def update_global_params(self, SS, rho=None, **kwargs):
        ''' Update (in-place) global parameters for this model.

            This is the M-step of EM/VB algorithm
        '''
        self.K = SS.K
        if self.inferType == 'EM':
            self.update_global_params_EM(SS)
        elif self.inferType == 'VB' or self.inferType.count('moVB'):
            self.update_global_params_VB(SS, **kwargs)
        elif self.inferType == 'GS':
            self.update_global_params_VB(SS, **kwargs)
        elif self.inferType == 'soVB':
            if rho is None or rho == 1:
                self.update_global_params_VB(SS, **kwargs)
            else:
                self.update_global_params_soVB(SS, rho, **kwargs)
        else:
            raise ValueError(
                'Unrecognized Inference Type! %s' % (self.inferType))

    # ----    ELBO functions
    def calc_evidence(self, Data, SS, LP):
        pass

    # ----    I/O functions
    def get_info_string(self):
        ''' Returns one-line human-readable terse description of this object
        '''
        pass

    def to_dict_essential(self):
        PDict = dict(name=self.__class__.__name__, inferType=self.inferType)
        if hasattr(self, 'K'):
            PDict['K'] = self.K
        return PDict

    def to_dict(self):
        pass

    def from_dict(self):
        pass

    def get_prior_dict(self):
        pass

    def make_hard_asgn_local_params(self, LP):
        ''' Convert soft to hard assignments for provided local params

        Parameters
        --------
        LP : dict
            Local parameters as key/value string/array pairs
            * resp : 2D array, size N x K 

        TODO
        '''
        LP['Z'] = np.argmax(LP['resp'], axis=1)
        K = LP['resp'].shape[1]
        LP['resp'].fill(0)
        for k in xrange(K):
            LP['resp'][LP['Z'] == k, k] = 1
        return LP

