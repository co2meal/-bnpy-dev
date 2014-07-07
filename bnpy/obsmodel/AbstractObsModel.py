class AbstractObsModel(object):

  def __init__(self):
    pass

  def to_dict(self):
    PDict = dict(name=self.__class__.__name__, 
                 inferType=self.inferType)
    if hasattr(self, 'min_covar'):
      PDict['min_covar'] = self.min_covar
    if hasattr(self, 'EstParams'):
      PDict['K'] = self.EstParams.K
      PDict['D'] = self.EstParams.D
      for key in self.EstParams._FieldDims.keys():
        PDict[key] = getattr(self.EstParams, key)
    if hasattr(self, 'Post'):
      PDict['K'] = self.Post.K
      PDict['D'] = self.Post.D
      for key in self.Post._FieldDims.keys():
        PDict[key] = getattr(self.Post, key)
    return PDict

  def get_prior_dict(self):
    PDict = dict()
    if hasattr(self, 'Prior'):
      PDict['D'] = self.Prior.D
      for key in self.Prior._FieldDims.keys():
        PDict[key] = getattr(self.Prior, key)
    return PDict
  

  def calc_local_params(self, Data, LP=None, **kwargs):
    if LP is None:
      LP = dict()
    if self.inferType == 'EM':
      LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromEstParams(Data)
    else:
      LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromPost(Data)      
    return LP

  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    SS = self.calcSummaryStats(Data, SS, LP, **kwargs)
    return SS

  def update_global_params(self, SS, rho=None, **kwargs):
    if self.inferType == 'EM':
      return self.updateEstParams_MaxLik(SS)
    else:
      return self.updatePost(SS)

  def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
    if self.inferType == 'EM':
      return 0
    else:
      return self.calcELBO_Memoized(SS)

  def GetCached(self, key, k=None):
    ''' Retrieved cached function evaluation if possible,
          otherwise compute fresh result and store in cache for later.
    '''
    ckey = key + '-' + str(k)
    try:
      return self.Cache[ ckey ]
    except KeyError:
      Val = getattr(self, '_' + key)(k)
      self.Cache[ ckey ] = Val
      return Val

  def ClearCache(self):
    self.Cache.clear()