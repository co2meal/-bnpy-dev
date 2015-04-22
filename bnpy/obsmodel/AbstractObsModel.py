'''
AbstractObsModel

Generic parent class for all observation models (aka data-generation models).

Implements basic functionality common to all models, such as
* determining which subroutines to call depending on inferType (EM or VB)
* initialization of global parameters (again depending on EM or VB)
* caching of temporary "helper" parameters, for fast re-use
'''

class AbstractObsModel(object):

  def __init__(self):
    pass

  def to_dict(self):
    PDict = dict(name=self.__class__.__name__, 
                 inferType=self.inferType)
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
    PDict['name'] = self.__class__.__name__
    if hasattr(self, 'min_covar'):
      PDict['min_covar'] = self.min_covar
    if hasattr(self, 'inferType'):
      PDict['inferType'] = self.inferType
    if hasattr(self, 'Prior'):
      PDict['D'] = self.Prior.D
      for key in self.Prior._FieldDims.keys():
        PDict[key] = getattr(self.Prior, key)
    return PDict

  ######################################################### Set global params
  #########################################################
  def set_global_params(self, **kwargs):
    ''' Set global parameters to specific provided values

        This method provides overall governing logic for setting 
        the global parambag attributes of this model, either EstParams or Post.
        
        Returns
        ---------
        None.  Exactly one of Post or EstParams will be updated in-place.
    '''
    ## First, try setEstParams, and fall back on setPost on any trouble
    didSetPost = 0
    try:
      self.setEstParams(**kwargs)
    except:
      try:
        self.setPostFactors(**kwargs)
        didSetPost = 1
      except:
        raise ValueError('Unrecognised args for set_global_params')

    ## Make sure EM methods have an EstParams field
    if self.inferType == 'EM' and didSetPost:
      self.setEstParamsFromPost(self.Post, **kwargs)
      del self.Post

    ## Make sure VB methods have a Post field
    if self.inferType != 'EM' and not didSetPost:
      self.setPostFromEstParams(self.EstParams, **kwargs)
      del self.EstParams


  ######################################################### Local step
  #########################################################  
  def calc_local_params(self, Data, LP=None, **kwargs):
    if LP is None:
      LP = dict()
    if self.inferType == 'EM':
      LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromEstParams(Data)
    else:
      LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromPost(Data)      
    return LP


  ######################################################### Summary step
  #########################################################
  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    SS = self.calcSummaryStats(Data, SS, LP)
    return SS

  ######################################################### Global step
  #########################################################
  def update_global_params(self, SS, rho=None, **kwargs):
    if self.inferType == 'EM':
      return self.updateEstParams_MaxLik(SS)
    elif rho is not None and rho < 1.0:
      return self.updatePost_stochastic(SS, rho)
    else:
      return self.updatePost(SS)

  ######################################################### Evidence step
  #########################################################
  def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
    if self.inferType == 'EM':
      return 0
    else:
      return self.calcELBO_Memoized(SS)

  ######################################################### Caching
  #########################################################
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

  def getPseudoSuffStats(self, pSS, SS):
    pass