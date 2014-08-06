import numpy as np

import bnpy
HModel = bnpy.HModel


###########################################################
###########################################################

def MakeData(dName, **kwargs):
  if dName == 'K1D1':
    return MakeData_K1D1(**kwargs)
  elif dName == 'K1D2':
    return MakeData_K1D1(**kwargs)
  elif dName == 'AsteriskK8':
    return MakeData_AsteriskK8D2(**kwargs)
  else:
    raise NotImplementedError(dName)

def MakeData_K1D1(N=10000):
  ''' Create simple toy XData from standard normal x[n] ~ N(0, 1)
  '''
  PRNG = np.random.RandomState(0)
  X = PRNG.randn(N,1)
  Data = bnpy.data.XData(X)
  
  TrueResp = np.ones((N,1))
  DupResp = np.zeros((N,2))
  DupResp[:N/2,0] = 1
  DupResp[N/2:,1] = 1
  Data.TrueParams = dict(TrueResp=TrueResp, DupResp=DupResp)
  return Data

def MakeData_K1D2(N=10000):
  ''' Create simple toy XData from standard normal x[n] ~ N(0, 1)
  '''
  PRNG = np.random.RandomState(0)
  X = PRNG.randn(N,2)
  Data = bnpy.data.XData(X)
  
  TrueResp = np.ones((N,1))
  DupResp = np.zeros((N,2))
  DupResp[:N/2,0] = 1
  DupResp[N/2:,1] = 1
  Data.TrueParams = dict(TrueResp=TrueResp, DupResp=DupResp)
  return Data

def MakeData_AsteriskK8D2(N=10000):
  ''' Create simple toy XData from standard normal x[n] ~ N(0, 1)
  '''
  PRNG = np.random.RandomState(0)
  import AsteriskK8
  Data = AsteriskK8.get_data(nObsTotal=N, seed=425)

  TrueResp = np.zeros( (N,8))
  DupResp = np.zeros( (N,8*2))
  for n in range(Data.nObs):
    k = Data.TrueLabels[n]
    TrueResp[n, k] = 1
    if n < Data.nObs/2:
      DupResp[n, k] = 1
    else:
      DupResp[n, k+8] = 1
  Data.TrueParams = dict(TrueResp=TrueResp, DupResp=DupResp)
  return Data

###########################################################
###########################################################

def MakeModel(mName, Data, **kwargs):
  if mName.lower() == 'true':
    return MakeModelWithTrueComps(Data, **kwargs)
  elif mName.count('duplicate'):
    return MakeModelWithDuplicateComps(Data, **kwargs)
  elif mName.count('K'):
    K = int(mName[1:])
    return MakeModelWithRandomComps(Data, K=K, **kwargs)
  else:
    raise NotImplementedError(mName)

def MakeModelWithTrueComps(Data, **kwargs):
  ''' Create hmodel with "true" components as self.Data, add to self
      Also create representative suff stats SS, add to self
  '''
  aDict = dict(alpha0=1.0, truncType='z')
  oDict = dict(kappa=1e-5, dF=1, ECovMat='eye', sF=1e-3)
  hmodel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', 
                                         aDict, oDict, Data)

  LP = dict(resp=Data.TrueParams['TrueResp'])
  SS = hmodel.get_global_suff_stats(Data, LP)
  for _ in range(3):
    hmodel.update_global_params(SS)
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
  return hmodel, SS, LP


def MakeModelWithDuplicateComps(Data, **kwargs):
  ''' Create model with "duplicated" components"
      For each true component k, we have "two" versions of it, k1 and k2.
        Half of the data generated by k is assigned to k1 
        and the other half is assigned to k2.
  '''
  aDict = dict(alpha0=1.0, truncType='z')
  oDict = dict(kappa=1e-5, dF=1, ECovMat='eye', sF=1e-3)
  dupModel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss',
                                           aDict, oDict, Data)

  LP = dict(resp=Data.TrueParams['DupResp'])
  SS = dupModel.get_global_suff_stats(Data, LP)
  for _ in range(3):
    dupModel.update_global_params(SS)
    LP = dupModel.calc_local_params(Data)
    SS = dupModel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
  return dupModel, SS, LP

def MakeModelWithRandomComps(Data, **kwargs):
  ''' Create hmodel with "true" components as self.Data, add to self
      Also create representative suff stats SS, add to self
  '''
  aDict = dict(alpha0=1.0, truncType='z')
  oDict = dict(kappa=1e-5, dF=1, ECovMat='eye', sF=1e-3)
  hmodel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', 
                                         aDict, oDict, Data)
  if 'initname' not in kwargs:
    kwargs['initname'] = 'randexamples'
  if 'K' not in kwargs:
    kwargs['K'] = 5
  hmodel.init_global_params(Data, **kwargs)
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  for _ in range(3):
    hmodel.update_global_params(SS)
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
  return hmodel, SS, LP