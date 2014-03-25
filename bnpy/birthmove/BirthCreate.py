'''
BirthCreate.py

Logic for *creating* new components given 
*  dataset (some subsample of full-dataset scale N)
*  existing model (with K comps, of scale N)
*  existing suff stats (with K comps, of scale N)

'''
from BirthProposalError import BirthProposalError
import BirthCleanup

def create_model_with_new_comps(bigModel, bigSS, freshData, **kwargs):
  '''

    Returns
    -------
    freshSS : SuffStatBag with Kfresh components,
                summarizing dataset Dchunk
  '''
  freshModel = bigModel.copy()

  if kwargs['creationroutine'] == 'findmissingtopics':
    freshModel = create_new_model_findmissingtopics(
                                  freshModel, freshData, 
                                  bigModel, **kwargs)
  elif kwargs['creationroutine'] == 'spectral':
    freshModel = create_new_model_spectral(freshModel, freshData, **kwargs)
  else:
    freshModel.init_global_params(freshData, K=kwargs['Kfresh'],
                                             initname=kwargs['creationroutine'],
                                             randstate=kwargs['randstate']) 
    
  freshLP = freshModel.calc_local_params(freshData)
  freshSS = freshModel.get_global_suff_stats(freshData, freshLP)

  if kwargs['cleanupDeleteEmpty']:
    freshModel, freshSS = BirthCleanup.delete_empty_comps(
                            freshData, freshModel, freshSS, Korig=0, **kwargs)
  return freshModel, freshSS

########################################################### Topic-model creation
###########################################################  

def create_new_model_findmissingtopics(freshModel, freshData, 
                                        bigModel, LP=None, **kwargs):
  Kfresh = kwargs['Kfresh']
  K = bigModel.obsModel.K

  if LP is None:
    LP = bigModel.calc_local_params(freshData)
  Prior = np.exp(LP['E_logPi'])
  Lik = np.exp(bigModel.obsModel.getElogphiMatrix())

  DocWordFreq_model = np.dot(Prior, Lik)
  DocWordFreq_model /= DocWordFreq_model.sum(axis=1)[:,np.newaxis]

  DocWordFreq_emp = Data.to_sparse_docword_matrix()
  DocWordFreq_emp /= 1e-9 + DocWordFreq_emp.sum(axis=1)[:,np.newaxis]

  DocWordFreq_missing = DocWordFreq_emp - DocWordFreq_model 
  DocWordFreq_missing /= 1e-9 + DocWordFreq_missing.sum(axis=1)[:,np.newaxis]

  WordFreq_ctrs, Z = KMeansRex.RunKMeans(DocWordFreq_missing, Kfresh,
                               initname='plusplus',
                               Niter=10, seed=0)
  np.maximum(1e-8, WordFreq_ctrs, out=WordFreq_ctrs)
  WordFreq_ctrs /= WordFreq_ctrs.sum(axis=1)[:,np.newaxis]

  freshModel.set_global_params(beta=np.ones(Kfresh)/Kfresh, K=Kfresh,
                               topics=WordFreq_ctrs
                              )
  return freshModel


def create_new_model_spectral(freshModel, freshData, **kwargs):
  pass