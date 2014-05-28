#! /contrib/projects/EnthoughtPython/epd64/bin/python -W ignore::DeprecationWarning
#$ -S /contrib/projects/EnthoughtPython/epd64/bin/python
# ------ set working directory
#$ -cwd 
# ------ attach environment variables
#$ -v PYTHONPATH -v BNPYOUTDIR -v BNPYDATADIR -v OMP_NUM_THREADS
# ------ send to particular queue
#$ -o /data/liv/liv-x/topic_models/birth-results/logs/$JOB_ID.$TASK_ID.out
#$ -e /data/liv/liv-x/topic_models/birth-results/logs/$JOB_ID.$TASK_ID.err

import argparse
from matplotlib import pylab
import numpy as np
import os
import sys
import joblib
from distutils.dir_util import mkpath

import bnpy
from bnpy.birthmove.BirthProposalError import BirthProposalError
from bnpy.birthmove import TargetPlanner, TargetDataSampler, BirthLogger
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB

CACHEDIR = '/Users/mhughes/git/bnpy2/local/dump/'
if not os.path.exists(CACHEDIR):
  CACHEDIR = '/data/liv/liv-x/topic_models/birth-results/'
assert os.path.exists(CACHEDIR)

TargetSamplerArgsIN = dict(
               randstate=np.random.RandomState(4),
               targetExample=0,
               targetMinSize=10,
               targetMaxSize=200,
               targetMinWordsPerDoc=100,
               targetNumWords=20,
               targetWordMinCount=1,
               targetMinKLPerDoc=0,
               targetHoldout=0,
               targetCompFrac=0.25,
                 )


########################################################### Target Data
###########################################################

def MakeTargetPlans(selectName, Data, model, SS, LP, initName=None,
                                nPlans=1, seed=0, **kwargs):
  TargetSamplerArgs = dict(**TargetSamplerArgsIN)
  TargetSamplerArgs.update(kwargs)
  TargetSamplerArgs['randstate'] = np.random.RandomState(seed)
  targetData = None
  anchors = None
  ktarget = None
  ps = None
  ktrue = None

  if selectName.count('anchor'):
    Q = Data.to_wordword_cooccur_matrix()
  else:
    Q = None

  Plans = list()
  BlankPlan = dict(ktarget=None, targetWordIDs=None, targetWordFreq=None)
  if selectName == 'none':
    Plans = [dict(**BlankPlan) for p in xrange(nPlans)]
  elif selectName.lower().count('freq'):
    Plans = TargetPlanner.makePlansToTargetWordFreq(model, Data=Data, LP=LP,
                                                    Q=Q, 
                                                    targetSelectName=selectName,
                                                    nPlans=nPlans,
                                                    **TargetSamplerArgs)
  else:
    for _ in range(nPlans):
      ktarget, ps = TargetPlanner.select_target_comp(SS.K, 
                                                Data=Data, model=model,
                                                SS=SS, LP=LP,
                                                targetSelectName=selectName,
                                                return_ps=1,
                                                **TargetSamplerArgs)
      Plans.append(dict(ktarget=ktarget, ps=ps))

  '''
  elif selectName == 'best':
    # TODO: identify true comp with largest L1 distance to nearest existing comp
    Kest = SS.K
    Ktrue = Data.TrueParams['K']
    TrueTopics = Data.TrueParams['topics']
    EstTopics = np.zeros( (Kest, Data.vocab_size))
    for kk in xrange(Kest):
      EstTopics[kk,:] = model.obsModel.comp[kk].lamvec
    EstTopics /= EstTopics.sum(axis=1)[:,np.newaxis]
    ktrue = findMostMissingTrueTopic( TrueTopics, EstTopics, Ktrue, Kest)
    docRanks = np.argsort( -1*Data.TrueParams['alphaPi'][:, ktrue] )
    targetMaxSize = TargetSamplerArgs['targetMaxSize']
    targetData = Data.select_subset_by_mask(docRanks[:targetMaxSize])
  '''
    
  '''
  elif selectName.lower().count('word'):
    anchors, ps = TargetPlanner.select_target_words(model=model, 
                           Data=Data, LP=LP, Q=Q,
                           targetSelectName=selectName, return_ps=1,
                           **TargetSamplerArgs)
  else:
    ktarget, ps = TargetPlanner.select_target_comp(SS.K, 
                                                Data=Data, model=model,
                                                SS=SS, LP=LP,
                                                targetSelectName=selectName,
                                                return_ps=1,
                                                **TargetSamplerArgs)
  '''
  return Plans

def SampleTargetDataForPlans(Plans, Data, model, LP, **kwargs):
  TargetSamplerArgs = dict(**TargetSamplerArgsIN)
  TargetSamplerArgs.update(kwargs)
  TargetSamplerArgs['randstate'] = np.random.RandomState(seed)
  if hasattr(Data, 'vocab_dict'):
    Vocab = [str(x[0][0]) for x in Data.vocab_dict]
  else:
    Vocab = None
  for pp in xrange(len(Plans)):
    Plan = Plans[pp]

    targetData = TargetDataSampler.sample_target_data(Data, 
                                               model=model, LP=LP,
                                               ktarget=Plan['ktarget'],
                                     targetWordIDs=Plan['targetWordIDs'],
                                     targetWordFreq=Plan['targetWordFreq'],          
                                               **TargetSamplerArgs)
    if 'log' not in Plan:
      Plan['log'] = list()
    Plan['log'].append('Target Data: %d docs' % (targetData.nDoc))
    Plan['log'].append(targetData.get_doc_stats_summary())

    Plan['log'].append('Freq. of all words in Target')
    Plan['log'].append(targetData.get_most_common_words_summary(Vocab))

    if 'topWordIDs' in Plan:
      Plan['log'].append('Freq. of Top-Ranked Words in Target')
      Plan['log'].append(targetData.get_most_common_words_summary(Vocab,
                                        targetWordIDs=Plan['topWordIDs']))

    Plan['log'].append(targetData.get_example_documents_summary(5, Vocab))
    Plan['Data'] = targetData

    BirthLogger.logPhase('Plan ' + str(pp))
    BirthLogger.writePlanToLog(Plan)
  return Plans         

def findMostMissingTrueTopic(TrueTopics, EstTopics, Ktrue, Kest):
  Dist = np.zeros( (Ktrue,Kest))
  for ktrue in xrange(Ktrue):
    for kk in xrange(Kest):
      Dist[ktrue,kk] = np.sum(np.abs(TrueTopics[ktrue,:] - EstTopics[kk,:]))
  minDistPerTrueTopic = Dist.min(axis=1)
  ktarget = minDistPerTrueTopic.argmax()
  return ktarget

########################################################### Load Data
###########################################################

def LoadData(dataname):
  if dataname.count('Bars'):
    import BarsK10V900
    Data = BarsK10V900.get_data(nDocTotal=2000, nWordsPerDoc=250)
  elif dataname.count('NIPS'):
    os.environ['BNPYDATADIR'] = '/data/NIPS/'
    if not os.path.exists(os.environ['BNPYDATADIR']):
      os.environ['BNPYDATADIR'] = '/data/liv/liv-x/topic_models/data/nips/'
    sys.path.append(os.environ['BNPYDATADIR'])
    import NIPSCorpus
    Data = NIPSCorpus.get_data()
  elif dataname.count('huffpost'):
    os.environ['BNPYDATADIR'] = '/data/liv/liv-x/topic_models/data/huffpost/'
    sys.path.append(os.environ['BNPYDATADIR'])
    import huffpost
    Data = huffpost.get_data()
  else:
    raise NotImplementedError(dataname)
  return Data

########################################################### Model creation
###########################################################
def InitModelWithKTopics(Data, aModel='HDPStickBreak', K=1):
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, K=K, initname='randexamples',
                                  seed=0)
  return hmodel

def InitModelWithTrueTopics(Data, aModel='HDPStickBreak'):
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')
  return hmodel

def InitModelWithOneMissing(Data, kmissing=0, aModel='HDPStickBreak'):
  # Remove specific comp from the model
  if type(kmissing) == str:
    kmissing = int(kmissing.split('=')[1])

  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')

  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  SS.removeComp(kmissing)
  hmodel.obsModel.update_global_params(SS)

  beta = OptimHDPSB._v2beta(hmodel.allocModel.rho)
  newbeta = np.hstack([beta[:kmissing], beta[kmissing+1:]])
  hmodel.allocModel.set_global_params(K=SS.K, beta=newbeta[:-1])
  return hmodel

def MakeBigModel(Data, initName, nIters=20):
  '''
      Returns
      -------
      model : bnpy model for Data
      SS : SuffStatBag summarizing all of Data
  '''
  if initName.count('K='):
    K = int(initName.split('=')[1])
    model = InitModelWithKTopics(Data, K=K)
  elif initName.count('missing='):
    kmissing = int(initName.split('=')[1])
    model = InitModelWithOneMissing(Data, kmissing=kmissing)
  elif initName == 'Truth':
    model = InitModelWithTrueTopics(Data)
  else:
    raise NotImplementedError("UNKNOWN init scenario: " + initName)
  LP = None
  for ii in xrange(nIters):
    LP = model.calc_local_params(Data, LP, methodLP='scratch,memo')
    SS = model.get_global_suff_stats(Data, LP)
    model.update_global_params(SS)

    if (ii+1) % 5 == 0:
      print '%d/%d' % (ii+1, nIters)

  del LP['E_logsoftev_WordsData']
  del LP['word_variational']
  del LP['expEloglik']
  del LP['expElogpi']
  del LP['digammaBoth']
  del LP['topics']

  return model, SS, LP

def LoadModelAndData(dataName, initName):
  cachepath = '%s/%s/' % (CACHEDIR, dataName)
  mkpath(cachepath)
  cachepath = os.path.join(cachepath, 'DataAndModel-%s.dump' % (initName))
  if os.path.exists(cachepath):
    Q = joblib.load(cachepath)
    Data = Q['Data']
    model = Q['model']
    SS = Q['SS']
    LP = Q['LP']

    print 'Loaded from the CACHE!'
  else:
    Data = LoadData(dataName)
    model, SS, LP = MakeBigModel(Data, initName)
    Data.deleteNonEssentialAttributes()
    
    joblib.dump(dict(Data=Data, model=model, SS=SS, LP=LP), cachepath)
  return Data, model, SS, LP, cachepath

def createOutPath(args, basename='BirthResults.dump'):
  outPath = os.path.join(CACHEDIR, args.data, 
                         args.jobName, str(args.task))
  mkpath(outPath)
  outPath = os.path.join(outPath, basename)
  return outPath

########################################################### main
###########################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', default='BarsK10V900')
  parser.add_argument('initName', default='K1')
  parser.add_argument('--jobName', type=str, default='')
  parser.add_argument('--selectName', type=str, default='none')
  parser.add_argument('--savepath', type=str, default=None)
  parser.add_argument('--task', type=int, default=1)
  args, unkList = parser.parse_known_args()
  kwargs = bnpy.ioutil.BNPYArgParser.arglist_to_kwargs(unkList)
  if len(args.jobName) == 0:
    args.jobName = args.selectName
  taskKey = 'SGE_TASK_ID'
  if taskKey in os.environ:
    args.task = int(os.environ[taskKey])
  print "TASK ", args.task

  BirthLogger.configure(args.savepath, doSaveToDisk=0, doWriteStdOut=1)

  Data, model, SS, LP, cachefile = LoadModelAndData(args.data, args.initName)

  targetData = None
  seed = 1000 * args.task
  Plans = MakeTargetPlans(args.selectName, 
                              Data, model, SS, LP,
                              seed=seed, **kwargs)
   
  Plans = SampleTargetDataForPlans(Plans, Data, model, LP, seed=seed, **kwargs)

  

