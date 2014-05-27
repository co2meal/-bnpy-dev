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
from bnpy.birthmove import BirthMove, BirthRefine, TargetPlanner, TargetDataSampler
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

BirthArgsIN = dict(
               birthDebug=1,              
               Kfresh=5,
               Kmax=300,
               birthRetainExtraMass=0,
               birthHoldoutData=0,
               birthVerifyELBOIncrease=0,
               birthVerifyELBOIncreaseFresh=0,
               creationRoutine='randexamples', 
               creationDoUpdateFresh=0,
               creationNumIters=0,
               expandOrder='expandThenRefine',
               expandAdjustSuffStats=1,
               refineNumIters=50,
               cleanupDeleteEmpty=0,
               cleanupDeleteToImprove=0,
               cleanupDeleteToImproveFresh=0,
               cleanupDeleteViaLP=0,
               cleanupMinSize=50,
                )

def MakeLeanSaveInfo(Info, doTraceBeta=0):
  Results = dict(traceELBO=Info['traceELBO'])
  if doTraceBeta:
    Q = dict( traceBeta=Info['traceBeta'],
              traceN=Info['traceN']
             )
    Results.update(Q)
  if 'xbigModelInit' in Info:
    Q = dict( initTopics=makeTopics(Info['xbigModelInit']),
              finalTopics=makeTopics(Info['xbigModelRefined'])
            )
    Q['Korig'] = Info['Korig']
    Results.update(Q)
  return Results

def makeTopics(model):
  K = model.obsModel.K
  topics = np.zeros((K, model.obsModel.comp[0].lamvec.size))
  for k in xrange(K):
    topics[k,:] = model.obsModel.comp[k].lamvec
  topics /= topics.sum(axis=1)[:,np.newaxis]
  return topics

def RunCurrentModelOnTargetData(outPath, model, bigSS, bigData, targetData,
                                         seed=0, **kwargs):
  ''' Play current model forward on the targeted dataset
  '''
  BirthArgs = dict(**BirthArgsIN)
  BirthArgs.update(**kwargs)
  BirthArgs['randstate'] = np.random.RandomState(seed)
  BirthArgs['seed'] = seed

  fwdmodel = model.copy()
  xmodel, _, _, Info = BirthRefine.refine_expanded_model_with_VB_iters(
                                          fwdmodel, targetData,
                                          xbigSS=bigSS, **BirthArgs)

  Results = MakeLeanSaveInfo(Info)
  joblib.dump(Results, outPath)
  print '... Wrote all results to file: ', outPath
  

def RunBirthOnTargetData(outPath, model, bigSS, bigData, targetData, targetInfo,
                               cachefile='', seed=0, **kwargs): 
  BirthArgs = dict(**BirthArgsIN)
  BirthArgs.update(**kwargs)
  BirthArgs['randstate'] = np.random.RandomState(seed)
  BirthArgs['seed'] = seed

  assert bigData.nDoc == bigSS.nDoc

  print '--------------------------------------- Data facts'
  print 'D=%d' % (bigData.nDoc)
  print bigData.get_doc_stats_summary()

  print '--------------------------------------- TargetData facts'
  print 'D=%d' % (targetData.nDoc)
  print targetData.get_doc_stats_summary()

  if BirthArgs['creationRoutine'] == 'xspectral':
    Q = bigData.to_wordword_cooccur_matrix()
  else:
    Q = None

  xmodel, xSS, Info = BirthMove.run_birth_move(model, bigSS, targetData,
                                                           Q=Q,
                                                           **BirthArgs)
 
  Results = MakeLeanSaveInfo(Info, doTraceBeta=1)
  targetData.deleteNonEssentialAttributes()
  Results['targetData'] = targetData
  Results['targetInfo'] = targetInfo
  Results['cachefile']=cachefile
  joblib.dump(Results, outPath)
  print 'Wrote all results to file: ', outPath

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


########################################################### Target Data
###########################################################

def MakeTargetData(selectName, Data, model, SS, LP, initName=None,
                                            seed=0, **kwargs):
  TargetSamplerArgs = dict(**TargetSamplerArgsIN)
  TargetSamplerArgs.update(kwargs)
  TargetSamplerArgs['randstate'] = np.random.RandomState(seed)

  targetData = None
  anchors = None
  ktarget = None
  ps = None
  ktrue = None
  if selectName == 'none':
    ktarget = None
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
  elif selectName.lower().count('word'):

    if selectName.count('anchor'):
      Q = Data.to_wordword_cooccur_matrix()
    else:
      Q = None
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

  if targetData is None:
    targetData = TargetDataSampler.sample_target_data(Data, 
                                               model=model, LP=LP,
                                               targetWordIDs=anchors,
                                               targetCompID=ktarget,
                                               **TargetSamplerArgs)
  Info = dict(ktarget=ktarget, targetWordIDs=anchors, ps=ps, ktrue=ktrue)
  return targetData, Info

def findMostMissingTrueTopic(TrueTopics, EstTopics, Ktrue, Kest):
  Dist = np.zeros( (Ktrue,Kest))
  for ktrue in xrange(Ktrue):
    for kk in xrange(Kest):
      Dist[ktrue,kk] = np.sum(np.abs(TrueTopics[ktrue,:] - EstTopics[kk,:]))
  minDistPerTrueTopic = Dist.min(axis=1)
  ktarget = minDistPerTrueTopic.argmax()
  return ktarget

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

  Data, model, SS, LP, cachefile = LoadModelAndData(args.data, args.initName)

  targetData = None
  for trial in xrange(10):
    try:
      targetData, targetInfo = MakeTargetData(args.selectName, Data, model, 
                              SS, LP,
                              seed=1000*args.task + trial, **kwargs)
    except BirthProposalError:
      print 'EMPTY! Retrying...'
      continue
    if targetData.nDoc >= 10:
      break
    else:
      print 'TOO SMALL! Retrying...'

  outPath = createOutPath(args)
  RunBirthOnTargetData(outPath, model, SS, Data, targetData, targetInfo,
                             cachefile=cachefile, seed=args.task, **kwargs)

  outPath = createOutPath(args, 'FastForwardResults.dump')
  RunCurrentModelOnTargetData(outPath, model, SS, Data, targetData,
                             cachefile=cachefile, seed=args.task, **kwargs)

