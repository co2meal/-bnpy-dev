import argparse
from matplotlib import pylab
import numpy as np
import os
import sys

import bnpy
from bnpy.birthmove import BirthMove, TargetPlanner, TargetDataSampler
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB


TargetSamplerArgsIN = dict( \
               randstate=np.random.RandomState(4),
               targetMaxSize=250,
               targetMinWordsPerDoc=100,
               targetMinKLPerDoc=0,
               targetHoldout=1,
               targetCompFrac=0.25,
                 )

BirthArgsIN = dict( \
               randstate=np.random.RandomState(0),
               Kfresh=5,
               Kmax=25,
               birthRetainExtraMass=0,
               birthVerifyELBOIncrease=0,
               birthHoldoutData=1,
               creationRoutine='randexamples', 
               creationNumIters=0,
               expandOrder='expandThenRefine',
               expandAdjustSuffStats=1,
               refineNumIters=3,
               cleanupDeleteEmpty=1,
               cleanupDeleteToImprove=0,
               cleanupDeleteToImproveFresh=1,
               cleanupDeleteViaLP=0,
               cleanupMinSize=25,
                )

def RunBirthMoveDemo(Data, initName, selectName, seed, savepath=None, **kwargs):
  if type(Data) == str:
    Data = loadData(Data)

  TargetSamplerArgs = dict(**TargetSamplerArgsIN)
  TargetSamplerArgs['randstate'] = np.random.RandomState(seed)
  BirthArgs = dict(**BirthArgsIN)
  BirthArgs['randstate'] = np.random.RandomState(seed)

  for key,val in kwargs.items():
    if key in TargetSamplerArgs:
      TargetSamplerArgs[key] = val
    elif key in BirthArgs:
      BirthArgs[key] = val

  if initName == 'K1':
    model = InitModelWithOneTopic(Data)
  elif initName == 'true':
    assert hasattr(Data, 'TrueParams')
    model = InitModelWithTrueTopics(Data)
  elif initName.count('missing'):
    model = InitModelWithTrueTopicsButMissingOne(Data, kmissing=initName)

  Kmax = model.allocModel.K + BirthArgs['Kfresh']

  LP = model.calc_local_params(Data)
  SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=1)

  if selectName == 'none':
    ktarget = None
  else:
    ktarget, ps = TargetPlanner.select_target_comp(SS.K, 
                                                Data=Data, model=model,
                                                SS=SS, LP=LP,
                                                targetSelectName=selectName,
                                                return_ps=1,
                                                **TargetSamplerArgs)


  targetData, holdData = TargetDataSampler.sample_target_data(Data, 
                                               model=model, LP=LP,
                                               targetCompID=ktarget,
                                               **TargetSamplerArgs)

  print '--------------------------------------- Data facts'
  print 'D=%d' % (Data.nDoc)
  print Data.get_doc_stats_summary()

  print '--------------------------------------- TargetData facts'
  print 'D=%d' % (targetData.nDoc)
  print targetData.get_doc_stats_summary()

  print '--------------------------------------- Selection'
  if ktarget is not None:
    print 'Target Index:', ktarget  
    print '   ', ' '.join(['%.2f' %(p) for p in ps])
    print '   ', '     ' * ktarget, '^'
  else:
    print 'not targeted at specific topic'

  xmodel, xSS, Info = BirthMove.run_birth_move(model, SS, targetData,
                                             **BirthArgs)
  

  print '--------------------------------------- Birth'
  print 'What msg did run_birth_move output? Remember we always accept.'
  print '   ', Info['msg']

  if id(xmodel) == id(model):
    sys.exit(0)

  pprint_accept_decision(targetData, 'target', model, xmodel, Info)
  pprint_accept_decision(holdData, 'heldout', model, xmodel, Info)
  pprint_accept_decision(Data, 'all', model, xmodel, Info)


  if Data.nDoc == 2000:
    ################################################## plot documents
    figID = pylab.figure(101)
    bnpy.viz.BarsViz.plotExampleBarsDocs(targetData, figID=figID, doShowNow=False)
    if savepath is not None:
      savefile = os.path.join(savepath, 'ExampleDocsTarget.png')
      pylab.savefig(savefile, bbox='tight')

    figID = pylab.figure(102)
    bnpy.viz.BarsViz.plotExampleBarsDocs(holdData, figID=figID, doShowNow=False)
    if savepath is not None:
      savefile = os.path.join(savepath, 'ExampleDocsHeldout.png')
      pylab.savefig(savefile, bbox='tight')
    pylab.show(block=False)

    ################################################## plot topics Before/After
    figID, ax = pylab.subplots(3, 2, figsize=(8, 8))
    bnpy.viz.BarsViz.plotBarsFromHModel(model, Kmax=Kmax,
                                      figH=pylab.subplot(3,2,1))
    pylab.title('BEFORE')
    bnpy.viz.BarsViz.plotBarsFromHModel(xmodel, Kmax=Kmax,
                                      figH=pylab.subplot(3,2,2))
    pylab.title('AFTER')
    bnpy.viz.BarsViz.plotBarsFromHModel(Info['freshInfo']['freshModelInit'],
                                      Kmax=Kmax,
                                      figH=pylab.subplot(3,2,3))
    pylab.title('Fresh Model INIT')
    bnpy.viz.BarsViz.plotBarsFromHModel(Info['freshInfo']['freshModelPostDelete'],
                                      Kmax=Kmax,
                                      figH=pylab.subplot(3,2,4))
    pylab.title('Fresh Model CLEAN')


    bnpy.viz.BarsViz.plotBarsFromHModel(Info['xInfo']['xbigModelInit'],
                                      Kmax=Kmax,
                                      figH=pylab.subplot(3,2,5))
    pylab.title('Expanded INIT')
    bnpy.viz.BarsViz.plotBarsFromHModel(Info['xInfo']['xbigModelRefined'],
                                      Kmax=Kmax,
                                      figH=pylab.subplot(3,2,6))
    pylab.title('Expanded CLEAN')
    figID.tight_layout()
    if savepath is not None:
      savefile = os.path.join(savepath, 'topics.png')
      pylab.savefig(savefile, bbox='tight')
  
  #################################################### ELBO traces
  figID, ax = pylab.subplots(3, 1, sharex=1, figsize=(6.0, 10.0))
  pylab.subplot(3,1,1)
  plotELBOTraces(targetData, 'target D=%d' % (targetData.nDoc), model, Info)
  pylab.subplot(3,1,2)
  plotELBOTraces(holdData, 'heldout D=%d' % (holdData.nDoc), model, Info)
  pylab.subplot(3,1,3)
  plotELBOTraces(Data, 'all D=%d' % (Data.nDoc), model, Info)
  pylab.legend(loc='lower right')
 
  pylab.show(block=False)
  if savepath is None:
    keypress = raw_input('Press any key>>')
  else:
    savefile = os.path.join(savepath, 'traceELBO.png')
    pylab.savefig(savefile, bbox='tight')
    


def pprint_accept_decision(Data, dataName, model, xmodel, Info):
  curELBO = model.calc_evidence(Data)
  propELBO = xmodel.calc_evidence(Data)
  fresh0ELBO = Info['freshInfo']['freshModelInit'].calc_evidence(Data)
  freshELBO = Info['freshInfo']['freshModelPostDelete'].calc_evidence(Data)

  print 'Using %s, should we accept? ' % (dataName)
  print '  fresh init? %d' % (fresh0ELBO >= curELBO)
  print '  fresh post? %d' % (freshELBO >= curELBO)
  print '      expand? %d' % (propELBO >= curELBO)


########################################################### ELBO trace plots
###########################################################
def plotELBOTraces(Data, title, *args, **kwargs):
  Colors = ['b', 'r', 'g', 'm', 'c']
  Traces = calc_all_ELBO_traces(Data, *args, **kwargs)
  keys = sorted(Traces.keys())
  ymin = np.inf
  ymax = -1*np.inf
  for tID, key in enumerate(keys):
    color = Colors[tID]
    pylab.plot( Traces[key], color + '.-', markersize=15, 
                                           markeredgecolor=color, label=key)
    initELBO = Traces[key][0]*np.ones(Traces[key].size)
    pylab.plot( initELBO, color + '--')
    ymin = np.minimum(ymin, Traces[key].min())
    ymax = np.maximum(ymax, Traces[key].max())
    if key == 'freshModel':
      freshColor = color
    if key == 'expandModel':
      xColor = color
  pylab.title(title, fontsize=14)
  pylab.xlim([-0.5, Traces[key].size-.5])
  pylab.xticks(np.arange(0, Traces[key].size))
  B = 0.05 * (ymax - ymin)
  pylab.ylim([ymin-B, ymax+B])

  freshAccept = np.flatnonzero(Traces['freshModel'] >= Traces['origModel'])
  if len(freshAccept) > 0:
    pylab.plot( freshAccept, Traces['freshModel'][freshAccept],
                freshColor + '*', markersize=15, markeredgecolor=freshColor)

  xAccept = np.flatnonzero(Traces['expandModel'] >= Traces['origModel'])
  if len(xAccept) > 0:
    pylab.plot( xAccept, Traces['expandModel'][xAccept],
                xColor + '*', markersize=15, markeredgecolor=xColor)


def calc_all_ELBO_traces(Data, model, Info, **kwargs):
  xInfo = Info['xInfo']
  freshInfo = Info['freshInfo']
  Traces = dict()
  Traces['origModel'] = calc_ELBO_trace(Data, model, **kwargs)
  Traces['freshModel'] = calc_ELBO_trace(Data, freshInfo['freshModelPostDelete'], **kwargs)
  Traces['expandModel'] = calc_ELBO_trace(Data, xInfo['xbigModelRefined'], **kwargs)
  return Traces
  

def calc_ELBO_trace(Data, model, nSteps=10):
  traceELBO = np.zeros(nSteps)
  for step in xrange(nSteps):
    LP = model.calc_local_params(Data)
    SS = model.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
    traceELBO[step] = model.calc_evidence(SS=SS)
    if step < nSteps - 1:
      model.update_global_params(SS)
  return traceELBO


########################################################### Model creation
###########################################################
def InitModelWithOneTopic(Data, aModel='HDPStickBreak'):
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, K=1, initname='randomfromprior',
                                  seed=0)
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)
  return hmodel

def InitModelWithTrueTopics(Data, aModel='HDPStickBreak'):
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')
  for _ in range(5):
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
  return hmodel

def InitModelWithTrueTopicsButMissingOne(Data, kmissing=0, aModel='HDPStickBreak'):
  aDict = dict(alpha0=5.0, gamma=0.5)
  oDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel('VB', aModel, 'Mult', 
                                          aDict, oDict, Data)
  hmodel.init_global_params(Data, initname='trueparams')

  # Remove specific comp from the model
  if type(kmissing) == str:
    kmissing = int(kmissing.split('-')[1])
  LP = hmodel.calc_local_params(Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  SS.removeComp(kmissing)
  hmodel.obsModel.update_global_params(SS)

  beta = OptimHDPSB._v2beta(hmodel.allocModel.rho)
  newbeta = np.hstack([beta[:kmissing], beta[kmissing+1:]])
  hmodel.allocModel.set_global_params(K=SS.K, beta=newbeta[:-1])

  for _ in range(5):
    LP = hmodel.calc_local_params(Data)
    SS = hmodel.get_global_suff_stats(Data, LP)
    hmodel.update_global_params(SS)
  return hmodel


########################################################### Data loading
###########################################################
def loadData(dataName):
  if dataName == 'BarsK10V900':  
    Data = loadBars()
  elif dataName.lower().count('nips'):
    Data = loadNIPS()
  return Data

def loadBars():
  import BarsK10V900
  Data = BarsK10V900.get_data(nDocTotal=2000, nWordsPerDoc=250)
  return Data

def loadNIPS():
  os.environ['BNPYDATADIR'] = '/data/NIPS/'
  if not os.path.exists(os.environ['BNPYDATADIR']):
    os.environ['BNPYDATADIR'] = '/data/liv/liv-x/topic_models/data/nips/'
  sys.path.append(os.environ['BNPYDATADIR'])
  import NIPSCorpus
  Data = NIPSCorpus.get_data()
  return Data


########################################################### main
###########################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', default='BarsK10V900')
  parser.add_argument('--selectName', default='uniform')
  parser.add_argument('--initName', default='K1')
  parser.add_argument('--seed', type=int, default=0)
  args, unkList = parser.parse_known_args()
  kwargs = bnpy.ioutil.BNPYArgParser.arglist_to_kwargs(unkList)

  RunBirthMoveDemo(args.data, args.initName, args.selectName, args.seed, **kwargs)
