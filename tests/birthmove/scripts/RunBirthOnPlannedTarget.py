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

import bnpy.ioutil.BNPYArgParser
from bnpy.birthmove.BirthProposalError import BirthProposalError
from bnpy.birthmove import TargetPlanner, TargetDataSampler, BirthLogger
from bnpy.birthmove import TargetPlannerWordFreq
from bnpy.allocmodel.admix import OptimizerForHDPStickBreak as OptimHDPSB
from bnpy.birthmove import BirthMove, BirthRefine

import MakeTargetPlots as MTP
import HTMLMakerForTarget as HTMLMaker
from RunTargetSampler import LoadModelAndData

from HTMLMakerForBirth import MakeHTMLForBirth

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
               cleanupDeleteNumIters=3,
               creationNumIters=0,
               expandOrder='expandThenRefine',
               expandAdjustSuffStats=1,
               refineNumIters=50,
               cleanupDeleteEmpty=0,
               cleanupDeleteToImprove=1,
               cleanupDeleteToImproveFresh=0,
               cleanupDeleteViaLP=1,
               cleanupMinSize=50,
                )


def RunCurrentModelOnTargetData(outPath, model, bigSS, bigData, Plan,
                                         seed=0, **kwargs):
  ''' Play current model forward on the targeted dataset
  '''
  BirthArgs = dict(**BirthArgsIN)
  BirthArgs.update(**kwargs)
  BirthArgs['randstate'] = np.random.RandomState(seed)
  BirthArgs['seed'] = seed

  fwdmodel = model.copy()
  xmodel, _, _, Info = BirthRefine.refine_expanded_model_with_VB_iters(
                                          fwdmodel, Plan['Data'],
                                          xbigSS=bigSS, **BirthArgs)

  Results = MakeLeanSaveInfo(Info)
  outPath = os.path.join(outPath, 'Orig.dump')
  joblib.dump(Results, outPath)
  print '... ORIG done. Wrote all results to file: ', outPath
  return Results


def RunBirthOnTargetData(outPath, model, bigSS, bigData, Plan,
                                         seed=0, **kwargs): 
  BirthArgs = dict(**BirthArgsIN)
  BirthArgs.update(**kwargs)
  BirthArgs['randstate'] = np.random.RandomState(seed)
  BirthArgs['seed'] = seed

  assert bigData.nDoc == bigSS.nDoc

  if BirthArgs['creationRoutine'] == 'xspectral':
    Q = bigData.to_wordword_cooccur_matrix()
  else:
    Q = None

  xmodel, xSS, Info = BirthMove.run_birth_move(model, bigSS, Plan['Data'],
                                                           Q=Q,
                                                           **BirthArgs)
  Results = MakeLeanSaveInfo(Info, doTraceBeta=1)

  outPath = os.path.join(outPath, 'Birth.dump')
  joblib.dump(Results, outPath)
  print '... BIRTH done. Wrote all results to file: ', outPath
  return Results

########################################################### utils
###########################################################
def MakeLeanSaveInfo(Info, doTraceBeta=0):
  Results = dict(traceELBO=Info['traceELBO'])
  if doTraceBeta:
    Q = dict( traceBeta=Info['traceBeta'],
              traceN=Info['traceN']
             )
    Results.update(Q)
  if 'xbigModelInit' in Info:
    Q = dict( initTopics=makeTopics(Info['xbigModelInit']),
              finalTopics=makeTopics(Info['xbigModelRefined']),
            )
    if 'xbigModelPostDelete' in Info:
      Q['cleanupTopics'] = makeTopics(Info['xbigModelPostDelete'])
      Q['ELBOPostDelete'] = Info['ELBOPostDelete']
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

########################################################### main
###########################################################
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', default='BarsK10V900')
  parser.add_argument('initName', default='K1')
  parser.add_argument('selectName', type=str, default='none')
  parser.add_argument('jobName', type=str, default='')
  parser.add_argument('--planID', type=int, default=1)
  parser.add_argument('--savepath', type=str, default=None)
  parser.add_argument('--nTrial', type=int, default=1)

  args, unkList = parser.parse_known_args()
  kwargs = bnpy.ioutil.BNPYArgParser.arglist_to_kwargs(unkList)

  targetpath = os.path.join(args.savepath, args.data, args.initName,
                                              args.selectName,
                                              str(args.planID), 'Plan.dump')

  args.savepath = os.path.join(args.savepath, args.data, args.initName,
                                              args.selectName, 
                                              str(args.planID),
                                              args.jobName)
  mkpath(args.savepath)


  Data, model, SS, LP, cachefile = LoadModelAndData(args.data, args.initName)
  Plan = joblib.load(targetpath)

  for trial in xrange(args.nTrial):
    logfile = os.path.join(args.savepath, str(trial+1), 'log.txt')
    BirthLogger.configure(logfile, doSaveToDisk=0, doWriteStdOut=1)

    outpath = os.path.join(args.savepath, str(trial+1))
    mkpath(outpath)
    CurResults = RunCurrentModelOnTargetData(outpath, model, SS, Data, Plan,
                                          seed=trial, **kwargs)

    BirthResults = RunBirthOnTargetData(outpath, model, SS, Data, Plan,
                                          seed=trial, **kwargs)

    MakeHTMLForBirth(args.savepath, BirthResults, CurResults, trial+1)

if __name__ == '__main__':
  main()