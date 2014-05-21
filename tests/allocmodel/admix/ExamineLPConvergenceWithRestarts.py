import os
import joblib
import numpy as np
import argparse
import time

import bnpy


dumppath = '/data/liv/liv-x/bnpy/local/dump/'

def plotELBOForManyRestarts(RunInfo, SInfo=None, MInfo=None, block=False):
  from matplotlib import pylab
  pylab.figure()
  for Info in RunInfo:
    pylab.plot( Info['iters'], Info['elbo'], '.--', markersize=3, linewidth=1)
  if SInfo is not None:
    pylab.plot( SInfo['iters'], SInfo['elbo'], '.-', markersize=3, linewidth=2, label='scratch')
  if MInfo is not None:
    pylab.plot( MInfo['iters'], MInfo['elbo'], '.-', markersize=3, linewidth=2, label='memo')
  pylab.legend()

  pylab.show(block=block)

def plotScatterELBOvsReconstructError(RunInfo, SInfo=None):
  from matplotlib import pylab
  pylab.figure()
  for Info in RunInfo:
    pylab.plot( Info['err'], Info['elbo'][-1], '.', markersize=8)
  pylab.show(block=False)

def loadDataAndRun(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  global dumppath
  DUMP = joblib.load(os.path.join(dumppath, 'MemoLPELBODrop.dump'))
  hmodel = DUMP['hmodel']
  Dchunk = DUMP['Dchunk']
  if Dchunk.nDoc > 1:
    Dchunk = Dchunk.select_subset_by_mask([docID])

  topicPriorVec = np.ones(hmodel.obsModel.K)

  ## From Scratch
  initDTC = np.zeros((1,hmodel.obsModel.K))
  LP, elbo, iters = traceLPInference(Dchunk, hmodel, initDTC, methodLP='scratch',
                                     Niters=Niters,
                                     convThrLP=convThrLP)
  Info = dict(elbo=elbo, iters=iters,
              DocTopicCount=LP['DocTopicCount'],
              err=calcReconstructError(Dchunk, hmodel, LP['DocTopicCount']))
  ScratchInfo = Info

  ## From Previous
  prevDTC = DUMP['prevDocTopicCount'][docID,:][np.newaxis,:].copy()
  LP, elbo, iters = traceLPInference(Dchunk, hmodel, prevDTC, methodLP='memo',
                                     Niters=Niters,
                                     convThrLP=convThrLP)
  Info = dict(elbo=elbo, iters=iters,
              DocTopicCount=LP['DocTopicCount'],
              err=calcReconstructError(Dchunk, hmodel, LP['DocTopicCount']))
  MemoInfo = Info

  
  ## From Random
  RunInfo = list()  
  stime = time.time()
  for task in xrange(nRestarts):
    initDTC = initDTC_random(Dchunk, hmodel, topicPriorVec, seed=task)
    LP, elbo, iters = traceLPInference(Dchunk, hmodel, initDTC,
                        Niters=Niters,
                        convThrLP=convThrLP)

    err = calcReconstructError(Dchunk, hmodel, LP['DocTopicCount'])
    RunInfo.append( dict(err=err,
                         elbo=elbo, iters=iters,
                         DocTopicCount=LP['DocTopicCount']))
    if (task+1) % 10 == 0:
      etime = time.time() - stime
      print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)




  ## Random perturb of Memoized counts
  print 'Perturbing!'
  PInfo = list()  
  stime = time.time()
  for task in xrange(nRestarts):
    prevDTC = DUMP['prevDocTopicCount'][docID,:][np.newaxis,:].copy()
    initDTC = initDTC_perturb(Dchunk, hmodel, prevDTC, seed=task)

    LP, elbo, iters = traceLPInference(Dchunk, hmodel, initDTC,
                        Niters=Niters, 
                        convThrLP=convThrLP)
    print elbo[0], elbo[-1]

    err = calcReconstructError(Dchunk, hmodel, LP['DocTopicCount'])
    PInfo.append( dict(err=err,
                         elbo=elbo, iters=iters,
                         DocTopicCount=LP['DocTopicCount']))
    if (task+1) % 10 == 0:
      etime = time.time() - stime
      print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

  # final status update
  etime = time.time() - stime
  print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

  if doSave:
    dpath = os.path.join( dumppath, 'ScienceK200-DocID%04d.dump' % (docID))
    joblib.dump(dict(RunInfo=RunInfo, PInfo=PInfo, 
                     MemoInfo=MemoInfo,
                     ScratchInfo=ScratchInfo),
                dpath)
    print 'Wrote to:', dpath

  return RunInfo, ScratchInfo, MemoInfo, PInfo

def calcReconstructError(Dchunk, hmodel, DTC):
  x = Dchunk.get_wordfreq_for_doc(0)
  L = hmodel.obsModel.getElogphiMatrix().T.copy()
  L -= L.max(axis=0)
  np.exp(L, out=L)
  L /= L.sum(axis=0)
  pi = DTC[0,:]
  return np.sum(np.square( x - np.dot(L,pi)))


def initDTC_perturb(Dchunk, hmodel, prevDTC, borrowFrac=0.50, M=20, seed=0):
  PRNG = np.random.RandomState(seed)

  smLocs = np.flatnonzero(prevDTC[0,:] < 1.0)
  M = np.minimum( smLocs.size, M)
  minLocs = PRNG.choice(smLocs, M)
  borrowMass = borrowFrac * prevDTC.sum()

  initDTC = prevDTC * (1-borrowFrac)
  initDTC[0, minLocs] += borrowMass
  return np.squeeze(initDTC)

def initDTC_random(Dchunk, hmodel, topicPriorVec, seed=0):
  PRNG = np.random.RandomState(seed)
  initPi = PRNG.dirichlet(topicPriorVec)
  initDTC = initPi * Dchunk.word_count.sum()
  return initDTC

def traceLPInference(Dchunk, hmodel, initDTC, Niters=100, convThrLP=1e-5,
                                              methodLP='memo'):
  elbo = np.zeros(Niters)
  for ii in xrange(Niters):
    if ii == 0:
      if initDTC.ndim == 1:
        DTC = initDTC[np.newaxis,:]
      elif initDTC.ndim == 2:
        DTC = initDTC
      else:
        raise ValueError('OH NO!')
    else:
      DTC = LP['DocTopicCount']
      methodLP = 'memo'

    LP = hmodel.calc_local_params(Dchunk, dict(DocTopicCount=DTC),
                                  methodLP=methodLP,
                                  doInPlaceLP=0,
                                  nCoordAscentItersLP=1,
                                  convThrLP=convThrLP)
    if LP['didConverge']:
      SS = hmodel.get_global_suff_stats(Dchunk, LP)
      elbo[ii] = hmodel.calc_evidence(Dchunk, SS, LP)
      break

    if (ii+1) % 5 == 0 or ii == 0:
      SS = hmodel.get_global_suff_stats(Dchunk, LP)
      elbo[ii] = hmodel.calc_evidence(Dchunk, SS, LP)



  iters = np.flatnonzero(elbo)
  elbo = elbo[iters]
  return LP, elbo, iters+1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('docID', type=int)
  parser.add_argument('--nRestarts', type=int, default=100)
  parser.add_argument('--Niters', type=int, default=500)
  parser.add_argument('--convThrLP', type=float, default=1e-6)
  parser.add_argument('--doSave', type=int, default=1)
  args = parser.parse_args()

  loadDataAndRun(args.docID, 
                 nRestarts=args.nRestarts,
                 Niters=args.Niters,
                 convThrLP=args.convThrLP,
                 doSave=args.doSave)


if __name__ == '__main__':
  main()
