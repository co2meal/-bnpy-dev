import os
import joblib
import numpy as np
import argparse
import time

import bnpy

dumppath = '/data/liv/liv-x/bnpy/local/dump/'


########################################################### Main
###########################################################
def RunManyRestarts(Dchunk, hmodel, prevLP=None, 
                            docID=0, nRestarts=2, Niters=100, 
                            convThrLP=1e-5, doSave=0, savefilename='ManyRestarts.dump'):
  if Dchunk.nDoc > 1:
    Dchunk = Dchunk.select_subset_by_mask([docID])

  ## From Scratch
  initDTC = np.zeros((1,hmodel.obsModel.K))
  ScratchInfo = traceLPInference(Dchunk, hmodel, initDTC, methodLP='scratch',
                                     Niters=Niters,
                                     convThrLP=convThrLP)

  ## From Memo
  if prevLP is not None:
    prevDTC = prevLP['DocTopicCount'][docID,:][np.newaxis,:].copy()
    MemoInfo = traceLPInference(Dchunk, hmodel, prevDTC, methodLP='memo',
                                     Niters=Niters,
                                     convThrLP=convThrLP)

  ## From Random
  RandomInfo = list()  
  stime = time.time()
  topicPriorVec = np.ones(hmodel.obsModel.K)
  for task in xrange(nRestarts):
    initDTC = initDTC_random(Dchunk, hmodel, topicPriorVec, seed=task)
    Info = traceLPInference(Dchunk, hmodel, initDTC, methodLP='memo',
                        Niters=Niters,
                        convThrLP=convThrLP)
    RandomInfo.append(Info)
    if (task+1) % 10 == 0:
      etime = time.time() - stime
      print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

  ## From perturbed memoized counts
  if prevLP is not None:
    prevDTC = prevLP['DocTopicCount'][docID,:][np.newaxis,:].copy() 
    PerturbInfo3 = list()  
    stime = time.time()
    for task in xrange(nRestarts):
      initDTC = initDTC_perturb(Dchunk, hmodel, prevDTC, flag=3, seed=task)
      Info = traceLPInference(Dchunk, hmodel, initDTC, methodLP='memo',
                        Niters=Niters, 
                        convThrLP=convThrLP)
      PerturbInfo3.append(Info)
      if (task+1) % 10 == 0:
        etime = time.time() - stime
        print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

     
    PerturbInfo = list()  
    stime = time.time()
    for task in xrange(nRestarts):
      initDTC = initDTC_perturb(Dchunk, hmodel, prevDTC, flag=1, seed=task)
      Info = traceLPInference(Dchunk, hmodel, initDTC, methodLP='memo',
                        Niters=Niters, 
                        convThrLP=convThrLP)
      PerturbInfo.append(Info)
      if (task+1) % 10 == 0:
        etime = time.time() - stime
        print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

    PerturbInfo2 = list()  
    stime = time.time()
    for task in xrange(nRestarts):
      initDTC = initDTC_perturb(Dchunk, hmodel, prevDTC, flag=2, seed=task)
      Info = traceLPInference(Dchunk, hmodel, initDTC, methodLP='memo',
                        Niters=Niters, 
                        convThrLP=convThrLP)
      PerturbInfo2.append(Info)
      if (task+1) % 10 == 0:
        etime = time.time() - stime
        print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)



  # Final status update
  etime = time.time() - stime
  print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

  FinalResults = dict()
  FinalResults['Scratch'] = ScratchInfo
  FinalResults['Random'] = RandomInfo
  if prevLP is not None:
    FinalResults['Memo'] = MemoInfo
    FinalResults['Perturb'] = PerturbInfo
    FinalResults['PerturbMoreMass'] = PerturbInfo2  
    FinalResults['PerturbAddConst'] = PerturbInfo3  

  if doSave:
    if savefilename.count('%'):
      savefilename = savefilename % (docID)
    dpath = os.path.join(dumppath, savefilename)


    joblib.dump(FinalResults, dpath)
    print 'Wrote to:', dpath
  return FinalResults

########################################################### Doc-topic init
##########################################################
def initDTC_perturb(Dchunk, hmodel, prevDTC, borrowFrac=0.50, M=20, flag=1, seed=0):
  ''' Randomly sample M<=20 locations that have small mass,
       and add some mass "borrowed" from all large mass
  '''
  PRNG = np.random.RandomState(seed)
  smLocs = np.flatnonzero(prevDTC[0,:] < 1.0)
  M = np.minimum( smLocs.size, M)
  minLocs = PRNG.choice(smLocs, M)
  borrowMass = borrowFrac * prevDTC.sum()

  initDTC = prevDTC * (1-borrowFrac)

  if flag == 2:
    initDTC[0, minLocs] += borrowMass
  elif flag == 3:
    initDTC = prevDTC + PRNG.rand(prevDTC.size) * borrowMass  
  else:
    initDTC[0, minLocs] += borrowMass / M
  return np.squeeze(initDTC)

def initDTC_random(Dchunk, hmodel, topicPriorVec, seed=0):
  PRNG = np.random.RandomState(seed)
  initPi = PRNG.dirichlet(topicPriorVec)
  initDTC = initPi * Dchunk.word_count.sum()
  return initDTC


########################################################### Run inference
###########################################################
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
                                  nCoordAscentFromScratchLP=1,
                                  convThrLP=convThrLP)
    if LP['didConverge']:
      SS = hmodel.get_global_suff_stats(Dchunk, LP)
      elbo[ii] = hmodel.calc_evidence(Dchunk, SS, LP)
      break

    if (ii+1) % 5 == 0 or ii == 0:
      SS = hmodel.get_global_suff_stats(Dchunk, LP)
      elbo[ii] = hmodel.calc_evidence(Dchunk, SS, LP)

  # Pack up and go home
  iters = np.flatnonzero(elbo)
  elbo = elbo[iters]
  return dict(DocTopicCount=LP['DocTopicCount'], elbo=elbo, iters=iters+1)

########################################################### Script Main
###########################################################

def RunSciK200Restarts(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  global dumppath
  DUMP = joblib.load(os.path.join(dumppath, 'MemoLPELBODrop.dump'))
  hmodel = DUMP['hmodel']
  Dchunk = DUMP['Dchunk']
  prevLP = dict(DocTopicCount=DUMP['prevDocTopicCount'])
  doSave=1
  savefilename='SciK200-doc%03d.dump' % (docID)
  RunManyRestarts(Dchunk, hmodel, prevLP, docID, nRestarts, Niters, 
                          convThrLP, doSave, savefilename)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('docID', type=int)
  parser.add_argument('--nRestarts', type=int, default=1)
  parser.add_argument('--Niters', type=int, default=5)
  parser.add_argument('--convThrLP', type=float, default=1e-6)
  parser.add_argument('--doSave', type=int, default=1)
  args = parser.parse_args()

  RunSciK200Restarts(args.docID, 
                 nRestarts=args.nRestarts,
                 Niters=args.Niters,
                 convThrLP=args.convThrLP,
                 doSave=args.doSave)


if __name__ == '__main__':
  main()
