import os
import joblib
import numpy as np
import argparse
import time

os.environ['BNPYDATADIR'] = '/data/liv/liv-x/topic_models/data/wiki/'
import bnpy
from bnpy.allocmodel.admix import OptimizerForMAPDocTopicSticks as MAPOptim

dumppath = '/data/liv/liv-x/bnpy/local/dump/'


########################################################### Main
###########################################################
def RunManyRestarts(Dchunk, hmodel, prevLP=None, 
                            docID=0, nRestarts=2, Niters=100, 
                            convThrLP=1e-5, doSave=0,
                            savefilename='ManyRestarts.dump'):
  if Dchunk.nDoc > 1:
    Dchunk = Dchunk.select_subset_by_mask([docID])

  topicPriorVec = np.ones(hmodel.obsModel.K)

  ## From Scratch
  initDTC = np.zeros((1,hmodel.obsModel.K))
  ScratchInfo = traceLPInference(Dchunk, hmodel, initDTC, methodLP='scratch',
                                     Niters=Niters,
                                     convThrLP=convThrLP)

  ## From piMAP
  initDTC = np.zeros((1,hmodel.obsModel.K))
  MAPInfo = traceLPInference(Dchunk, hmodel, initDTC, methodLP='piMAP',
                                     Niters=Niters,
                                     convThrLP=convThrLP)

  print '-------------------------- logitMAP restarts'
  MAPRestarts = list()
  stime = time.time()
  for task in xrange(nRestarts):
    initEta = initEta_random(Dchunk, hmodel, topicPriorVec, seed=task)
    Info = traceLPInference(Dchunk, hmodel, initDTC, methodLP='piMAP-sneaky',
                        Niters=Niters, initEta=initEta[np.newaxis,:],
                        convThrLP=convThrLP)
    MAPRestarts.append(Info)
    if (task+1) % 10 == 0:
      etime = time.time() - stime
      print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)
  # Final status update
  etime = time.time() - stime
  if (task+1) % 10 != 0:
    print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)



  ## From Memo
  if prevLP is not None:
    prevDTC = prevLP['DocTopicCount'][docID,:][np.newaxis,:].copy()
    MemoInfo = traceLPInference(Dchunk, hmodel, prevDTC, methodLP='memo',
                                     Niters=Niters,
                                     convThrLP=convThrLP)

  print '-------------------------- Random restarts'
  ## From Random
  RandomInfo = list()  
  stime = time.time()
  for task in xrange(nRestarts):
    initDTC = initDTC_random(Dchunk, hmodel, topicPriorVec, seed=task)
    Info = traceLPInference(Dchunk, hmodel, initDTC, methodLP='memo',
                        Niters=Niters,
                        convThrLP=convThrLP)
    RandomInfo.append(Info)
    if (task+1) % 10 == 0:
      etime = time.time() - stime
      print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)

  # Final status update
  etime = time.time() - stime
  if (task+1) % 10 != 0:
    print '%.3f sec | %d/%d' % (etime, task+1, nRestarts)


  ## From perturbed memoized counts
  if prevLP is not None:
    print '-------------------------- Perturb restarts'
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
  FinalResults['MAP'] = MAPInfo
  FinalResults['MAPRestarts'] = MAPRestarts
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

    
    Data = Dchunk
    LP = hmodel.calc_local_params(Data, None, doInPlaceLP=0, methodLP='scratch',
                                              nCoordAscentFromScratchLP=1)
    Ld = LP['expEloglik']
    Xd = Data.word_count
    avec = hmodel.allocModel.topicPrior1
    bvec = hmodel.allocModel.topicPrior0

    datafile = dpath.replace('.dump', '-DataAndModel.dump')
    joblib.dump(dict(Data=Data, model=hmodel, Xd=Xd,
                     Ld=Ld, avec=avec, bvec=bvec),
                datafile)
    print 'Wrote to:', datafile
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

def initEta_random(Dchunk, hmodel, topicPriorVec, seed=0):
  PRNG = np.random.RandomState(seed)
  initPi = PRNG.dirichlet(np.hstack([topicPriorVec, 0.01]))
  return MAPOptim.invsigmoid( MAPOptim._beta2v(initPi))

########################################################### Run inference
###########################################################
def traceLPInference(Dchunk, hmodel, initDTC, Niters=100, convThrLP=1e-5,
                                              initEta=None, methodLP='memo'):
  initMAPscore = None
  initMAPPi = None
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

    LP = hmodel.calc_local_params(Dchunk, dict(DocTopicCount=DTC, expElogpi=initEta),
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

    if 'initMAPscore' in LP:
      initMAPscore = LP['initMAPscore'][0]
      initMAPPi = LP['initMAPPi'][0]

  Xd = Dchunk.word_count
  Ld = LP['expEloglik']
  pid = LP['expElogpi'][0,:]
  eta = MAPOptim.pi2eta(pid)
  MAPscore = MAPOptim.objFunc_unconstrained(eta, Xd, Ld, hmodel.allocModel.topicPrior1,
                                          hmodel.allocModel.topicPrior0)
 
  pi2 = MAPOptim.eta2pi(eta)
  if not np.allclose(pi2, LP['expElogpi'][0,:], rtol=0.01):
    print 'WHOA! transform invertability is in question!'
    from IPython import embed; embed()

  # Pack up and go home
  iters = np.flatnonzero(elbo)
  elbo = elbo[iters]
  return dict(MAPscore=MAPscore,
              finalPi=MAPOptim.eta2pi(eta),
              initMAPscore=initMAPscore, 
              initMAPPi=initMAPPi,
              DocTopicCount=LP['DocTopicCount'], elbo=elbo, iters=iters+1)

########################################################### Script Main
###########################################################

def RunSciK200_FromSpectral(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  global dumppath
  DUMP = joblib.load(os.path.join(dumppath, 'MemoLPELBODrop.dump'))
  hmodel = DUMP['hmodel']
  Dchunk = DUMP['Dchunk']
  prevLP = dict(DocTopicCount=DUMP['prevDocTopicCount'])
  doSave=1
  savefilename='SciK200-doc%03d.dump' % (docID)
  RunManyRestarts(Dchunk, hmodel, prevLP, docID, nRestarts, Niters, 
                          convThrLP, doSave, savefilename)


def RunSciK200_FromRandExamples(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  jobpath = '/data/liv/liv-x/topic_models/results/bnpy/science/HDPStickBreak/Mult/moVB/static-K200-randexamples/2/'
  hmodel = bnpy.load_model(jobpath)

  global dumppath
  DUMP = joblib.load(os.path.join(dumppath, 'MemoLPELBODrop.dump'))
  Dchunk = DUMP['Dchunk']
  doSave=1
  savefilename='SciK200-RX-doc%03d.dump' % (docID)
  RunManyRestarts(Dchunk, hmodel, None, docID, nRestarts, Niters, 
                          convThrLP, doSave, savefilename)

def RunWikiK50_FromRandExamples(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  jobpath = '/data/liv/liv-x/topic_models/results/bnpy/wiki/HDPStickBreak/Mult/moVB/static-K50-randexamples/1/'
  hmodel = bnpy.load_model(jobpath)

  import wiki
  Data = wiki.get_data()
  
  doSave=1
  savefilename='WikiK50-RX-doc%03d.dump' % (docID)
  RunManyRestarts(Data, hmodel, None, docID, nRestarts, Niters, 
                          convThrLP, doSave, savefilename)

def RunBarsK20(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  jobpath = '/data/liv/liv-x/topic_models/results/bnpy/BarsK10V900/HDPStickBreak/Mult/moVB/static-K20-randomfromprior/1/'
  hmodel = bnpy.load_model(jobpath)
  import BarsK10V900
  Data = BarsK10V900.get_data()
  doSave=1
  savefilename='BarsK20-RX-doc%03d.dump' % (docID)
  RunManyRestarts(Data, hmodel, None, docID, nRestarts, Niters, 
                          convThrLP, doSave, savefilename)

def RunBarsTrue(docID=0, nRestarts=2, Niters=100, convThrLP=1e-5, doSave=0):
  jobpath = '/data/liv/liv-x/topic_models/results/bnpy/BarsK10V900/HDPStickBreak/Mult/moVB/static-Ktrue/1/'
  hmodel = bnpy.load_model(jobpath)
  import BarsK10V900
  Data = BarsK10V900.get_data()
  doSave=1
  savefilename='BarsTrue-RX-doc%03d.dump' % (docID)
  RunManyRestarts(Data, hmodel, None, docID, nRestarts, Niters, 
                          convThrLP, doSave, savefilename)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('scenario', type=str)
  parser.add_argument('docID', type=int)
  parser.add_argument('--nRestarts', type=int, default=1)
  parser.add_argument('--Niters', type=int, default=5)
  parser.add_argument('--convThrLP', type=float, default=1e-6)
  parser.add_argument('--doSave', type=int, default=1)
  args = parser.parse_args()

  if args.scenario == 'science-K200-spectral':
    RunFunc = RunSciK200_FromSpectral
  elif args.scenario == 'wiki-K50-randexamples':
    RunFunc = RunWikiK50_FromRandExamples
  elif args.scenario == 'bars-K20':
    RunFunc = RunBarsK20
  elif args.scenario == 'bars-true':
    RunFunc = RunBarsTrue
  else:
    RunFunc = RunSciK200_FromRandExamples
  
  RunFunc(args.docID, 
                 nRestarts=args.nRestarts,
                 Niters=args.Niters,
                 convThrLP=args.convThrLP,
                 doSave=args.doSave)


if __name__ == '__main__':
  main()
