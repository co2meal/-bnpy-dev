import argparse
import sys
import os
import bnpy
import time

pathparts = os.path.abspath(__file__).split(os.path.sep)
parentpath = os.path.sep.join(pathparts[:-2])
sys.path.append(parentpath)
import RunWithProfiler

def run_benchmark(args):
  # Load data
  datamod = __import__(args.dataName,fromlist=[])
  Data = datamod.get_data(dataseed=0)
  DataIterator = datamod.get_minibatch_iterator(dataseed=0, nBatch=args.nBatch,   nLap=args.nLap)

  # Create model
  algName = 'moVB'
  aName = args.allocModelName
  oName = 'Mult'
  aPriorDict = dict(gamma=0.5, alpha0=5)
  oPriorDict = {'lambda':0.1}
  hmodel = bnpy.HModel.CreateEntireModel(algName, aName, oName, aPriorDict,   oPriorDict, Data)
  hmodel.init_global_params(Data, K=args.K, initname='randexamples')

  # Run local steps for benchmarking speed
  lapFrac = 0 
  stime = time.time()
  while DataIterator.has_next_batch():
    Dchunk = DataIterator.get_next_batch()

    curstime = time.time()  
    LP = hmodel.calc_local_params(Dchunk,
              nCoordAscentItersLP=args.nCoordAscentItersLP,
              convThrLP=args.convThrLP,
              do_nObs2nDoc_fast=args.do_nObs2nDoc_fast,
              )
    ttime = time.time()
    lapFrac += 1.0 / args.nBatch
    print "%6.3f laps | %7.3f sec overall | %7.3f sec on current step" % (lapFrac, ttime - stime, ttime - curstime)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=int, default=0)
  args, unk = parser.parse_known_args()
  if args.profile:
    RunWithProfiler.run_with_profiler(pyFunc, pyArgs)
  else:

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--dataName', type=str, default='BarsK10V900')
    parser.add_argument('--allocModelName', type=str, default='HDPModel')
    parser.add_argument('--nLap', type=int, default=1)
    parser.add_argument('--nBatch', type=int, default=10)

    parser.add_argument('--convThrLP', type=float, default=0.01)
    parser.add_argument('--nCoordAscentItersLP', type=int, default=20)
    parser.add_argument('--do_nObs2nDoc_fast', type=int, default=0)
    args, unk = parser.parse_known_args()

    run_benchmark(args)