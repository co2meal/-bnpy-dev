import argparse
import os
from distutils.dir_util import mkpath

from VizBirthOnRealData import RunBirthMoveDemo

saveroot = '/data/liv/liv-x/topic_models/birth-results/'

########################################################### main
###########################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', default='BarsK10V900')
  parser.add_argument('--initName', default='K1')
  parser.add_argument('--nTask', type=int, default=1)
  parser.add_argument('--targetCompFrac', type=float, default=0)

  parser.add_argument('--creationRoutine', default='randexamples')
  parser.add_argument('--creationNumIters', type=int, default=0)
  parser.add_argument('--refineNumIters', type=int, default=0)
  parser.add_argument('--Kfresh', type=int, default=5)
  args, unkList = parser.parse_known_args()

  kwargs = dict()
  if  args.targetCompFrac == 0:
    kwargs['selectName'] = 'none'
  else:
    kwargs['selectName'] = 'sizebiased'
  kwargs['targetCompFrac'] = args.targetCompFrac
  kwargs['Kfresh'] = args.Kfresh
  kwargs['creationRoutine'] = args.creationRoutine
  kwargs['creationNumIters'] = args.creationNumIters
  kwargs['refineNumIters'] = args.refineNumIters

  jobname = 'targetFrac%.2f-%s-Kfresh%d-fIters%d-xIters%d'
  jobname = jobname % (args.targetCompFrac, 
                       args.creationRoutine,
                       args.Kfresh,
                       args.creationNumIters,
                       args.refineNumIters)

  savedir = os.path.join(saveroot, args.data + '-' + args.initName)
  savedir = os.path.join(savedir, jobname)

  for task in xrange(1,args.nTask+1):
    savepath = os.path.join(savedir, str(task))
    print savepath
    mkpath(savepath)
    RunBirthMoveDemo(args.data, initName=args.initName, seed=task, 
                           savepath=savepath, **kwargs)
