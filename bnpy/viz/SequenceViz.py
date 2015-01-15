'''
SequenceViz.py

Visualizes sequential data as a plot of colored bars of the estimated state
as calculated by the Viterbi algorithm.  Displays a separate grid for each
jobname specified where each row corresponds to a sequence and each column
to a taskid.  By default, the final lap from each taskid is used; however,
a different lap can be specified with the --lap parameter.

Note that this script requires you
to have made a run of bnpy with the custom func learnalg/extras/XViterbi.py
running (use --customFuncPath path/goes/here/XViterbi.py)


Example, showing the final lap:

python SequenceViz.py --dataset MoCap6 --jobnames defaultjob --taskids 1,3,6
                        --sequences 1,2,4,6

Example, showing the 5th lap for two different jobnames:

python SequenceViz.py --dataset MoCap6 --jobnames defaultjob,EM --taskids 1
                        --lap 5 --sequences 1,2,4,6
'''

import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import imp
import sys
import argparse

from bnpy.ioutil import BNPYArgParser

def plotSingleJob(dataset, jobname, taskids, lap, sequences, 
                  showELBOInTitle=False,
                  dispTrue = True,
                  aspectFactor=4.0,
                 ):
  '''
  Returns the array of data corresponding to a single sequence to display

  If dispTrue = True, the true labels will be shown underneath the
    estimated labels
  '''
  jobpath = os.path.join( os.path.expandvars('$BNPYOUTDIR'), dataset, jobname)
  if type(taskids) == str:
    taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)
  elif type(taskids) == int:
    taskids = [str(taskids)]
  sequences = np.asarray(sequences, dtype=np.int32)

  #Load in the data module
  datamod = imp.load_source(dataset,
                            os.path.expandvars('$BNPYDATADIR/'+dataset+'.py'))
  data = datamod.get_data()

  # Determine the maximum length among any of the sequences to be plotted
  Ts = data.doc_range[sequences+1] - data.doc_range[sequences]
  maxT = np.max(Ts)

  # Define the number of pixels used by vertical space of figure
  NUM_STACK = (maxT / aspectFactor) #/ len(sequences)
  if dispTrue:
    NUM_STACK /= 2

  f, axes = plt.subplots(len(sequences), len(taskids),
                         sharex='col', sharey='row')

  # For singleton case, make sure that axes is index-able
  if len(sequences) == 1 and len(taskids) == 1:
    axes = [axes]

  for tt, taskidstr in enumerate(taskids):
    path = os.path.join(jobpath, taskidstr) + os.path.sep

    #Figure out which lap to use
    if lap == 'final':
      lapsFile = open(path+'laps.txt')
      curLap = lapsFile.readlines()
      curLap = float(curLap[-1])
      lapsFile.close()
    else:
      curLap = int(lap)

    if showELBOInTitle:
      Kvals = np.loadtxt(os.path.join(path, 'K.txt'))
      ELBOscores = np.loadtxt(os.path.join(path, 'evidence.txt'))
      laps = np.loadtxt(os.path.join(path, 'laps.txt'))
      loc = np.flatnonzero(laps == curLap)
      ELBO = ELBOscores[loc]
      Kfinal = Kvals[loc]

    #Load in the saved data from $BNPYOUTDIR
    filename = 'Lap%08.3fMAPStateSeqsAligned.mat' % curLap

    zHatBySeq = scipy.io.loadmat(path + filename)
    zHatBySeq = zHatBySeq['zHatBySeqAligned'][0]
    hammingFile = open(path+'hamming-distance.txt', 'r')
    hammingDists = hammingFile.readlines()
    hammingDists = [float(x) for x in hammingDists]
    hammingFile.close()

    # Find maximum number of states we need to display
    Kmax = np.max([zHatBySeq[i].max() for i in xrange(data.nDoc)])
    Kmax = np.maximum(data.TrueParams['Z'].max(), Kmax)

    # In case there's only one sequence, make sure it's index-able
    if len(np.shape(zHatBySeq)) == 1:
      zHatBySeq = [zHatBySeq]

    for ii, seqNum in enumerate(sequences):
      image = np.tile(zHatBySeq[seqNum], (NUM_STACK, 1))

      #Add the true labels to the image (if they exist)
      if ( (data.TrueParams is not None) and ('Z' in data.TrueParams)
           and (dispTrue):
        start = data.doc_range[seqNum]
        stop = data.doc_range[seqNum+1]
        image = np.vstack((image, np.tile(data.TrueParams['Z'][start:stop],
                                          (NUM_STACK, 1))))

      if len(sequences) == 1 or len(taskids) == 1:
        cur_ax = axes[ii+tt]
      else:
        cur_ax = axes[ii,tt]
    
      cur_ax.imshow(image, interpolation='nearest',
                           vmin=0, vmax=Kmax,
                           cmap='Set1')
      if tt == 0:
        cur_ax.set_ylabel('Seq. %d' % sequences[ii], fontsize=13)

      if ii == 0:
        if showELBOInTitle:
          cur_ax.set_title('ELBO: %.3f  K=%d' % (ELBO, Kfinal))
        else:
          cur_ax.set_title('Task %s' % taskidstr)
      cur_ax.set_xlim([0, maxT])
      cur_ax.set_ylim([0, image.shape[0]])
      cur_ax.set_yticks([])
      
      # ... end loop over sequences    

    #f.suptitle(jobname+', lap = '+lap, fontsize = 18)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset')
  parser.add_argument('--jobnames')
  parser.add_argument('--taskids', type=str, default='1',
         help="int ids of tasks (trials/runs) to plot from given job." \
              + " Example: '4' or '1,2,3' or '2-6'.")

  parser.add_argument('--lap', default = 'final')
  parser.add_argument('--sequences', default='1')
  args = parser.parse_args()

  if args.jobnames is None:
    raise ValueError('BAD ARGUMENT: String jobname.\n' 
                     + 'Usage: SequenceViz --dataset Name --jobnames a,b,c')
  jobs = args.jobnames.split(',')

  sequences = np.asarray([x for x in args.sequences.split(',')], dtype=np.int32)

  for job in jobs:
    plotSingleJob(dataset = args.dataset,
                  jobname = job,
                  taskids = args.taskids,
                  lap = args.lap,
                  sequences =  sequences,
                  dispTrue = True)
    

  plt.show()
