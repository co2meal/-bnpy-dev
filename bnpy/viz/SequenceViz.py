'''
SequenceViz.py

Visualizes sequential Data as a plot of colored bars of the estimated state
as calculated by the Viterbi algorithm.  Displays a separate grid for each
jobname specified where each row corresponds to a sequence and each column
to a taskid.  By default, the final lap from each taskid is used; however,
a different lap can be specified with the --lap parameter.

Note that this script requires you
to have made a run of bnpy with the custom func learnalg/extras/XViterbi.py
running (use --customFuncPath path/goes/here/XViterbi.py)


Example, showing the final lap:

python SequenceViz.py --Dataset MoCap6 --jobnames defaultjob --taskids 1,3,6
                        --sequences 1,2,4,6

Example, showing the 5th lap for two different jobnames:

python SequenceViz.py --Dataset MoCap6 --jobnames defaultjob,EM --taskids 1
                        --lap 5 --sequences 1,2,4,6
'''

import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import imp
import sys
import argparse

from bnpy.util.StateSeqUtil import convertStateSeq_MAT2list
from bnpy.ioutil import BNPYArgParser
from bnpy.viz.TaskRanker import rankTasksForSingleJobOnDisk

def plotSingleJob(dataset, jobname, taskids='1', lap='final',
                  sequences=[1],
                  showELBOInTitle=False,
                  dispTrue = True,
                  aspectFactor=4.0,
                  maxT=None,
                 ):
  '''
  Returns the array of Data corresponding to a single sequence to display

  If dispTrue = True, the true labels will be shown underneath the
    estimated labels
  '''
  # Make sequences zero-indexed
  if isinstance(sequences, str):
    sequences = np.asarray([int(x) for x in args.sequences.split(',')],
                           dtype=np.int32)
  sequences = np.asarray(sequences, dtype=np.int32)
  if np.min(sequences) < 1:
    raise ValueError('Sequences need to be one-index.\n'
                     + 'Valid values are 1,2,...N.')
  sequences -= 1

  # Load Data from its python module
  Datamod = imp.load_source(dataset,
                            os.path.expandvars('$BNPYDATADIR/'+dataset+'.py'))
  if dataset == 'SpeakerDiar':
    if len(sequences) > 1:
      raise ValueError('Joint modeling of several sequences makes no sense')
    Data = Datamod.get_data(meetingNum=sequences[0]+1)
    sequences[0] = 0
  else:
    Data = Datamod.get_data()

  # Determine the jobpath and taskids 
  jobpath = os.path.join(os.path.expandvars('$BNPYOUTDIR'), 
                         Datamod.get_short_name(), jobname)
  if type(taskids) == str:
    taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)
  elif type(taskids) == int:
    taskids = [str(taskids)]

  # Determine the maximum length among any of the sequences to be plotted
  if maxT is None:
    Ts = Data.doc_range[sequences+1] - Data.doc_range[sequences]
    maxT = np.max(Ts)
  
  # Define the number of pixels used by vertical space of figure
  NUM_STACK = int(np.ceil(maxT / float(aspectFactor)))
  if dispTrue:
    NUM_STACK /= 2

  f, axes = plt.subplots(len(sequences), len(taskids),
                         sharex='col', sharey='row')

  # For singleton case, make sure that axes is index-able
  if len(sequences) == 1 and len(taskids) == 1:
    axes = [axes]

  for tt, taskidstr in enumerate(taskids):
    if tt == 0 and taskidstr.startswith('.'):
      rankTasksForSingleJobOnDisk(jobpath)

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

      hdists = np.loadtxt(os.path.join(path, 'hamming-distance.txt'))
      hlaps = np.loadtxt(os.path.join(path, 'laps-saved-params.txt'))

      loc = np.argmin(np.abs(laps - curLap))
      ELBO = ELBOscores[loc]
      Kfinal = Kvals[loc]

      loc = np.argmin(np.abs(hlaps - curLap))
      hdist = hdists[loc]

    #Load in the saved Data from $BNPYOUTDIR
    filename = 'Lap%08.3fMAPStateSeqsAligned.mat' % curLap

    zHatBySeq = scipy.io.loadmat(path + filename)
    zHatBySeq = convertStateSeq_MAT2list(zHatBySeq['zHatBySeqAligned'])

    # Find maximum number of states we need to display
    Kmax = np.max([zHatBySeq[i].max() for i in xrange(Data.nDoc)])
    Kmax = np.maximum(Data.TrueParams['Z'].max(), Kmax)

    # In case there's only one sequence, make sure it's index-able
    for ii, seqNum in enumerate(sequences):
      image = np.tile(zHatBySeq[seqNum], (NUM_STACK, 1))

      #Add the true labels to the image (if they exist)
      if ((Data.TrueParams is not None)
           and ('Z' in Data.TrueParams)
           and dispTrue):
        start = Data.doc_range[seqNum]
        stop = Data.doc_range[seqNum+1]
        image = np.vstack((image, np.tile(Data.TrueParams['Z'][start:stop],
                                          (NUM_STACK, 1))))
      
      image = image[:, :maxT]
      if len(sequences) == 1 or len(taskids) == 1:
        cur_ax = axes[ii+tt]
      else:
        cur_ax = axes[ii,tt]
    
      cur_ax.imshow(image, interpolation='nearest',
                           vmin=0, vmax=Kmax,
                           cmap='Set1')
      if tt == 0:
        if len(sequences) > 4:
          cur_ax.set_ylabel('%d' % (seqNum+1), fontsize=13)
        else:
          cur_ax.set_ylabel('Seq. %d' % (seqNum+1), fontsize=13)

      if ii == 0:
        if showELBOInTitle:
          cur_ax.set_title('ELBO: %.3f  K=%d  dist=%.2f' % (ELBO, Kfinal,hdist))
        else:
          cur_ax.set_title('Task %s' % taskidstr)
      cur_ax.set_xlim([0, maxT])
      cur_ax.set_ylim([0, image.shape[0]])
      cur_ax.set_yticks([])
      # ... end loop over sequences    


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
                     + 'Usage: SequenceViz --Dataset Name --jobnames a,b,c')
  jobs = args.jobnames.split(',')

  for job in jobs:
    plotSingleJob(dataset = args.dataset,
                  jobname = job,
                  taskids = args.taskids,
                  lap = args.lap,
                  sequences = args.sequences,
                  dispTrue = True)
    

  plt.show()
