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
from bnpy.viz import TaskRanker

def plotSingleJob(dataset, jobname, taskids='1', lap='final',
                  sequences=[1],
                  showELBOInTitle=False,
                  dispTrue = True,
                  aspectFactor=4.0,
                  specialStateIDs=None,
                  seqNames=None,
                  cmap='Set1',
                  maxT=None,
                  title=''
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
  if taskids == 'best':
    ranks = TaskRanker.rankTasksForSingleJob(jobpath)
    taskids = [str(ranks[0])]
    print jobname, '    ', ranks[0]
  elif type(taskids) == str:
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
    try:
      filename = 'Lap%08.3fMAPStateSeqsAligned.mat' % curLap
      zHatBySeq = scipy.io.loadmat(path + filename)
  
      zHatBySeq = convertStateSeq_MAT2list(zHatBySeq['zHatBySeqAligned'])
    except IOError:
      filename = 'Lap%08.3fMAPStateSeqs.mat' % curLap
      zHatBySeq = scipy.io.loadmat(path + filename)
      zHatBySeq = convertStateSeq_MAT2list(zHatBySeq['zHatBySeq'])

    # TEMPORARY BY WILL
    '''
    from bnpy.util.StateSeqUtil import convertStateSeq_list2flat
    amod = scipy.io.loadmat(path + 'BestAllocModel.mat')
    pi = amod['transTheta']
    print 'diag mass = ', np.sum(pi[np.diag_indices(20)])
    print 'off-diag mass = ', np.sum(pi) - np.sum(pi[np.diag_indices(20)])
    pi /= np.sum(pi, axis=1)[:,np.newaxis]
    fig, ax = plt.subplots(1)
    ax.imshow(1.0-pi, interpolation='nearest', cmap='Greys',
              vmin=0.0, vmax=1.0)
    flat = convertStateSeq_list2flat(zHatBySeq,Data)
    #fig,ax = plt.subplots(1)
    #cnt = np.bincount(flat.astype(int))
    #ax.bar(np.arange(20), cnt)
    #print 'AVG CNT = ', np.average(cnt)

    # Estimate top-level beta
    from bnpy.util.StickBreakUtil import rho2beta
    omega = amod['omega']
    rho = amod['rho']
    fig,ax=plt.subplots(1)
    ax.bar(np.arange(21), rho2beta(rho))
    ax.set_title('beta')
    eta1 = omega * rho
    eta0 = omega * (1-rho)
    ax.set_ylim([0.0, 0.15])
    
    
    print np.sum(pi[np.diag_indices(20)])
    print jobname, '   ', ranks[0]
    print '--------'
    '''




    
    if specialStateIDs is not None:
      zHatBySeq = relabelAllSequences(zHatBySeq, specialStateIDs)

    # Find maximum number of states we need to display
    Kmax = np.max([zHatBySeq[i].max() for i in xrange(Data.nDoc)])
    hasGroundTruth = False
    if hasattr(Data, 'TrueParams') and 'Z' in Data.TrueParams:
      hasGroundTruth = True
      Kmax = np.maximum(Data.TrueParams['Z'].max(), Kmax)

    # In case there's only one sequence, make sure it's index-able
    for ii, seqNum in enumerate(sequences):
      image = np.tile(zHatBySeq[seqNum], (NUM_STACK, 1))

      #Add the true labels to the image (if they exist)
      if hasGroundTruth and dispTrue:
        start = Data.doc_range[seqNum]
        stop = Data.doc_range[seqNum+1]
        image = np.vstack((image, np.tile(Data.TrueParams['Z'][start:stop],
                                          (NUM_STACK, 1))))
      
      image = image[:, :maxT]
      if len(sequences) == 1 or len(taskids) == 1:
        cur_ax = axes[ii+tt]
      else:
        cur_ax = axes[ii,tt]
    
      if hasattr(cmap, 'N'):
        vmax = cmap.N
      else:
        vmax = Kmax
      cur_ax.imshow(image+.0001, interpolation='nearest',
                           vmin=0, vmax=vmax,
                           cmap=cmap)
      if tt == 0:
        if seqNames is not None:
          h = cur_ax.set_ylabel('%s' % (seqNames[ii]), fontsize=13)
          h.set_rotation(0)

        elif len(sequences) > 4:
          cur_ax.set_ylabel('%d' % (seqNum+1), fontsize=13)
        else:
          #cur_ax.set_ylabel('Seq. %d' % (seqNum+1), fontsize=13)
          pass

      if ii == 0:
        if showELBOInTitle:
          title += '\n' + 'ELBO: %.3f  K=%d  dist=%.2f' % (ELBO, Kfinal,hdist)

      cur_ax.set_title(title)
      cur_ax.set_xlim([0, maxT])
      cur_ax.set_ylim([0, image.shape[0]])
      cur_ax.set_yticks([])
      # ... end loop over sequences    
  return zHatBySeq

def relabelAllSequences(zBySeq, specialStateIDs):
  '''
      Returns
      -------
      zBySeq, relabelled so that each label in specialStateIDs 
              now corresponds to ids 0, 1, 2, ... L-1
              and all other labels not in that set get ids L, L+1, ...
  '''
  import copy
  zBySeq = copy.deepcopy(zBySeq)
  L = len(specialStateIDs)

  uniqueVals = []
  for z in zBySeq:
    z += 1000
    for kID, kVal in enumerate(specialStateIDs):
      z[ z == 1000+kVal ] = -1000 + kID
      uniqueVals = np.union1d(uniqueVals, np.unique(z))
  
  for z in zBySeq:
    for kID, kVal in enumerate(sorted(uniqueVals)):
      z[z == kVal] = kID

  return zBySeq

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
