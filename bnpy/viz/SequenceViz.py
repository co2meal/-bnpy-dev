'''
SequenceViz.py
Visualizes sequential data as a plot of colored bars of the estimated state
as calculated by the Viterbi algorithm.  Note that this script requires you
to have made a run of bnpy with the custom func learnalg/extras/XViterbi.py
running.

Example use:

python SequenceViz.py --dataset MoCap6 --jobname defaultjob --taskid 1
                        --lap 5 --sequences 1,2,4,6

'''



import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import imp
import sys
import argparse


def plotSingleJob(dataset, jobname, taskids, lap, sequences, dispTrue = True):
  '''
  Returns the array of data corresponding to a single sequence to display

  If dispTrue = True, the true labels will be shown underneath the
    estimated labels
  '''

  NUM_STACK = 550 / len(sequences) #why 550?  It looks nice
  if dispTrue:
    NUM_STACK /= 2

  f, axes = plt.subplots(len(sequences), len(taskids),
                          sharex='col', sharey='row')
    
  for tt, taskid in enumerate(taskids):
    #Load in the saved data
    filename = 'Lap%08.3fMAPStateSeqsAligned.mat' % lap
    path = os.path.expandvars('$BNPYOUTDIR/'+ dataset + '/'+ \
                              jobname + '/' + str(taskid) + '/')

    zHatBySeq = scipy.io.loadmat(path + filename)
    zHatBySeq = zHatBySeq['zHatBySeqAligned'][0]
    hammingFile = open(path+'hamming-distance.txt', 'r')
    hammingDists = hammingFile.readlines()
    hammingDists = [float(x) for x in hammingDists]
    hammingFile.close()


    #Load in the data module
    datamod = imp.load_source(dataset,
                              os.path.expandvars('$BNPYDATADIR/'+dataset+'.py'))
    data = datamod.get_data()


    for ii, seqNum in enumerate(sequences):
      image = np.tile(zHatBySeq[seqNum], (NUM_STACK, 1))

      #Add the true labels to the plot (if they exist)
      if ( (data.TrueParams is not None) and ('Z' in data.TrueParams)
           and (dispTrue) ):
        start = data.doc_range[seqNum]
        stop = data.doc_range[seqNum+1]
        image = np.vstack((image, np.tile(data.TrueParams['Z'][start:stop],
                                          (NUM_STACK, 1))))
      
      #Title the rows and columns
      if tt == 0:
        if len(sequences) == 1 or len(taskids) == 1:
          axes[ii].set_ylabel('Sequence %d' % sequences[ii], fontsize=13)
        else:
          axes[ii, 0].set_ylabel('Sequence %d' % sequences[ii], fontsize=13)
      if ii == 0:
        if len(sequences) == 1 or len(taskids) == 1:
          axes[tt].set_title('Taskid %d' % taskids[tt])
        else:
          axes[0, tt].set_title('Taskid %d' % taskids[tt])
      

      if len(sequences) == 1 or len(taskids) == 1:
        axes[ii+tt].imshow(image)
      else:
        axes[ii,tt].imshow(image)

      f.suptitle(jobname, fontsize = 18)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset')
  parser.add_argument('--jobnames')
  parser.add_argument('--taskids')
  parser.add_argument('--lap')
  parser.add_argument('--sequences')
  args = parser.parse_args()

  jobs = args.jobnames.split(',')

  for job in jobs:
    plotSingleJob(dataset = args.dataset,
                  jobname = job,
                  taskids = [int(x) for x in args.taskids.split(',')],
                  lap = int(args.lap),
                  sequences =  [int(x) for x in args.sequences.split(',')],
                  dispTrue = True)
    

  plt.show()
