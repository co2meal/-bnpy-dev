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

NUM_STACK = 10

def plotColoredBars(dataset, jobname, taskid, lap, sequences):

  #Load in the saved data
  filename = 'Lap%08.3fMAPStateSeqs.mat' % lap
  path = os.path.expandvars('$BNPYOUTDIR/'+ dataset + '/'+ \
                            jobname + '/' + str(taskid) + '/')

  zHatBySeq = scipy.io.loadmat(path + filename)
  zHatBySeq = zHatBySeq['zHatBySeq'][0]
  hammingFile = open(path+'hamming-distance.txt', 'r')
  hammingDists = hammingFile.readlines()
  hammingDists = [float(x) for x in hammingDists]
  hammingFile.close()


  #Load in the data module
  datamod = imp.load_source(dataset,
                            os.path.expandvars('$BNPYDATADIR/'+dataset+'.py'))
  data = datamod.get_data()

  f, axes = plt.subplots(len(sequences), 1, sharex='col', sharey='row')

  for ii, seqNum in enumerate(sequences):
    image = np.tile(zHatBySeq[seqNum], (NUM_STACK, 1))

    #Add the true labels to the plot (if they exist)
    if (data.TrueParams is not None) and (data.TrueParams['Z'] is not None):
      start = data.doc_range[seqNum]
      stop = data.doc_range[seqNum+1]
      image = np.vstack((image, np.tile(data.TrueParams['Z'][start:stop],
                                        (NUM_STACK, 1))))
      
    #Subplot titles
    if len(sequences) > 1:
      axes[ii].imshow(image)
      axes[ii].set_title('Sequence %d' % sequences[ii])
    else:
      axes.imshow(image)
      axes.set_title('Sequence %d' % sequences[ii])
    
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset')
  parser.add_argument('--jobname')
  parser.add_argument('--taskid')
  parser.add_argument('--lap')
  parser.add_argument('--sequences')
  args = parser.parse_args()

  plotColoredBars(dataset = args.dataset,
                  jobname = args.jobname,
                  taskid = int(args.taskid),
                  lap = int(args.lap),
                  sequences =  [int(x) for x in args.sequences.split(',')])
