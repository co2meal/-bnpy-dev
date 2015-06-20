import os
import numpy as np
import glob

from bnpy.ioutil.BNPYArgParser import parse_task_ids

def rankTasksForSingleJobOnDisk(joboutpath):
  ''' Make files for job which allow rank-based referencing of tasks 

      Returns
      ----------
      None. Files are written to disk.
  '''
  # First, rank the tasks from best to worst
  sortedTaskIDs = rankTasksForSingleJob(joboutpath)

  # If we've sorted these same tasks before, just quit early
  infotxtfile = os.path.join(joboutpath, '.info-ranking.txt')
  if os.path.exists(infotxtfile):
    prevRankedTaskIDs = [int(i) for i in np.loadtxt(infotxtfile)]
    curRankedTaskIDs = [int(i) for i in sortedTaskIDs]
    if len(sortedTaskIDs) == len(prevRankedTaskIDs):
      if np.allclose(curRankedTaskIDs, prevRankedTaskIDs):
        return None

  # Remove all old hidden rank files
  hiddenFileList = glob.glob(os.path.join(joboutpath, '.*'))
  for fpath in hiddenFileList:
    if os.path.islink(fpath):
      os.unlink(fpath)

  # Save record of new info file
  np.savetxt(infotxtfile, sortedTaskIDs, fmt='%s')

  for rankID, taskidstr in enumerate(sortedTaskIDs):
    rankID += 1 # 1 based indexing!

    # Make symlink to file like .rank1 or .rank27
    os.symlink(os.path.join(joboutpath, taskidstr), 
               os.path.join(joboutpath, '.rank'+str(rankID)))

    if len(sortedTaskIDs) > 3:
      if rankID == 1:
        os.symlink(os.path.join(joboutpath, taskidstr), 
                   os.path.join(joboutpath, '.best'))
      elif rankID == len(sortedTaskIDs):
        os.symlink(os.path.join(joboutpath, taskidstr), 
                   os.path.join(joboutpath, '.worst'))
      elif rankID == int(np.ceil(len(sortedTaskIDs)/2.0)):
        os.symlink(os.path.join(joboutpath, taskidstr), 
                   os.path.join(joboutpath, '.median'))


def rankTasksForSingleJob(joboutpath):
  ''' Get list of tasks for job, ranked best-to-worst by final ELBO score

      Returns
      ----------
      sortedtaskIDs : list of task names, each entry is an int
  '''
  taskids = parse_task_ids(joboutpath)
  # Read in the ELBO score for each task
  ELBOScores = np.zeros(len(taskids))
  #from IPython import embed; embed()
  for tid, taskidstr in enumerate(taskids):
    assert isinstance(taskidstr, str)
    
    taskELBOTrace = np.loadtxt(os.path.join(joboutpath, 
                                        taskidstr, 'evidence.txt'))
    ELBOScores[tid] = taskELBOTrace[-1]
  #from IPython import embed; embed()
  # Sort in descending order, largest to smallest!
  sortIDs = np.argsort(-1 * ELBOScores)
  return [taskids[t] for t in sortIDs]
