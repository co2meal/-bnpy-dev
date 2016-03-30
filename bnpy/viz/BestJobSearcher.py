import argparse
import numpy as np
import bnpy
import glob
import os

def makeBestJobPathViaGridSearch(
        jobpathPattern=None,
        wildcard='WILD',
        **kwargs):
    ''' Given a wildcard jobpath, make a jobpath-var-BEST directory
    '''
    # Remove trailing '/', if present
    jobpathPattern = jobpathPattern.rstrip(os.path.sep)
    bestjobpath, bestjobwildpairs = findBestJobViaGridSearch(
        jobpathPattern, wildcard=wildcard, **kwargs)
    LINKtobestjobpath = jobpathPattern.replace(wildcard, 'BEST')
    # Remove any old version from this search
    if os.path.islink(LINKtobestjobpath):
        os.unlink(LINKtobestjobpath)
    assert os.path.exists(bestjobpath)
    
    # Make a new symlink
    os.symlink(bestjobpath, LINKtobestjobpath)
    print "Made New Directory"
    print "------------------"
    print LINKtobestjobpath

def findBestJobViaGridSearch(
        jobpathPattern=None,
        wildcard='WILD',
        taskids='all',
        scoreTxtFile='validation-predlik-avgLikScore.txt',
        singleTaskScoreFunc=np.max,
        multiTaskScoreFunc=np.median,
        **kwargs):
    ''' Given a wildcard jobpath, find best of all matching jobs

    Returns
    -------
    jobpath : full path to experiment directory that scored best
    jobwildstr : short string summarizing the values that won
    '''
    # Identify the fields that we are searching over
    wildVarNames = list()
    keyvalPairs = jobpathPattern.split('-')
    if len(keyvalPairs) <= 1:
        raise ValueError("jobpathPattern not parseable into name=val pairs")
    basepath = keyvalPairs[0]
    keyvalPairs = keyvalPairs[1:]
    keyvalStartLocs = np.asarray([len(kv) for kv in keyvalPairs])
    keyvalStartLocs = np.hstack([len(basepath), len(basepath) + np.cumsum(keyvalStartLocs+1)])
    start = 0
    while True:
        wcloc = jobpathPattern.find(wildcard, start)
        if wcloc < 0:
            break
        pairScores = np.abs(keyvalStartLocs - wcloc)
        pairScores[keyvalStartLocs > wcloc] = 555555
        pairID = np.argmin(pairScores)
        key = keyvalPairs[pairID]
        key = key[:key.index('=')]
        wildVarNames.append(key)
        start = wcloc + 1
    print "Grid search over field names: ", wildVarNames

    jobpathList = [jpath for jpath in 
        sorted(glob.glob(jobpathPattern.replace(wildcard, '*')))
        if jpath.count("BEST") == 0]
    if len(jobpathList) == 0:
        raise ValueError("No matching jobs found on disk for pattern:\n" + 
            jobpathPattern)    

    jobWildDescrList = list()
    jobScores = np.zeros(len(jobpathList))
    for jj, jobpath in enumerate(jobpathList):
        # Loop over the tasks for this job
        taskIDstrList = bnpy.ioutil.BNPYArgParser.parse_task_ids(
            jobpath, taskids)
        taskScores = np.zeros(len(taskIDstrList))
        for ii, taskIDstr in enumerate(taskIDstrList):
            tasktxtfile = os.path.join(jobpath, taskIDstr, scoreTxtFile)
            taskScores[ii] = singleTaskScoreFunc(np.loadtxt(tasktxtfile))
        jobScores[jj] = multiTaskScoreFunc(taskScores)

        # Pretty print a summary
        jobwildstr = ""
        for varName in wildVarNames:
            start = jobpath.find("-" + varName + "=")
            if start >= 0:
                start += 1 # correct for leading '-'
            else:
                raise ValueError("jobpath did not have key=val: %s=val" % (varName))
            stop = jobpath.find('-', start)
            if stop < 0:
                stop = len(jobpath)
            kvstr = jobpath[start:stop]
            jobwildstr += kvstr + "-"
        print "%.3f %s" % (jobScores[jj], jobwildstr)
        jobWildDescrList.append(jobwildstr[:-1])

    bestJobID = np.argmax(jobScores)
    print "Winner"
    print "------"
    print "%.3f %s" % (jobScores[bestJobID], jobWildDescrList[bestJobID])
    return jobpathList[bestJobID], jobWildDescrList[bestJobID]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobpathPattern",
        help="Experiment file path, with BEST as placeholder for wildcards",
        )
    args = parser.parse_args()

    makeBestJobPathViaGridSearch(**args.__dict__)