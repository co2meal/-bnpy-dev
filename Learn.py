'''
 User-facing executable script for learning bnpy models
  with a variety of possible inference algorithms, such as
    ** Expectation Maximization (EM)
    ** Variational Bayesian Inference (VB)
    ** Collapsed MCMC Gibbs Sampling (CGS)
    
 Author: Mike Hughes (mike@michaelchughes.com)

  Quickstart
  -------
  To run EM for a 3-component GMM on easy toy data, do
  >> python Learn.py EasyToyData MixModel Gaussian EM --K=3

  Usage
  -------
  python Learn.py <data_module_name> <aModel name> <oModel name> <alg name>  [options]

  TODO: write better doc
'''
import os
import sys
import logging
import json

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

import numpy as np
import bnpy
import BNPYArgParser

ConfigPaths=['config/allocmodel.conf','config/obsmodel.conf', 
             'config/init.conf', 'config/inference.conf', 'config/output.conf']

def main( jobID=1, taskID=1, LOGFILEPREFIX=None):
  ArgDict = BNPYArgParser.parseArgs(ConfigPaths)
  starttaskid = ArgDict['OutputPrefs']['taskid']
  nTask = ArgDict['OutputPrefs']['nTask']
  for taskid in xrange(starttaskid, starttaskid+nTask):
    run_training_task(ArgDict, taskid=taskid, nTask=1)
  
def run_training_task(ArgDict, taskid=0, nTask=1, doSaveToDisk=True, doWriteStdOut=True): 
    ''' Run training given specifications for data, model and inference
    '''
    
    taskoutpath = getOutputPath(ArgDict, taskID=taskid)
    if doSaveToDisk:
        createEmptyOutputPathOnDisk(taskoutpath)
        writeArgsToFile(ArgDict, taskoutpath)
    configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk, doWriteStdOut)
    
    jobname = ArgDict['OutputPrefs']['jobname']
    algseed = createUniqueRandomSeed(jobname, taskID=taskid)
    dataseed = createUniqueRandomSeed('', taskID=taskid)
    
    Data, InitData = loadData(ArgDict, dataseed=dataseed)

    # Create and initialize model parameters
    hmodel = createModel(Data, ArgDict)
    hmodel.init_global_params(InitData, seed=algseed, **ArgDict['Initialization'])

    learnAlg = createLearnAlg(Data, hmodel, ArgDict, \
                              algseed=algseed, savepath=taskoutpath)

    Log.info(Data.summary)
    Log.info(hmodel.get_model_info())

    printTaskSummary(taskid, nTask, algseed, dataseed)
    Log.info('savepath: %s' % (taskoutpath))

    learnAlg.fit(hmodel, Data)
  
  
def loadData(ArgDict, dataseed=0): 
  sys.path.append(os.environ['BNPYDATADIR'])
  datagenmod = __import__(ArgDict['dataName'],fromlist=[])
  Data = datagenmod.get_data(seed=dataseed)
  return Data, Data
  
def createModel(Data, ArgDict):
  algName = ArgDict['algName']
  aName = ArgDict['allocModelName']
  oName = ArgDict['obsModelName']
  aPriorDict = ArgDict[aName]
  oPriorDict = ArgDict[oName]
  hmodel = bnpy.HModel.InitFromData(algName, aName, oName, aPriorDict, oPriorDict, Data)
  return hmodel  

def createLearnAlg(Data, model, ArgDict, algseed=0, savepath=None):
  algName = ArgDict['algName']
  algP = ArgDict[algName]
  outputP = ArgDict['OutputPrefs']
  if algName == 'EM' or algName == 'VB':
    learnAlg = bnpy.learn.VBLearnAlg(savedir=savepath, seed=algseed, \
                                      algParams=algP, outputParams=outputP)
  else:
    raise NotImplementedError("Unknown learning algorithm " + algName)
  return learnAlg

def printTaskSummary( taskID, nTask, algseed, dataseed): 
  Log.info( 'Trial %2d/%d | alg. seed: %d | data seed: %d' \
                 % (taskID+1, nTask, algseed, dataseed)
            )

  
def createUniqueRandomSeed( jobname, taskID=0):
  ''' Get unique RNG seed from the jobname, reproducible on any machine
  '''
  import hashlib
  if len(jobname) > 5:
    jobname = jobname[:5]
  seed = int( hashlib.md5( jobname+str(taskID) ).hexdigest(), 16) % 1e7
  return int(seed)
  
  
def getOutputPath( ArgDict, taskID=0 ):
  return os.path.join(os.environ['BNPYOUTDIR'], 
                       ArgDict['dataName'], 
                       ArgDict['allocModelName'],
                       ArgDict['obsModelName'],
                       ArgDict['algName'],
                       ArgDict['OutputPrefs']['jobname'], 
                       str(taskID) )

def createEmptyOutputPathOnDisk( taskoutpath ):
  from distutils.dir_util import mkpath
  # Ensure the path (including all parent paths) exists
  mkpath( taskoutpath )
  # Ensure the path has no data from previous runs
  deleteAllFilesFromDir( taskoutpath )
  
def deleteAllFilesFromDir( savefolder, prefix=None ):
  '''  Erase (recursively) all contents of a folder
         so that we can write fresh results to it
  '''
  for the_file in os.listdir( savefolder ):
    if prefix is not None:
      if not the_file.startswith(prefix):
        continue
    file_path = os.path.join( savefolder, the_file)
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)

def writeArgsToFile( ArgDict, taskoutpath ):
  ''' Save arguments as key/val pairs to a plain text file
      so that we can figure out what settings were used for a saved run later on
  '''
  RelevantOpts = dict()
  for key in ArgDict:
    if key.count('Name') > 0:
      RelevantOpts[ ArgDict[key] ] = 1
  for key in ArgDict:
    if key.count('Name') > 0 or key not in RelevantOpts:
      continue
    with open( os.path.join(taskoutpath, 'args-'+key+'.txt'), 'w') as fout:
      json.dump(ArgDict[key], fout)



def configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk=True, doWriteStdOut=True):
  Log.handlers = [] # remove pre-existing handlers!
  formatter = logging.Formatter('%(message)s')
  
  if doSaveToDisk:
    fh = logging.FileHandler(os.path.join(taskoutpath,"transcript.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    Log.addHandler(fh)

  if doWriteStdOut: 
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    Log.addHandler(ch)

if __name__ == '__main__':
  main()
