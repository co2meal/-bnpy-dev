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
import data
import json

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

#import numpy as np
#import bnpy
import BNPYArgParser

ConfigPaths=['config/allocmodel.conf','config/obsmodel.conf', 
             'config/init.conf', 'config/inference.conf', 'config/output.conf']

def main( jobID=1, taskID=1, LOGFILEPREFIX=None):
  ArgDict = BNPYArgParser.parseArgs(ConfigPaths)
  starttaskid = ArgDict['OutputPrefs']['taskid']
  nTask = ArgDict['OutputPrefs']['nTask']
  for task in xrange(starttaskid, starttaskid+nTask):
    run_training_task(ArgDict, taskid=0, nTask=1)
  
def run_training_task(ArgDict, taskid=0, nTask=1, doSaveToDisk=True, doWriteStdOut=True): 
    ''' Run training given specifications for data, model and inference
    '''
    
    taskoutpath = getOutputPath(ArgDict, taskID=task)
    if doSaveToDisk:
        createEmptyOutputPathOnDisk(taskoutpath)
        writeArgsToFile(ArgDict, taskoutpath)
    configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk, doWriteStdOut)
    
    jobname = ArgDict['OutputPrefs']['jobname']
    algseed = createUniqueRandomSeed(jobname, taskID=task)
    dataseed = createUniqueRandomSeed('', taskID=task)
    
    # Load data
    Data, InitData = loadDataAndPrintSummary(ArgDict, taskID=task, dataseed=dataseed)
  
    # Create the model
    model = createModelAndPrintSummary(Data, ArgDict, taskID=task)
    
    learnAlg = createLearnAlgAndPrintSummary(Data, model, ArgDict, algseed=algseed, 
                                              taskID=0
                                            )
    printTaskSummary(task, nTask, algseed, dataseed)

    #learnAlg.initialize( model, InitData, **ArgDict['init'] )
    #learnAlg.fit( model, Data)
  
  
def loadDataAndPrintSummary( ArgDict, taskID=0, dataseed=0 ): 
  Log.info( 'Data 123 and xyz' )
  return dict(summary='abc123'), None
  
def createModelAndPrintSummary( Data, ArgDict, taskID=0 ):
  return None 
  
def createLearnAlgAndPrintSummary( Data, model, ArgDict, algseed=0, taskID=0):
  return None
  
def printTaskSummary( task, nTask, algseed, dataseed): 
  Log.info( 'Trial %2d/%d | alg. seed: %d | data seed: %d' \
                 % (task, nTask, algseed, dataseed)
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
  # Make sure the path (including all parent paths) exists
  mkpath( taskoutpath )
  # Make sure the path is clean
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
    with open( os.path.join(taskoutpath, key+'Args.txt'), 'w') as fout:
      json.dump( ArgDict[key], fout)



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
  
  
  
'''
    ####################################################### Data Module parsing
    dataParams = dict()
    for argName in ['nBatch', 'nRep', 'batch_size', 'seed', 'orderseed', 'nObsTotal']:
      dataParams[argName] = args.__getattribute__( argName )
      
    # Dynamically load module provided by user as data-generator
    sys.path.append( os.environ['BNPYDATADIR'] )
    datagenmod = __import__( args.datagenModule,fromlist=[])
      
    algName = args.algName
    if args.doonline:
      algName = 'o'+algName
    elif args.doincremental or args.doi:
      algName = 'i'+algName
    if args.dosplitmerge:
      algName = algName + 'sm'
    
    if args.algName.count('VB')>0 or args.algName.count('GS')>0:
      obsPrior = PriorConstr[ args.obsName ]()
    else:
      obsPrior = None  
    am = AllocModelConstructor[ args.modelName ]( qType=algName, **modelParams )
    model = bnpy.HModel( am, args.obsName, obsPrior )

    doAdmix = (args.modelName.count('Admix') + args.modelName.count('HDP') )> 0
    doHMM = args.modelName.count('HMM') > 0

    if 'get_short_name' in dir( datagenmod ):
      datashortname = datagenmod.get_short_name()
    else:
      datashortname = args.datagenModule[:7]
    jobpath = os.path.join(datashortname, args.modelName, args.obsName, algName, args.jobname)
    
    Data, dataSummaryStr = load_data( datagenmod, dataParams, args.doonline, args.doi, doAdmix, doHMM )

    if args.doonline or args.doi:
      InitData,_ = load_data( datagenmod, dataParams, False, False, doAdmix, doHMM )
      nTotal = dataParams['batch_size']*dataParams['nBatch']
      InitData['nTotal'] = nTotal
      model.config_from_data( InitData, **obsPriorParams )
    else:
      model.config_from_data( Data, **obsPriorParams )

    if args.dotest:
      #ToDo: unused
      TestData = load_test_data( datagenmod, dataParams, doAdmix, doHMM)

    # Print Message!
    if 'print_data_info' in dir( datagenmod ):
      datagenmod.print_data_info( args.modelName )
    print 'Data Specs:\n', dataSummaryStr
    model.print_model_info()
    print 'Learn Alg:  %s' % (algName)

    ####################################################### Spawn individual tasks
    for task in xrange( args.taskid, args.taskid+args.nTask ):    
      if (args.doonline or args.doi): # and task is not args.taskid:
        # Reload the data generator
        dataParams['orderseed'] = task
        Data, dataSummaryStr = load_data( datagenmod, dataParams, args.doonline, args.doi, doAdmix, doHMM )

      # Get unique RNG seed from the jobname, reproducible on any machine
      #seed = hash( args.jobname+str(task) ) % np.iinfo(int).max
      seed = int( hashlib.md5( args.jobname[:5]+str(task) ).hexdigest(), 16) % 1e7
      seed = int(seed)
      algParams['seed'] = seed

      basepath = os.path.join( os.environ['BNPYOUTDIR'], jobpath, str(task) )
      mkpath(  basepath )
      clear_folder( basepath )
      algParams['savefilename'] = os.path.join( basepath, '' )

      print 'Trial %2d/%d | alg. seed: %d | data seed: %d' \
                 % (task, args.nTask, algParams['seed'], dataParams['seed'])
      print '  savefile: %s' % (algParams['savefilename'])

      if jobID > 1:
        logpath = os.path.join( os.environ['BNPYLOGDIR'], jobpath )
        mkpath( logpath )
        clear_folder( logpath, prefix=str(task) )
        os.symlink( LOGFILEPREFIX+'.out', '%s/%d.out' % (logpath, task) )
        os.symlink( LOGFILEPREFIX+'.err', '%s/%d.err' % (logpath, task) )
        print '   logfile: %s' % (logpath)
    
      curmodel = copy.deepcopy(model)

      ##########################################################  Run Learning Alg
      if args.doonline:
        if args.dosplitmerge:
          learnAlg = bnpy.learn.OnlineVBSMLearnAlg( **algParams )
        else:
          learnAlg = bnpy.learn.OnlineVBLearnAlg( **algParams )
        learnAlg.init_global_params( curmodel, InitData, seed, nIterInit=args.nIterInit )
        learnAlg.fit( curmodel, Data ) # remember, Data is a Generator object here!
      elif algName == 'CGS':
        learnAlg = bnpy.learn.GibbsSamplerAlg( **algParams )
        learnAlg.fit( curmodel, Data, seed )
      elif args.doi:
        if args.dosplitmerge:
          print 'iVB + SPLIT + MERGE'
          learnAlg = bnpy.learn.iVBSMLearnAlg( **algParams )
        else:
          print 'iVB fixed truncation'
          learnAlg = bnpy.learn.iVBLearnAlg(  **algParams )
        
        learnAlg.init_global_params( curmodel, InitData, seed, nIterInit=args.nIterInit )

        if args.dodebugelbocalc:
          assert InitData['nObs'] == args.batch_size*args.nBatch
          learnAlg.fit( curmodel, Data, AllData=InitData ) # remember, Data is a Generator object here!
        else:
          learnAlg.fit( curmodel, Data ) # remember, Data is a Generator object here!
      elif args.dobatch:
        if args.dosplitmerge:
          learnAlg = bnpy.learn.VBSMLearnAlg( **algParams )
        elif args.doincremental:
          learnAlg = bnpy.learn.IncrementalVBLearnAlg(  **algParams )
        else:
          learnAlg = bnpy.learn.VBLearnAlg(  **algParams )
          
        learnAlg.init_global_params( curmodel, Data, seed, nIterInit=args.nIterInit )
        learnAlg.fit( curmodel, Data )
            
      if args.doprintfinal:
        curmodel.print_global_params()

    ##########################################################  Wrap Up
    return None
'''

if __name__ == '__main__':
  main()
