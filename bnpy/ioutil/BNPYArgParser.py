import argparse
import ConfigParser
import os
import sys

OnlineDataAlgSet = ['soVB', 'moVB']

# Hard-code the relative path to config/
cfgroot = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
cfgroot = os.path.join(cfgroot, 'config/')
ConfigPaths={cfgroot + 'allocmodel.conf':'allocModelName',
             cfgroot + 'obsmodel.conf':'obsModelName', 
             cfgroot + 'learnalg.conf':'algName',
             cfgroot + 'init.conf':None,
             cfgroot + 'output.conf':None}             

OnlineDataConfigPath =  cfgroot + 'onlinedata.conf'

UNKARGLIST = None

def parseUnknownArgs():
  return arglist_to_kwargs(UNKARGLIST)

def parseRequiredArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName')
  parser.add_argument('allocModelName')
  parser.add_argument('obsModelName')
  parser.add_argument('algName')
  args, unk = parser.parse_known_args()
  return args.__dict__
  
def applyParserToKeywordArgDict(parser, **kwargs):
  ''' Use externally-defined parser to parse given kwargs
      if none-provided, defaults to reading from stdin

      Returns
      --------
  '''
  if len(kwargs.keys()) > 0:
    alist = kwargs_to_arglist(**kwargs)
    args, UnkArgList = parser.parse_known_args(alist)
  else:
    args, UnkArgList = parser.parse_known_args()
  return args, arglist_to_kwargs(UnkArgList)

def parseKeywordArgs(ReqArgs, **kwargs):
  global UNKARGLIST
  if ReqArgs['algName'] in OnlineDataAlgSet:
    ConfigPaths[OnlineDataConfigPath] = None
  else:
    if OnlineDataConfigPath in ConfigPaths:
      del ConfigPaths[OnlineDataConfigPath]

  # BUILD parser using default opts in the config files
  parser = argparse.ArgumentParser()
  parser.add_argument('--moves', type=str,
                       help="String names of moves to perform to escape local optima. Options: {birth,merge}. To perform multiple moves, separate with commas like 'birth,merge' (no spaces).")
  parser.add_argument('--kwhelp', action='store_true', help="Include --kwhelp to print our keyword argument help and exit")
  
  for fpath, secName in ConfigPaths.items():
    if secName is not None:
      secName = ReqArgs[secName]
    addArgGroupFromConfigFile(parser, fpath, secName) 
    if fpath.count('learn') > 0:
      addArgGroupFromConfigFile(parser, fpath, 'birth') 
      addArgGroupFromConfigFile(parser, fpath, 'merge') 

  # PARSE keyword args
  if len(kwargs.keys()) > 0:
    alist = kwargs_to_arglist(**kwargs)
    args, UNKARGLIST = parser.parse_known_args(alist)
  else:
    args, UNKARGLIST = parser.parse_known_args()
  if args.moves is not None:
    args.moves = args.moves.split(',')

    
  
  if args.kwhelp:
    parser.print_help()
    sys.exit(-1)

  # CONVERT parsed arg namespace to a nice dictionary
  argDict = dict()
  for fpath, secName in ConfigPaths.items():
    if secName is not None:
      secName = ReqArgs[secName]
    addArgsToDictByConfigFile(argDict, args, fpath, secName)
    if fpath.count('learn') > 0 and args.moves is not None:
      for moveName in args.moves:
        addArgsToDictByConfigFile(argDict, args, fpath, moveName)
  return argDict

def arglist_to_kwargs(alist):
  kwargs = dict()
  a = 0
  while a < len(alist):
    curarg = alist[a]
    if curarg.startswith('--'):
      argname = curarg[2:]
      argval = alist[a+1]
      curType = getType(argval)
      kwargs[argname] = curType(argval)
      a += 1
    a += 1
  return kwargs

def kwargs_to_arglist(**kwargs):
  arglist = list()
  for key,val in kwargs.items():
    arglist.append('--' + key)
    arglist.append(str(val))
  return arglist

def addArgsToDictByConfigFile(argDict, args, filepath, targetSectionName=None):
  config = readConfigParser( filepath)
  for secName in config.sections():
    if secName.count("Help") > 0:
      continue
    if targetSectionName is not None:
      if secName != targetSectionName:
        continue
    DefDict = dict(config.items(secName))  
    secArgDict = dict([ (k,v) for (k,v) in vars(args).items() if k in DefDict])
    argDict[secName] = secArgDict

def addArgGroupFromConfigFile(parser, confFilePath, targetSectionName=None):
  config = readConfigParser(confFilePath)
  for secName in config.sections():
    if secName.count("Help") > 0:
      continue
    if targetSectionName is not None:
      if secName != targetSectionName:
        continue
    DefDict = dict( config.items( secName ) )
    try:
      HelpDict = dict( config.items( secName+"Help") )
    except ConfigParser.NoSectionError:
      HelpDict = dict()
      
    group = parser.add_argument_group(secName)    
    for optName, defVal in DefDict.items():
      defType = getType( defVal)
      if optName in HelpDict:
        helpMsg = '[def=%s] %s' % (defVal, HelpDict[optName])
      else:
        helpMsg = '[def=%s]' % (defVal)
      
      if defType == True or defType == False:
        group.add_argument( '--%s' % (optName), default=defType, help=helpMsg, action='store_true')
      else:
        group.add_argument( '--%s' % (optName), default=defVal, help=helpMsg, type=defType)


def readConfigParser(filepath):
  config = ConfigParser.SafeConfigParser()
  config.optionxform = str
  config.read( filepath )  
  return config

def getType( defVal ):
  ''' Determine Python type from the provided default value
      Returns
      ---------
      a Python type object
      {True, False, int, float, str}
  '''
  if defVal == 'true' or defVal == 'True':
    return True
  if defVal == 'false' or defVal == 'False':
    return False
  try:
    int(defVal)
    return int
  except Exception:
    pass
  try:
    float(defVal)
    return float
  except Exception:
    return str


def addRequiredVizArgsToParser(parser):
  ''' Update parser to include required args: data, model, learn algorithm
  '''
  parser.add_argument('dataName', type=str,
        help='name of python script that produces data to analyze.')
  parser.add_argument('allocModelName', type=str,
        help='name of allocation model. {MixModel, DPMixModel}')
  parser.add_argument('obsModelName', type=str,
        help='name of observation model. {Gauss, ZMGauss}')
  parser.add_argument('algNames', type=str,
        help='name of learning algorithm, {EM, VB, moVB, soVB}. comma-separated if multiple')

def addStandardVizArgsToParser(parser):
  ''' Update parser to include standard visualization arguments
  '''
  parser.add_argument('--jobnames', type=str, default='defaultjob',
        help='name of experiment whose results should be plotted')
        
  parser.add_argument('--taskids', type=str, default=None,
        help="int ids for tasks (individual runs) of the given job to plot." + \
              'Ex: "1" or "3" or "1,2,3" or "1-6"')
  
  parser.add_argument('--savefilename', type=str, default=None,
        help="location where to save figure (absolute path directory)")

          
def parse_task_ids(jobpath, taskids=None):
  ''' Return list of task ids
  '''
  import glob
  import numpy as np
  if taskids is None:
    fulltaskpaths = glob.glob(os.path.join(jobpath,'*'))
    taskids = [os.path.split(tpath)[-1] for tpath in fulltaskpaths]
  elif taskids.count(',') > 0:
    taskids = [t for t in taskids.split(',')]
  elif taskids.count('-') > 0:
    fields = taskids.split('-')
    if not len(fields)==2:
      raise ValueError("Bad taskids specification")
    fields = np.int32(np.asarray(fields))
    taskids = np.arange(fields[0],fields[1]+1)
    taskids = [str(t) for t in taskids]
  if type(taskids) is not list:
    taskids = list(taskids)
  return taskids