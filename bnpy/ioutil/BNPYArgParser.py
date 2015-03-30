import argparse
import ConfigParser
import os
import sys
import numpy as np

OnlineDataAlgSet = ['soVB', 'moVB', 'strVB']

dataHelpStr = 'Name of dataset, defined by a python script in $BNPYDATADIR.'

from bnpy.allocmodel import AllocModelNameSet
choiceStr = ' {' + ','.join([x for x in (AllocModelNameSet)]) + '}'
aModelHelpStr = 'Name of allocation model.' + choiceStr

from bnpy.obsmodel import ObsModelNameSet
choiceStr = ' {' + ','.join([x for x in (ObsModelNameSet)]) + '}'
oModelHelpStr = 'Name of observation model.' + choiceStr

algChoices = set(['EM','VB','moVB','moVBsimple', 'soVB','GS', 'strVB'])
choiceStr = ' {' + ','.join([x for x in (algChoices)]) + '}'
algHelpStr = 'Name of learning algorithm.' + choiceStr

MovesHelpStr = "String names of moves to perform to escape local optima. Options: {birth,merge}. To perform several move types, separate with commas like 'birth,merge' (no spaces)."

KwhelpHelpStr = "Include --kwhelp to print our keyword argument help and exit"

########################################################### User-facing 
###########################################################  functions
def parseRequiredArgs():
  ''' Process standard input for bnpy's required arguments, as a dict.

      Required Args: dataName, allocModelName, obsModelName, algName

      All other args in stdin are left alone (for later processing).

      Returns
      --------
      ArgDict : dict, with fields
      * dataName
      * allocModelName
      * obsModelName
      * algName
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName', type=str, help=dataHelpStr)
  parser.add_argument('allocModelName',
                       type=str, help=aModelHelpStr)
  parser.add_argument('obsModelName',
                       type=str, help=oModelHelpStr)
  parser.add_argument('algName', 
                       type=str, help=algHelpStr)
  args, unk = parser.parse_known_args()
  if args.allocModelName not in AllocModelNameSet:
    raise ValueError('Unrecognized allocModelName %s' % (args.allocModelName))
  if args.obsModelName not in ObsModelNameSet:
    raise ValueError('Unrecognized obsModelName %s' % (args.obsModelName))
  if args.algName not in algChoices:
    raise ValueError('Unrecognized learning algName %s' % (args.algName))
  return args.__dict__

def parseKeywordArgs(ReqArgs, **kwargs):
  ''' Process standard input (or provided input) for keyword args.

      Returns
      --------
      KwDict : dict, with key/val pair for every keyword argument
  '''
  movesParser = argparse.ArgumentParser()
  movesParser.add_argument('--moves', type=str, default=None, help=MovesHelpStr)
  MovesArgDict, unkDict = applyParserToStdInOrKwargs(movesParser, **kwargs)

  Moves = set()  
  if MovesArgDict['moves'] is not None:
    for move in MovesArgDict['moves'].split(','):
      Moves.add(move)
  
  ## Create parser, fill with default options from files
  parser = makeParserWithDefaultsFromConfigFiles(ReqArgs, Moves)
  parser.add_argument('--kwhelp', action='store_true', help=KwhelpHelpStr)

  ## Apply the parser to input keywords
  kwargs, unkDict = applyParserToStdInOrKwargs(parser, **kwargs)
  if kwargs['kwhelp']:
    parser.print_help()
    sys.exit(-1)

  if 'moves' in unkDict:
    del unkDict['moves']

  ## Transform kwargs from "flat" dict, with no sense of sections
  #  into a multi-level dict, with sections for 'EM', 'Gauss', 'MixModel', etc.
  kwargs = organizeParsedArgDictIntoSections(ReqArgs, Moves, kwargs)
  return kwargs, unkDict

########################################################### Parser Utils
########################################################### 
def applyParserToStdInOrKwargs(parser, **kwargs):
  ''' Extract all fields defined in parser from stdin (or provided kwargs)

     If no kwargs provided, they are read from stdin.

     Returns
     --------
     ArgDict : dict of parsed arg/value pairs for all fields defined in parser
     UnkDict : dict of unknown arg/value pairs
  '''
  if len(kwargs.keys()) > 0:
    kwlist, ComplexTypeDict = kwargs_to_arglist(**kwargs)
    args, unkList = parser.parse_known_args(kwlist)
    ArgDict = args.__dict__
    ArgDict.update(ComplexTypeDict)
  else:
    ## Parse all args/kwargs from stdin
    args, unkList = parser.parse_known_args()
    ArgDict = args.__dict__
  return ArgDict, arglist_to_kwargs(unkList)


########################################################### Conversion
########################################################### argdict <-> arglist
def arglist_to_kwargs(alist):
  ''' Return dict where key/val pairs are neighboring entries in given list

      Examples
      ---------
      >> arglist_to_kwargs(['--a', 1, '--b', 'stegosaurus'])
      dict(a=1, b='stegosaurus')
      >> arglist_to_kwargs(['requiredarg', 1])
      dict()
  '''
  kwargs = dict()
  a = 0
  while a + 1 < len(alist):
    curarg = alist[a]
    if curarg.startswith('--'):
      argname = curarg[2:]
      argval = alist[a+1]
      curType = _getTypeFromString(argval)
      if type(curType) == type:
        kwargs[argname] = curType(argval)
      else:
        kwargs[argname] = curType
      a += 1
    a += 1
  return kwargs

def kwargs_to_arglist(**kwargs):
  ''' Return arglist where consecutive entries are the input key/value pairs

      Returns
      -------
      arglist : list
      SafeDictForComplexTypes : dict
  '''
  keys = kwargs.keys()
  keys.sort(key=len) # sorty by length, smallest to largest
  arglist = list()
  SafeDict = dict()
  for key in keys:
    val = kwargs[key]
    if isinstance(val, dict) or isinstance(val, np.ndarray):
      SafeDict[key] = val
    else:
      arglist.append('--' + key)
      arglist.append(str(val))
  return arglist, SafeDict

########################################################### Setup Parser
########################################################### from config file
def makeParserWithDefaultsFromConfigFiles(ReqArgs, Moves):
  ''' Returns parser object, filled with default settings from config files
      
      Only certain sections of the config files are included,
      selected by the provided RequiredArgs and Moves
      
      Returns
      -------
       parser : argparse.ArgumentParser, with updated expected args
  '''
  parser = argparse.ArgumentParser()
  configFiles = _getConfigFileDict(ReqArgs)
  for fpath, secName in configFiles.items():
    if secName is not None:
      secName = ReqArgs[secName]
    fillParserWithDefaultsFromConfigFile(parser, fpath, secName) 
    if fpath.count('learn') > 0:
      for moveName in Moves: 
        fillParserWithDefaultsFromConfigFile(parser, fpath, moveName)
  return parser

def _getConfigFileDict(ReqArgs):
  ''' Returns dict of config files to inspect for parsing keyword options,
      
      These files contain default settings for bnpy.

      Returns
      --------
      ConfigPathMap : dict with key/value pairs s.t. 
      * key : absolute filepath to config file 
      * value : string name of required arg 

      Examples
      --------
      >> CMap = _getConfigFilePathMap()
      >> CMap[ '/path/to/bnpy/bnpy/config/allocmodel.conf' ]
      'allocModelName'
  '''
  bnpyroot = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
  cfgroot = os.path.join(bnpyroot, 'config/')
  ConfigPathMap = {
    cfgroot + 'allocmodel.conf':'allocModelName',
    cfgroot + 'obsmodel.conf':'obsModelName', 
    cfgroot + 'learnalg.conf':'algName',
    cfgroot + 'init.conf':None,
    cfgroot + 'output.conf':None,
    }
  OnlineDataConfigPath =  cfgroot + 'onlinedata.conf'
  if ReqArgs['algName'] in OnlineDataAlgSet:
    ConfigPathMap[OnlineDataConfigPath] = None
  return ConfigPathMap

def fillParserWithDefaultsFromConfigFile(parser, confFile,
                                         targetSectionName=None):
  ''' Add default arg key/value pairs from confFile to the parser.
      
      Returns
      -------
      None. Parser object updated in-place.
  '''
  config = _readConfigFile(confFile)
  for curSecName in config.sections():
    if curSecName.count("Help") > 0:
      continue
    if targetSectionName is not None:
      if curSecName != targetSectionName:
        continue

    DefDict = dict(config.items(curSecName))
    try:
      HelpDict = dict(config.items(curSecName+"Help"))
    except ConfigParser.NoSectionError:
      HelpDict = dict()
      
    group = parser.add_argument_group(curSecName)    
    for argName, defVal in DefDict.items():
      defType = _getTypeFromString(defVal)
      if argName in HelpDict:
        helpMsg = '[%s] %s' % (defVal, HelpDict[argName])
      else:
        helpMsg = '[%s]' % (defVal)
      argName = '--' + argName

      if defType == True or defType == False:
        group.add_argument(argName, default=defType,
                             help=helpMsg, action='store_true')
      elif defType == str:
        ## Don't enforce types, so we can pass dicts
        group.add_argument(argName, default=defVal, help=helpMsg)
      else:
        group.add_argument(argName, default=defVal, help=helpMsg, type=defType)

def _readConfigFile(filepath):
  ''' Read entire configuration from a .conf file
  '''
  config = ConfigParser.SafeConfigParser()
  config.optionxform = str
  config.read(filepath)  
  return config

def _getTypeFromString(defVal):
  ''' Determine Python type from the provided default value
      Returns
      ---------
      a Python type object
      {True, False, int, float, str}
  '''
  if defVal.lower() == 'true':
    return True
  if defVal.lower() == 'false':
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

########################################################### Organize Parsed Args
###########################################################  into sections
def organizeParsedArgDictIntoSections(ReqArgs, Moves, kwargs):
  ''' Organize 'flat' dictionary of key/val pairs into sections
      
      Returns
      --------
      finalArgDict : dict with sections for algName, obsModelName, etc.
  '''
  outDict = dict()
  configFileDict = _getConfigFileDict(ReqArgs)
  for fpath, secName in configFileDict.items():
    if secName is not None:
      secName = ReqArgs[secName]
    addArgsFromSectionToDict(kwargs, fpath, secName, outDict)
    if fpath.count('learn') > 0:
      for moveName in Moves:
        addArgsFromSectionToDict(kwargs, fpath, moveName, outDict)
  return outDict

def addArgsFromSectionToDict(inDict, confFile, targetSecName, outDict):
  ''' Transfer 'flat' dictionary kwargs into argDict by section
  '''
  config = _readConfigFile(confFile)
  for secName in config.sections():
    if secName.count("Help") > 0:
      continue
    if targetSecName is not None:
      if secName != targetSecName:
        continue
    BigSecDict = dict(config.items(secName))
    secDict = dict([ (k,v) for (k,v) in inDict.items() if k in BigSecDict])
    outDict[secName] = secDict
  return outDict




########################################################### Parse args for viz
###########################################################

def addRequiredVizArgsToParser(parser):
  ''' Update parser to include required args: data, model, learn algorithm
  '''
  parser.add_argument('dataName', type=str,
        help='Name of dataset.')
  parser.add_argument('allocModelName', type=str, help=aModelHelpStr)
  parser.add_argument('obsModelName', type=str, help=oModelHelpStr)
  parser.add_argument('algNames', type=str, 
      help=algHelpStr + " Comma-separated if multiple needed, like 'VB,soVB'")


def addStandardVizArgsToParser(parser):
  ''' Update parser to include standard visualization arguments
  '''
  parser.add_argument('--jobnames', type=str, default='defaultjob',
        help='name of experiment whose results should be plotted')
  parser.add_argument('--taskids', type=str, default=None,
        help="int ids of tasks (trials/runs) to plot from given job." \
              + " Example: '4' or '1,2,3' or '2-6'.")
  parser.add_argument('--savefilename', type=str, default=None,
        help="location where to save figure (absolute path directory)")

          
def parse_task_ids(jobpath, taskids=None):
  ''' Return list of task ids
      Examples
      ---------
      >>> parse_task_ids(None, taskids='1-3')
      [1,2,3]
      >>> parse_task_ids(None, taskids='4')
      [4]
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
    taskids = [taskids]
  taskids.sort()
  return taskids
