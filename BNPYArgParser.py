import argparse
import ConfigParser

ConfigPaths=['config/allocmodel.conf','config/obsmodel.conf', 
             'config/init.conf', 'config/inference.conf', 'config/output.conf']
OnlineDataConfigPath = 'config/onlinedata.conf'

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName')
  parser.add_argument('allocModelName')
  parser.add_argument('obsModelName')
  parser.add_argument('algName')
  parser.add_argument('--moves', type=str, default=None)
  args, unkargs = parser.parse_known_args()
  if args.algName.count('o') > 0:
    ConfigPaths.append(OnlineDataConfigPath)

  # Loop over all ConfigPaths, adding expected arguments to the parser object
  for filepath in ConfigPaths:
    if filepath.count('inference') > 0:
      addArgGroupFromConfigFile(parser, filepath, args.algName)
      if args.moves is not None:
        args.moves = args.moves.split(',')
        for movename in args.moves:
          addArgGroupFromConfigFile(parser, filepath, movename)      
    elif filepath.count('allocmodel') > 0:
      addArgGroupFromConfigFile(parser, filepath, args.allocModelName)
    elif filepath.count('obsmodel') > 0:
      addArgGroupFromConfigFile(parser, filepath, args.obsModelName)
    else:
      addArgGroupFromConfigFile(parser, filepath)
  args = parser.parse_args()
  argDict = dict( dataName=args.dataName,
                  allocModelName=args.allocModelName,
                  obsModelName=args.obsModelName,
                  algName=args.algName
                )

  # Loop over all ConfigPaths, adding key/value pairs to argDict
  # from either user-provided args (priority) or defaults in the configfile
  for filepath in ConfigPaths:
    addArgsToDictByConfigFile(argDict, args, filepath)
  
  return argDict

def readConfigParser( filepath):
  config = ConfigParser.SafeConfigParser()
  config.optionxform = str
  config.read( filepath )  
  return config
  
def addArgsToDictByConfigFile(argDict, args, filepath):
  config = readConfigParser(filepath)
  for secName in config.sections():
    if secName.count("Help") > 0:
      continue
    if not doProcessSectionName(filepath, secName, args):
      continue
    DefDict = dict( config.items( secName ) )  
    secArgDict = dict([ (k,v) for (k,v) in vars(args).items() if k in DefDict])
    argDict[secName] = secArgDict

def doProcessSectionName(fileName, secName, args):
  if fileName.count('inference') > 0:
    if secName == args.algName:
      return True
    elif args.moves is not None and secName in args.moves:
      return True
    else:
      return False
  if fileName.count('alloc'):
    if secName == args.allocModelName:
      return True
    return False
  if fileName.count('obs'):
    if secName == args.obsModelName:
      return True
    return False
  else:
    return True

def addArgGroupFromConfigFile(parser, confFilePath, targetSectionName=None):
  config = readConfigParser( confFilePath)
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
      
    group = parser.add_argument_group( secName )    
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

def getType( defVal ):
  ''' Auto determine what type the argument takes based on its default value.
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
    int( defVal )
    return int
  except Exception:
    pass
  try:
    float(defVal)
    return float
  except Exception:
    return str
    
if __name__ == '__main__':
  print parseArgs( ['config/allocmodel.conf','config/obsmodel.conf'])

