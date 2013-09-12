import argparse
import ConfigParser

def parseArgs(ConfigPaths):
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName')
  parser.add_argument('allocModelName')
  parser.add_argument('obsModelName')
  parser.add_argument('algName')
  
  for filepath in ConfigPaths:
    addArgGroupFromConfigFile( parser, filepath)
  args = parser.parse_args()
  argDict = dict( dataName=args.dataName,
                  allocModelName=args.allocModelName,
                  obsModelName=args.obsModelName,
                  algName=args.algName
                )
  for filepath in ConfigPaths:
    addArgsToDictByConfigFile( argDict, args, filepath)
  return argDict

def readConfigParser( filepath):
  config = ConfigParser.SafeConfigParser()
  config.optionxform = str
  config.read( filepath )  
  return config
  
def addArgsToDictByConfigFile( argDict, args, filepath):
  config = readConfigParser( filepath)
  for secName in config.sections():
    if secName.count("Help") > 0:
      continue
    DefDict = dict( config.items( secName ) )  
    secArgDict = dict([ (k,v) for (k,v) in vars(args).items() if k in DefDict])
    argDict[secName] = secArgDict

def addArgGroupFromConfigFile( parser, confFilePath):
  config = readConfigParser( confFilePath)
  for secName in config.sections():
    if secName.count("Help") > 0:
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

'''
Automatically determine what type the argument takes based on its default value
  Decides between int/float based on whether a decimal point is included!
'''
def getType( defVal ):
  if defVal.count('.') > 0:
    return float
  if defVal == 'true' or defVal == 'True':
    return True
  if defVal == 'false' or defVal == 'False':
    return False
  try:
    int( defVal )
    return int
  except Exception:
    try:
      float(defVal)
      return float
    except Exception:
      return str
    
if __name__ == '__main__':
  print parseArgs( ['config/allocmodel.conf','config/obsmodel.conf'])

