import numpy as np
import glob
import os

from collections import defaultdict
from bnpy.ioutil import BNPYArgParser

def findKeysWithDiffVals(dA, dB):
  ''' Find subset of keys in dicts dA, dB that are in both dicts, with diff vals
  '''
  keepKeys = list()
  for key in dA:
    if key in dB:
      if dA[key] != dB[key]:
        keepKeys.append(key)
  return keepKeys

def mapToKey(val):
  if val.count('DP') or val.count('HDP'):
    return 'allocModelName'
  elif val.count('rand') or val.count('true') or val.count('contig') \
       or val.count('sacb') \
       or val.count('spectral') or val.count('kmeans') or val.count('plusplus'):
    return 'initname'
  return None

def jpath2jdict(jpath):
  '''
      Example
      ---------
      >> jpath2dict('abc-nBatch=10-randexamples-HDP')
      dict(UNK=abc, nBatch=10, initname=randexamples, allocModelName=HDP)
  '''
  basename = jpath.split(os.path.sep)[-1]
  fields = basename.split('-')
  D = dict()
  for fID, field in enumerate(fields):
    if field.count('=') == 1:
      key, val = field.split('=')
      try:
        D[key] = float(val)
      except Exception as e:
        D[key] = str(val)
    else:
      val = field
      key = mapToKey(val)
      if key is not None:
        D[key] = val
      else:
        D['field' + str(fID+1)] = val
  return D

## kwargs that arent needed for any job pattern matching
SkipKeys = ['taskids', 'savefilename', 'fileSuffix', 'xvar', 'yvar']

def filterJobs(jpathPattern, verbose=0, **reqKwArgs):
  for key in SkipKeys:
    if key in reqKwArgs:
      del reqKwArgs[key]

  if not jpathPattern.endswith('*'):
    jpathPattern += '*'

  jpathdir = os.path.sep.join(jpathPattern.split(os.path.sep)[:-1] )
  if not os.path.isdir(jpathdir):
    raise ValueError('Not valid directory:\n %s' % (jpathdir))

  jpathList = glob.glob(jpathPattern)
  
  if verbose:
    print 'Looking for jobs with pattern:'
    print jpathPattern
    print '%d candidates found (before filtering by keywords)' % (len(jpathList))
    
  if len(jpathList) == 0:
    raise ValueError('No matching jobs found.')

  for key in reqKwArgs:
    try:
      reqKwArgs[key] = float(reqKwArgs[key])
    except:
      pass # keep as string

  if verbose:
    print '\nRequirements:'
    for key in reqKwArgs:
      print '%s = %s' % (key, reqKwArgs[key])


  keepListP = list() # list of paths to keep
  keepListD = list() # list of dicts to keep (one for each path)
  reqKwMatches = defaultdict(int)
  for jpath in jpathList:
    jdict = jpath2jdict(jpath)
    doKeep = True
    for reqkey in reqKwArgs:
      if reqkey not in jdict:
        doKeep = False
        continue
      reqval = reqKwArgs[reqkey]
      if jdict[reqkey] != reqval:
        doKeep = False
      else:
        reqKwMatches[reqkey] += 1
    if doKeep:
      keepListP.append(jpath)
      keepListD.append(jdict)
 
  if len(keepListP) == 0:
    for reqkey in reqKwArgs:
      if reqKwMatches[reqkey] == 0:
        raise ValueError('BAD REQUIRED PARAMETER.\n'
              + 'No matches found for %s=%s: ' % (reqkey, reqKwArgs[reqkey]))

  if verbose:
    print '\nCandidates matching requirements'
    for p in keepListP:
      print p.split(os.path.sep)[-1]

  ## Figure out intelligent labels for the final jobs
  K = len(keepListD)
  varKeys = set()
  for kA in xrange(K):
    for kB in xrange(kA+1, K):
      varKeys.update(findKeysWithDiffVals(keepListD[kA], keepListD[kB]))
  varKeys = [x for x in varKeys]

  RangeMap = dict()
  for key in varKeys:
    RangeMap[key] = set()
    for jdict in keepListD:
      RangeMap[key].add(jdict[key])
    RangeMap[key] = [x for x in sorted(RangeMap[key])] # to list

  if len(varKeys) > 1:
    print 'ERROR! Need to constrain more variables'
    for key in RangeMap:
      print key, RangeMap[key]
    raise ValueError('ERROR! Need to constrain more variables')

  elif len(varKeys) == 1:
    plotkey = varKeys[0]
    if type(RangeMap[plotkey][0]) == str:
      legNames = ['%s' % (x) for x in RangeMap[plotkey]]
    else:
      legNames = ['%s=%s' % (plotkey, x) for x in RangeMap[plotkey]]

    ## Build list of final jpaths in order of decided legend
    keepListFinal = list()
    for x in RangeMap[plotkey]:
      for jID, jdict in enumerate(keepListD):
        if jdict[plotkey] == x:
          keepListFinal.append(keepListP[jID])
  else:
    keepListFinal = keepListP[:1]
    legNames = [None]

  if verbose:
    print '\nLegend entries for selected jobs (auto-selected)'
    for name in legNames:
      print name

  return keepListFinal, legNames

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('dataName', default='AsteriskK8')
  parser.add_argument('jobName', default='bm')
  args, unkList = parser.parse_known_args()
  reqDict = BNPYArgParser.arglist_to_kwargs(unkList)

  jpath = os.path.join(os.environ['BNPYOUTDIR'],
                              args.dataName,
                              args.jobName)

  keepJobs, legNames = filterJobs(jpath, verbose=1, **reqDict)

