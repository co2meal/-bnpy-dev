import numpy as np
import glob
import os

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
  elif val.count('rand') or val.count('true') or val.count('spectral'):
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


def filterJobs(jpathPattern, varsToPlot=[], **reqKwArgs):
  print jpathPattern

  for key in reqKwArgs:
    try:
      reqKwArgs[key] = float(reqKwArgs[key])
    except:
      pass # keep as string

  jpathList = glob.glob(jpathPattern)
  print jpathList
  print reqKwArgs.keys()

  keepListP = list() # list of paths to keep
  keepListD = list() # list of dicts to keep (one for each path)
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
    if doKeep:
      keepListP.append(jpath)
      keepListD.append(jdict)
  print keepListP

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

  print varKeys
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

  return keepListFinal, legNames

if __name__ == '__main__':
  #keepJobs, legNames = filterJobs('/results/AdmixAsteriskK8/rmergetest*', 
  #                                K=80)
  keepJobs, legNames = filterJobs('/results/AsteriskK8/demo*')
  for j in keepJobs:
    print j
  print legNames