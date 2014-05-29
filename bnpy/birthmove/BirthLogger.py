import logging
import os
import sys
from collections import defaultdict

# Configure Logger
Log = None
Cache = defaultdict(lambda: list())
CacheOrder = list()

def log(msg):
  if Log is None:
    return
  Log.info(msg)

def logStartMove(lapFrac, moveID, nMoves):
  msg = '=' * (50) 
  msg = msg + ' lap %.2f %d/%d' % (lapFrac, moveID, nMoves)
  log(msg)

def logPhase(title):
  title = '.'*(50-len(title)) + ' %s' % (title)
  log(title)

def logPosVector(vec, fmt='%8.1f', Nmax=10):
  if Log is None:
    return
  vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
  Log.info(vstr)

def logProbVector(vec, fmt='%8.4f', Nmax=10):
  if Log is None:
    return
  vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
  Log.info(vstr)

########################################################### Advanced caching
###########################################################
def addToCache(cID, msg):
  if cID not in Cache:
    CacheOrder.append(cID)
  Cache[cID].append(msg)

def writeNextCacheToLog():
  cID = CacheOrder.pop(0)
  for line in Cache[cID]:
    log(line)

def writePlanToLog(Plan):
  for line in Plan['log']:
    log(line)

'''
, Ntop=10):
  for Plan in Plans:
    if 'ktarget' in Plan and Plan['ktarget'] is not None:
      pass
    elif 'targetWordID' in Plan and Plan['targetWordID'] is not None:
      pass    
    elif 'targetWordFreq' in Plan and Plan['targetWordFreq'] is not None:
      wordFreq = Plan['targetWordFreq']
      topWords = np.argsort(-1*wordFreq)[:Ntop]
      topWordStr = ' '.join(topWords)
      Cache.append(topWordStr)
'''

########################################################### Configuration
###########################################################
def configure(taskoutpath, doSaveToDisk=0, doWriteStdOut=0):
  global Log
  Log = logging.getLogger('birthmove')

  Log.setLevel(logging.DEBUG)
  Log.handlers = [] # remove pre-existing handlers!
  formatter = logging.Formatter('%(message)s')
  ###### Config logger to save transcript of log messages to plain-text file  
  if doSaveToDisk:
    fh = logging.FileHandler(os.path.join(taskoutpath,"birth-transcript.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    Log.addHandler(fh)
  ###### Config logger that can write to stdout
  if doWriteStdOut:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    Log.addHandler(ch)
  ##### Config null logger, avoids error messages about no handler existing
  if not doSaveToDisk and not doWriteStdOut:
    Log.addHandler(logging.NullHandler())
