import logging
import os
import sys
from collections import defaultdict

# Configure Logger
Log = None
Cache = defaultdict(lambda: list())
CacheOrder = list()


def log(msg, level='debug'):
    if Log is None:
        return
    if level == 'info':
        Log.info(msg)
    elif level == 'moreinfo':
        Log.log(15, msg)
    elif level == 'debug':
        Log.debug(msg)
    else:
        Log.log(level, msg)


def logStartPrep(lapFrac):
    msg = '=' * (50)
    msg = msg + ' lap %.2f Target Selection' % (lapFrac)
    log(msg, 'moreinfo')


def logStartMove(lapFrac, moveID, nMoves):
    msg = '=' * (50)
    msg = msg + ' lap %.2f %d/%d' % (lapFrac, moveID, nMoves)
    log(msg, 'moreinfo')


def logPhase(title):
    title = '.' * (50 - len(title)) + ' %s' % (title)
    log(title, 'debug')


def logPosVector(vec, fmt='%8.1f', Nmax=10, label='', level='debug'):
    if Log is None:
        return
    vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
    if len(label) > 0:
        log(vstr + " | " + label, level)
    else:
        log(vstr, level)


def logProbVector(vec, fmt='%8.4f', Nmax=10, level='debug'):
    if Log is None:
        return
    vstr = ' '.join([fmt % (x) for x in vec[:Nmax]])
    log(vstr, level)

# Advanced caching
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

# Configuration
###########################################################


def configure(taskoutpath, doSaveToDisk=0, doWriteStdOut=0):
    global Log
    Log = logging.getLogger('birthmove')

    Log.setLevel(logging.DEBUG)
    Log.handlers = []  # remove pre-existing handlers!
    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        # birth-vtranscript.txt logs everything
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-vtranscript.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

        # birth-transcript.txt logs high-level messages
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-transcript.txt"))
        fh.setLevel(logging.DEBUG + 1)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

    # Config logger that can write to stdout
    if doWriteStdOut:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        Log.addHandler(ch)
    # Config null logger, avoids error messages about no handler existing
    if not doSaveToDisk and not doWriteStdOut:
        Log.addHandler(logging.NullHandler())
