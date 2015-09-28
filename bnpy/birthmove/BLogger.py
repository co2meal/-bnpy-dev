import logging
import os
import sys
from collections import defaultdict
from bnpy.viz.PrintTopics import vec2str, count2str

# Configure Logger
Log = None
taskoutpath = None

def pprint(msg, level=logging.INFO):
    global Log
    if Log is None:
        return
    if isinstance(level, str):
        if level.count('info'):
            level = logging.INFO
        elif level.count('debug'):
            level = logging.DEBUG
    Log.log(level, msg)

def startUIDSpecificLog(uid=0):
    ''' Open log file (in append mode) for specific uid.

    Post condition
    --------------
    Creates a log file specific to the given uid, 
    which will capture all subsequent log output.
    '''
    global taskoutpath
    fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-log-by-uid-%d.txt" % (uid)))
    fh.setLevel(0)
    fh.setFormatter(logging.Formatter('%(message)s'))
    Log.addHandler(fh)

def stopUIDSpecificLog(uid=0):
    ''' Close log file corresponding to specific uid.

    Post condition
    --------------
    If the specified uid has an associated log open,
    then it will be closed.
    '''
    for i in range(len(Log.handlers)):
        fh = Log.handlers[i]
        if isinstance(fh, logging.FileHandler):
            if fh.baseFilename.count('uid-%d' % (uid)):
                fh.close()
                Log.removeHandler(fh)
                break

def configure(taskoutpathIN, doSaveToDisk=0, doWriteStdOut=0):
    global Log
    global taskoutpath

    taskoutpath = taskoutpathIN
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
                "birth-transcript-verbose.txt"))
        fh.setLevel(0)
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
        ch.setLevel(logging.DEBUG+1)
        ch.setFormatter(formatter)
        Log.addHandler(ch)
    # Config null logger, avoids error messages about no handler existing
    if not doSaveToDisk and not doWriteStdOut:
        Log.addHandler(logging.NullHandler())

def makeFunctionToPrettyPrintCounts(initSS):
    def pprintCountVec(SS, uids=initSS.uids, 
                       cleanupMassRemoved=None, 
                       cleanupSizeThr=None, 
                       uidpairsToAccept=None):
        s = ''
        emptyVal = '     '
        for uid in uids:
            try:
                k = SS.uid2k(uid)
                s += ' ' + count2str(SS.getCountVec()[k])
            except:
                didWriteThisUID = False
                if uidpairsToAccept:
                    for uidA, uidB in uidpairsToAccept:
                        if uidB == uid:
                            s += ' m' + '%3d' % (uidA)
                            didWriteThisUID = True
                            break
                if not didWriteThisUID:
                    s += emptyVal
        if cleanupSizeThr:
            s += " (removed %d units from comps below minimum size of %d)" % (cleanupMassRemoved, cleanupSizeThr)
        pprint('  ' + s, 'info')
    return pprintCountVec
