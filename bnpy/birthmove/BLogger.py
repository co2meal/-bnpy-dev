import logging
import os
import sys
from collections import defaultdict
from bnpy.util import split_str_into_fixed_width_lines

# Configure Logger
Log = None
RecentMessages = None
taskoutpath = '/tmp/'
DEFAULTLEVEL = logging.DEBUG

def pprint(msg, level=None, prefix='', linewidth=80):
    global Log
    global DEFAULTLEVEL
    global RecentMessages
    if isinstance(msg, list):
        msgs = list()
        prefixes = list()
        for ii, m_ii in enumerate(msg):
            prefix_ii = prefix[ii]
            msgs_ii = split_str_into_fixed_width_lines(m_ii,
                linewidth=linewidth-len(prefix_ii))
            msgs.extend(msgs_ii)
            prefixes.extend([prefix[ii] for i in range(len(msgs_ii))])
        for ii in range(len(msgs)):
            pprint(prefixes[ii] + msgs[ii], level=level)
        return
    if DEFAULTLEVEL == 'print':
        print msg
    if Log is None:
        return
    if level is None:
        level = DEFAULTLEVEL
    if isinstance(level, str):
        if level.count('info'):
            level = logging.INFO
        elif level.count('debug'):
            level = logging.DEBUG
    Log.log(level, msg)
    if isinstance(RecentMessages, list):
        RecentMessages.append(msg)


def startUIDSpecificLog(uid=0):
    ''' Open log file (in append mode) for specific uid.

    Post condition
    --------------
    Creates a log file specific to the given uid, 
    which will capture all subsequent log output.
    '''
    global Log
    global RecentMessages
    global taskoutpath
    if Log is None:
        return
    fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-log-by-uid-%d.txt" % (uid)))
    fh.setLevel(0)
    fh.setFormatter(logging.Formatter('%(message)s'))
    Log.addHandler(fh)
    RecentMessages = list()

def stopUIDSpecificLog(uid=0):
    ''' Close log file corresponding to specific uid.

    Post condition
    --------------
    If the specified uid has an associated log open,
    then it will be closed.
    '''
    global Log
    global RecentMessages
    if Log is None:
        return
    for i in range(len(Log.handlers)):
        fh = Log.handlers[i]
        if isinstance(fh, logging.FileHandler):
            if fh.baseFilename.count('uid-%d' % (uid)):
                fh.close()
                Log.removeHandler(fh)
                break
    RecentMessages = None

def configure(taskoutpathIN, 
        doSaveToDisk=0, doWriteStdOut=0,
        verboseLevel=0,
        summaryLevel=logging.DEBUG+1,        
        stdoutLevel=logging.DEBUG+1):
    ''' Configure this singleton Logger to write logs to disk or stdout.

    Post condition
    --------------
    Log will have at least one handler, either a null, stdout, or file handler.
    '''
    global Log
    global taskoutpath

    taskoutpath = taskoutpathIN
    Log = logging.getLogger('birthmove')

    Log.setLevel(logging.DEBUG)
    Log.handlers = []  # remove pre-existing handlers!
    formatter = logging.Formatter('%(message)s')
    # Config logger to save transcript of log messages to plain-text file
    if doSaveToDisk:
        # birth-transcript-verbose.txt logs all messages that describe births
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-transcript-verbose.txt"))
        fh.setLevel(verboseLevel)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

        # birth-transcript-summary.txt logs one summary message per lap
        fh = logging.FileHandler(
            os.path.join(
                taskoutpath,
                "birth-transcript-summary.txt"))
        fh.setLevel(summaryLevel)
        fh.setFormatter(formatter)
        Log.addHandler(fh)

    # Config logger that can write to stdout
    if doWriteStdOut:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(stdoutLevel)
        ch.setFormatter(formatter)
        Log.addHandler(ch)
    # Config null logger, avoids error messages about no handler existing
    if not doSaveToDisk and not doWriteStdOut:
        Log.addHandler(logging.NullHandler())

def makeFunctionToPrettyPrintCounts(initSS):
    from bnpy.viz.PrintTopics import count2str
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
            s += " (removed comps below minimum size of %.2f)" % (
                cleanupSizeThr)
        pprint('  ' + s)
    return pprintCountVec
