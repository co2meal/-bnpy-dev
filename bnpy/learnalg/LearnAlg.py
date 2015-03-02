'''
Classes
-------
LearnAlg
    Defines some generic routines for
        * saving global parameters
        * assessing convergence
        * printing progress updates to stdout
        * recording run-time
'''
import numpy as np
import time
import logging
import os
import sys
import scipy.io

from bnpy.ioutil import ModelWriter
from bnpy.util import isEvenlyDivisibleFloat

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)


class LearnAlg(object):
    """ Abstract base class for learning algorithms that train HModels.

    Attributes
    ------
    savedir : str
        file system path to directory where files are saved
    seed : int
        seed used for random initialization
    PRNG : np.random.RandomState
        random number generator
    algParams : dict
        keyword parameters controlling algorithm behavior
        default values for each algorithm live in config/
        Can be overrided by keyword arguments specified by user
    outputParams : dict
        keyword parameters controlling saving to file / logs
    """

    def __init__(self, savedir=None, seed=0,
                 algParams=dict(), outputParams=dict()):
        ''' Constructs and returns a LearnAlg object
        '''
        if isinstance(savedir, str):
            self.savedir = os.path.splitext(savedir)[0]
        else:
            self.savedir = None
        self.seed = int(seed)
        self.PRNG = np.random.RandomState(self.seed)
        self.algParams = algParams
        self.outputParams = outputParams
        self.TraceLaps = set()
        self.evTrace = list()
        self.SavedIters = set()
        self.PrintIters = set()
        self.totalDataUnitsProcessed = 0
        self.status = 'active. not converged.'

        self.algParamsLP = dict()
        for k, v in algParams.items():
            if k.count('LP') > 0:
                self.algParamsLP[k] = v

    def fit(self, hmodel, Data):
        ''' Execute learning algorithm to train hmodel on Data.

        This method is extended by any subclass of LearnAlg

        Returns
        -------
        Info : dict of diagnostics about this run
        '''
        pass

    def set_random_seed_at_lap(self, lap):
        ''' Set internal random generator based on current lap.

        Reset the seed deterministically for each lap.
        using combination of seed attribute (unique to this run),
        and the provided lap argument. This allows reproducing
        exact values from this run later without starting over.

        Post Condition
        ------
        self.PRNG rest to new random seed.
        '''
        if isEvenlyDivisibleFloat(lap, 1.0):
            self.PRNG = np.random.RandomState(self.seed + int(lap))

    def set_start_time_now(self):
        ''' Record start time (in seconds since 1970)
        '''
        self.start_time = time.time()

    def updateNumDataProcessed(self, N):
        ''' Update internal count of total number of data observations processed.
            Each lap thru dataset of size N, this should be updated by N
        '''
        self.totalDataUnitsProcessed += N

    def get_elapsed_time(self):
        ''' Returns float of elapsed time (in seconds) since this object's
            set_start_time_now() method was called
        '''
        return time.time() - self.start_time

    def buildRunInfo(self, **kwargs):
        ''' Create dict of information about the current run.

        Returns
        ------
        Info : dict
            contains information about completed run.
        '''
        # Convert TraceLaps from set to 1D array, sorted in ascending order
        lapTrace = np.asarray(list(self.TraceLaps))
        lapTrace.sort()
        evTrace = np.asarray(self.evTrace)
        return dict(status=self.status,
                    evTrace=evTrace,
                    lapTrace=lapTrace,
                    elapsedTimeInSec=time.time() - self.start_time,
                    **kwargs)

    def hasMove(self, moveName):
        if moveName in self.algParams:
            return True
        return False

    def verify_evidence(self, evBound=0.00001, prevBound=0,
                        lapFrac=None):
        ''' Compare current and previous evidence (ELBO) values,
            verify that (within numerical tolerance) increases monotonically
        '''
        if np.isnan(evBound):
            raise ValueError("Evidence should never be NaN")
        if np.isinf(prevBound):
            return False
        isIncreasing = prevBound <= evBound

        thr = self.algParams['convergeThrELBO']
        isWithinTHR = np.abs(evBound - prevBound) < thr
        mLPkey = 'doMemoizeLocalParams'
        if not isIncreasing and not isWithinTHR:
            serious = True
            if mLPkey in self.algParams and not self.algParams[mLPkey]:
                warnMsg = 'ev decreased when doMemoizeLocalParams=0'
                warnMsg += ' (so monotonic increase not guaranteed)\n'
                serious = False
            else:
                warnMsg = 'evidence decreased!\n'
            warnMsg += '    prev = % .15e\n' % (prevBound)
            warnMsg += '     cur = % .15e\n' % (evBound)
            if lapFrac is None:
                prefix = "WARNING: "
            else:
                prefix = "WARNING @ %.3f: " % (lapFrac)

            if serious or not self.algParams['doShowSeriousWarningsOnly']:
                Log.error(prefix + warnMsg)

    def isSaveDiagnosticsCheckpoint(self, lap, nMstepUpdates):
        ''' Answer True/False whether to save trace stats now
        '''
        traceEvery = self.outputParams['traceEvery']
        if traceEvery <= 0:
            return False
        return isEvenlyDivisibleFloat(lap, traceEvery) \
            or nMstepUpdates < 3 \
            or lap in self.TraceLaps

    def saveDiagnostics(self, lap, SS, evBound, ActiveIDVec=None):
        ''' Save trace stats to disk
        '''
        if lap in self.TraceLaps:
            return
        self.TraceLaps.add(lap)

        # Record current evidence
        self.evTrace.append(evBound)

        # Exit here if we're not saving to disk
        if self.savedir is None:
            return

        # Record current state to plain-text files
        with open(self.mkfile('laps.txt'), 'a') as f:
            f.write('%.4f\n' % (lap))
        with open(self.mkfile('evidence.txt'), 'a') as f:
            f.write('%.9e\n' % (evBound))
        with open(self.mkfile('times.txt'), 'a') as f:
            f.write('%.3f\n' % (self.get_elapsed_time()))
        with open(self.mkfile('K.txt'), 'a') as f:
            f.write('%d\n' % (SS.K))
        with open(self.mkfile('total-data-processed.txt'), 'a') as f:
            f.write('%d\n' % (self.totalDataUnitsProcessed))

        # Record active counts in plain-text files
        counts = None
        try:
            counts = SS.N
        except AttributeError:
            counts = SS.SumWordCounts
        assert counts.ndim == 1
        counts = np.asarray(counts, dtype=np.float32)
        np.maximum(counts, 0, out=counts)
        with open(self.mkfile('ActiveCounts.txt'), 'a') as f:
            flatstr = ' '.join(['%.3f' % x for x in counts])
            f.write(flatstr + '\n')

        with open(self.mkfile('ActiveIDs.txt'), 'a') as f:
            if ActiveIDVec is None:
                ActiveIDVec = np.arange(SS.K)
            flatstr = ' '.join(['%d' % x for x in ActiveIDVec])
            f.write(flatstr + '\n')

    def isCountVecConverged(self, Nvec, prevNvec):
        if Nvec.size != prevNvec.size:
            # Warning: the old value of maxDiff is still used for printing
            return False

        maxDiff = np.max(np.abs(Nvec - prevNvec))
        isConverged = maxDiff < self.algParams['convergeThr']
        self.ConvergeInfo = dict(isConverged=isConverged,
                                 maxDiff=maxDiff
                                 )
        return isConverged

    def isSaveParamsCheckpoint(self, lap, nMstepUpdates):
        ''' Answer True/False whether to save full model now
        '''
        saveEvery = self.outputParams['saveEvery']
        if saveEvery <= 0 or self.savedir is None:
            return False
        return isEvenlyDivisibleFloat(lap, saveEvery) \
            or nMstepUpdates < 3 \
            or np.allclose(lap, 1.0) \
            or np.allclose(lap, 2.0) \
            or np.allclose(lap, 4.0) \
            or np.allclose(lap, 8.0)

    def saveParams(self, lap, hmodel, SS=None):
        ''' Save current model to disk
        '''
        if lap in self.SavedIters or self.savedir is None:
            return
        self.SavedIters.add(lap)
        prefix = ModelWriter.makePrefixForLap(lap)
        with open(self.mkfile('laps-saved-params.txt'), 'a') as f:
            f.write('%.4f\n' % (lap))
        if self.outputParams['doSaveFullModel']:
            ModelWriter.save_model(
                hmodel, self.savedir, prefix,
                doSavePriorInfo=np.allclose(lap, 0.0),
                doLinkBest=True,
                doSaveObsModel=self.outputParams['doSaveObsModel'])
        if self.outputParams['doSaveTopicModel']:
            ModelWriter.saveTopicModel(hmodel, SS, self.savedir, prefix)

    def mkfile(self, fname):
        """ Create full system path for provided file basename.

        Returns
        -------
        fpath : str

        Examples
        -------
        >> mkfile("K.txt")
        "/path/to/output/K.txt"
        """
        return os.path.join(self.savedir, fname)

    def setStatus(self, lapFrac, isConverged):
        nLapsCompleted = lapFrac - self.algParams['startLap']

        nLap = self.algParams['nLap']
        minLapReq = np.minimum(nLap, self.algParams['minLaps'])
        minLapsCompleted = nLapsCompleted >= minLapReq
        if isConverged and minLapsCompleted:
            self.status = "done. converged."
        elif isConverged:
            self.status = "active. converged but minLaps unfinished."
        elif nLapsCompleted < nLap:
            self.status = "active. not converged."
        else:
            self.status = "done. not converged. max laps thru data exceeded."

        if self.savedir is not None:
            with open(self.mkfile('status.txt'), 'w') as f:
                f.write(self.status + '\n')

    def isLogCheckpoint(self, lap, nMstepUpdates):
        ''' Answer True/False whether to save full model now
        '''
        printEvery = self.outputParams['printEvery']
        if printEvery <= 0:
            return False
        return isEvenlyDivisibleFloat(lap, printEvery) or nMstepUpdates < 3

    def printStateToLog(self, hmodel, evBound, lap, iterid,
                        isFinal=0, rho=None):
        """ Print state of provided model and progress variables to log.
        """
        if hasattr(self, 'ConvergeInfo') and 'maxDiff' in self.ConvergeInfo:
            countStr = 'Ndiff%9.3f' % (self.ConvergeInfo['maxDiff'])
        else:
            countStr = ''

        if rho is None:
            rhoStr = ''
        else:
            rhoStr = 'lrate %.4f' % (rho)

        if iterid == lap:
            lapStr = '%7d' % (lap)
        else:
            lapStr = '%7.3f' % (lap)
        maxLap = self.algParams['nLap'] + self.algParams['startLap']
        maxLapStr = '%d' % (maxLap)

        logmsg = '  %s/%s after %6.0f sec. | K %4d | ev % .9e | %s %s'
        logmsg = logmsg % (lapStr,
                           maxLapStr,
                           self.get_elapsed_time(),
                           hmodel.allocModel.K,
                           evBound,
                           countStr,
                           rhoStr)

        if iterid not in self.PrintIters:
            self.PrintIters.add(iterid)
            Log.info(logmsg)
        if isFinal:
            Log.info('... %s' % (self.status))

    def print_msg(self, msg):
        ''' Prints a string msg to the log for this training experiment.

        Avoids need to import all logging utilities into a subclass.
        '''
        Log.info(msg)

    # Checkpoints
    #########################################################
    def isFirstBatch(self, lapFrac):
        ''' Returns True/False if at first batch in the current lap.
        '''
        if self.lapFracInc == 1.0:  # Special case, nBatch == 1
            isFirstBatch = True
        else:
            lapRem = lapFrac - np.floor(lapFrac)
            isFirstBatch = np.allclose(lapRem, self.lapFracInc)
        return isFirstBatch

    def isLastBatch(self, lapFrac):
        ''' Returns True/False if at last batch in the current lap.
        '''
        return lapFrac % 1 == 0

    def do_birth_at_lap(self, lapFrac):
        ''' Returns True/False for whether birth happens at given lap
        '''
        if 'birth' not in self.algParams:
            return False
        nLapTotal = self.algParams['nLap']
        frac = self.algParams['birth']['fracLapsBirth']
        if lapFrac > nLapTotal:
            return False
        return (nLapTotal <= 5) or (lapFrac <= np.ceil(frac * nLapTotal))

    def doMergePrepAtLap(self, lapFrac):
        if 'merge' not in self.algParams:
            return False
        return lapFrac > self.algParams['merge']['mergeStartLap'] \
            and self.isFirstBatch(lapFrac)

    def eval_custom_func(self, isFinal=0, isInitial=0, lapFrac=0, **kwargs):
        ''' Evaluates a custom hook function
        '''
        cFuncPath = self.outputParams['customFuncPath']
        if cFuncPath is None or cFuncPath == 'None':
            return None

        cFuncArgs_string = self.outputParams['customFuncArgs']
        nLapTotal = self.algParams['nLap']
        if isinstance(cFuncPath, str):
            pathParts = cFuncPath.split(os.path.sep)
            cFuncDir = os.path.sep.join(pathParts[:-1])
            cFuncModName = pathParts[-1].split('.py')[0]
            sys.path.append(cFuncDir)
            cFuncModule = __import__(cFuncModName, fromlist=[])
        else:
            cFuncModule = cFuncPath  # directly passed in as object

        kwargs['lapFrac'] = lapFrac
        kwargs['isFinal'] = isFinal
        kwargs['isInitial'] = isInitial
        if isInitial:
            kwargs['lapFrac'] = 0
            kwargs['iterid'] = 0

        if hasattr(cFuncModule, 'onBatchComplete') and not isFinal:
            cFuncModule.onBatchComplete(args=cFuncArgs_string, **kwargs)
        if hasattr(cFuncModule, 'onLapComplete') \
           and isEvenlyDivisibleFloat(lapFrac, 1.0) and not isFinal:
            cFuncModule.onLapComplete(args=cFuncArgs_string, **kwargs)
        if hasattr(cFuncModule, 'onAlgorithmComplete') \
           and isFinal:
            cFuncModule.onAlgorithmComplete(args=cFuncArgs_string, **kwargs)

    def saveDebugStateAtBatch(self, name, batchID, LPchunk=None, SS=None,
                              SSchunk=None, hmodel=None,
                              Dchunk=None):
        if self.outputParams['debugBatch'] == batchID:
            debugLap = self.outputParams['debugLap']
            debugLapBuffer = self.outputParams['debugLapBuffer']
            import joblib
            if self.lapFrac < 1:
                joblib.dump(dict(Dchunk=Dchunk),
                            os.path.join(self.savedir, 'Debug-Data.dump'))
            belowWindow = self.lapFrac < debugLap - debugLapBuffer
            aboveWindow = self.lapFrac > debugLap + debugLapBuffer
            if belowWindow or aboveWindow:
                return
            filename = 'DebugLap%04.0f-%s.dump' % (np.ceil(self.lapFrac), name)
            SaveVars = dict(LP=LPchunk, SS=SS, hmodel=hmodel,
                            SSchunk=SSchunk,
                            lapFrac=self.lapFrac)
            joblib.dump(SaveVars, os.path.join(self.savedir, filename))
            if self.lapFrac < 1:
                joblib.dump(dict(Dchunk=Dchunk),
                            os.path.join(self.savedir, 'Debug-Data.dump'))


def makeDictOfAllWorkspaceVars(**kwargs):
    ''' Create dict of all active variables in workspace

    Necessary to avoid call to self.

    Returns
    ------
    kwargs : dict
        key/value for every variable in the workspace.
    '''
    if 'self' in kwargs:
        kwargs['learnAlg'] = kwargs.pop('self')
    if 'lap' in kwargs:
        kwargs['lapFrac'] = kwargs['lap']
    for key in kwargs.keys():
        if key.startswith('_'):
            kwargs.pop(key)
    return kwargs
