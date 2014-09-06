'''
Inspector module implements the bnpy custom_hook interface,
so we can double-check calculations happening within bnpy.run 
without modifying any learning algorithm code.
'''

import numpy as np
import os
import time

from ExecuteNotebook import writeReportForTask

RecordTimes = [0]
RECORD_INTERVAL_SEC = 5

def onLapComplete(lapFrac=0, learnAlg=None, **kwargs):
  if lapFrac == learnAlg.algParams['nLap']:
    ## skip writing report on final lap, since onAlgComplete takes care of this
    return

  starttime = learnAlg.start_time
  etime = time.time() - starttime

  if etime > np.max(RecordTimes) + RECORD_INTERVAL_SEC:
    rtime = RecordTimes[-1] + RECORD_INTERVAL_SEC
    RecordTimes.append(rtime)    
    writeReportForTask(learnAlg.savedir)
    
def onAlgorithmComplete(lapFrac=0, learnAlg=None, **kwargs):
  writeReportForTask(learnAlg.savedir)