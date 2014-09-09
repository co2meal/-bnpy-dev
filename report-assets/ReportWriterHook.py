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
RECORD_INTERVAL_SEC = 10 * 60 # Every ten minutes

def onLapComplete(lapFrac=0, learnAlg=None, **kwargs):
  print "WRITING REPORT | AT LAP %.2f" % (lapFrac)
  if lapFrac == learnAlg.algParams['nLap']:
    ## skip writing report on final lap, since onAlgComplete takes care of this
    return

  starttime = learnAlg.start_time
  etime = time.time() - starttime

  if etime > np.max(RecordTimes) + RECORD_INTERVAL_SEC:
    rtime = RecordTimes[-1] + RECORD_INTERVAL_SEC
    RecordTimes.append(rtime)
    try:  
      writeReportForTask(learnAlg.savedir)
    except:
      print 'CAUGHT ERROR IN writeReportForTask!'
      print str(e)
    
def onAlgorithmComplete(lapFrac=0, learnAlg=None, **kwargs):
  print "WRITING REPORT | FINAL"
  try:
    writeReportForTask(learnAlg.savedir)
  except:
    print 'CAUGHT ERROR IN writeReportForTask!'
    print str(e)
