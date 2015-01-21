'''
SpeakDiar.py

21 audio recordings of academic conferences making up the NIST speaker 
  diarization dataset, created to asses the ability of different models to 
  segment speech data into unique speakers.

The 21 recordings are meant to be trained on independently, so get_data() takes
  a meetingNum parameter (default 1) which determines which recording number
  will be loaded into bnpy.  The meeting number can be changed with the argument
  --meetingNum 3
'''

import numpy as np
from bnpy.data import GroupXData
import scipy.io
import os

fileNames = [
'AMI_20041210-1052_Nmeans25features_SpNsp.mat',
'AMI_20050204-1206_Nmeans25features_SpNsp.mat',
'CMU_20050228-1615_Nmeans25features_SpNsp.mat',
'CMU_20050301-1415_Nmeans25features_SpNsp.mat',
'CMU_20050912-0900_Nmeans25features_SpNsp.mat',
'CMU_20050914-0900_Nmeans25features_SpNsp.mat',
'EDI_20050216-1051_Nmeans25features_SpNsp.mat',
'EDI_20050218-0900_Nmeans25features_SpNsp.mat',
'ICSI_20000807-1000_Nmeans25features_SpNsp.mat',
'ICSI_20010208-1430_Nmeans25features_SpNsp.mat',
'LDC_20011116-1400_Nmeans25features_SpNsp.mat',
'LDC_20011116-1500_Nmeans25features_SpNsp.mat',
'NIST_20030623-1409_Nmeans25features_SpNsp.mat',
'NIST_20030925-1517_Nmeans25features_SpNsp.mat',
'NIST_20051024-0930_Nmeans25features_SpNsp.mat',
'NIST_20051102-1323_Nmeans25features_SpNsp.mat',
'TNO_20041103-1130_Nmeans25features_SpNsp.mat',
'VT_20050304-1300_Nmeans25features_SpNsp.mat',
'VT_20050318-1430_Nmeans25features_SpNsp.mat',
'VT_20050623-1400_Nmeans25features_SpNsp.mat',
'VT_20051027-1400_Nmeans25features_SpNsp.mat']


datasetdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
  raise ValueError('CANNOT FIND DATASET DIRECTORY:\n' + datasetdir)

def get_data_info():
  global meetNum
  return 'Pre-processed audio data from NIST meeting %s (meeting %d / 21)' \
         % (fileNames[meetNum-1][:-27], meetNum)
  
def get_short_name():
  global meetNum
  return 'SpeakDiar'+str(meetNum)


def get_data(meetingNum=1, **kwargs):
  '''
  Returns a GroupXData object corresponding to the passed in meetingNum, which
  indexes the fileNames listed above from 1 to 21
  '''
  global meetNum
  meetNum = meetingNum

  if meetNum <= 0 or meetNum > len(fileNames):
    raise ValueError('Bad value for meetingNum: %s' % (meetNum))

  matfilepath = os.path.join(datasetdir, 'rawData',
                             'speakerDiarizationData', fileNames[meetNum-1])
  if not os.path.isfile(matfilepath):
    raise ValueError('CANNOT FIND SPEAKDIAR DATASET MAT FILE:\n' + matfilepath)

  Data = GroupXData.read_from_mat(matfilepath)
  Data.summary = get_data_info()
  Data.name = get_short_name()

  Data.fileNames = [fileNames[meetNum-1]]
  return Data

############################################# Extract what we need from the
############################################# given NIST .mat files

def saveMatFile(dataPath):
  for file in fileNames:
    fpath = os.path.join(dataPath, file)
    data = scipy.io.loadmat(fpath)
    X = np.transpose(data['u'])
    TrueZ = data['zsub']
    doc_range = [0, np.size(TrueZ)]
    scipy.io.savemat(os.path.join(os.path.expandvars('$BNPYDATADIR/rawData/speakerDiarizationData'), file), {'X' : X, 'TrueZ' : TrueZ, 'doc_range' : doc_range})
                                                      
    
