'''
SpeakDiar.py

Speaker diaritization dataset as used in 'A Sticky HDP-HMM With Application to
  Speaker Diarization' by Fox et. al. for details.
'''

import numpy as np
from bnpy.data import SeqXData, MinibatchIterator
import scipy.io

files = ['AMI_20041210-1052_Nmeans25features_SpNsp.mat',
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

def get_minibatch_iterator(seed=8675309, dataorderseed=0, nBatch=3, nObsBatch=2, nObsTotal=25000, nLap=1, startLap=0, **kwargs):
    print 'umm'


def get_XZ():
    X = None
    Z = None
    seqInds = [0]

    for file in xrange(np.size(files)):
        data = scipy.io.loadmat('/data/people/sudderth/kalmanHDPHMM/speaker_diarization/gt_files/'+files[file])
        if X is None:
            X = np.transpose(data['u'])
            Z = data['zsub']
        else:
            X = np.vstack((X, np.transpose(data['u'])))
            Z = np.append(Z, data['zsub'])
        seqInds = np.append(seqInds, np.size(data['zsub']) + seqInds[file])

    return X, Z, seqInds

def get_data_info():
    return 'Multiple sequences of audio data from NIST proceedings'

def get_short_name():
    return 'SpeakDiar'


def get_data(**kwargs):
    X, trueZ, seqInds = get_XZ()
    Data = SeqXData(X = X, seqInds = seqInds, TrueZ = trueZ)
    Data.summary = get_data_info()
    return Data

