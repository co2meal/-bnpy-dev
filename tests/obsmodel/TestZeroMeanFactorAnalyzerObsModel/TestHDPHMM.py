import numpy as np
import bnpy
from bnpy.data import XData
from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.obsmodel import ZeroMeanFactorAnalyzerObsModel
from numpy.linalg import inv, solve, det, slogdet, eig, LinAlgError
from scipy.linalg import eigh
from scipy.special import psi, gammaln
from bnpy.util import dotATA
from bnpy.util import LOGTWOPI, EPS


if __name__ == '__main__':
    hmodel, RInfo = bnpy.run('DDToyHMM', 'HDPHMM', 'Gauss', 'moVB',
                             creationProposalName='bisectGrownBlocks', K=1, nLap=5000, moves='seqcreate,merge,delete,shuffle')