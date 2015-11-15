"""
The :mod:`util` module gathers utility functions
"""

import RandUtil

from .PrettyPrintUtil import np2flatstr, flatstr2np
from .PrettyPrintUtil import split_str_into_fixed_width_lines
from .MatMultUtil import dotATA, dotATB, dotABT
from .RandUtil import choice, multinomial
from .SpecialFuncUtil import MVgammaln, MVdigamma, digamma, gammaln
from .SpecialFuncUtil import LOGTWO, LOGPI, LOGTWOPI, EPS
from .SpecialFuncUtil import logsumexp
from .VerificationUtil import isEvenlyDivisibleFloat, assert_allclose
from .ShapeUtil import as1D, as2D, as3D, toCArray, argsort_bigtosmall_stable
from .ParallelUtil import numpyToSharedMemArray, sharedMemToNumpyArray
from .ParallelUtil import sharedMemDictToNumpy, fillSharedMemArray

__all__ = ['RandUtil',
           'split_str_into_fixed_width_lines',
           'np2flatstr', 'flatstr2np',
           'dotATA', 'dotATB', 'dotABT',
           'choice', 'multinomial',
           'MVgammaln', 'MVdigamma', 'logsumexp', 'digamma', 'gammaln',
           'isEvenlyDivisibleFloat', 'assert_allclose',
           'LOGTWO', 'LOGTWOPI', 'LOGPI', 'EPS',
           'as1D', 'as2D', 'as3D', 'toCArray', 'argsort_bigtosmall_stable',
           'numpyToSharedMemArray', 'sharedMemToNumpyArray',
           'sharedMemDictToNumpy', 'fillSharedMemArray',
           ]
