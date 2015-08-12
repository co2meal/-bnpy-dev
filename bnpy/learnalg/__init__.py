"""
The:mod:`learnalg' module provides learning algorithms.
"""

from LearnAlg import LearnAlg
from VBAlg import VBAlg
from MOVBAlg import MOVBAlg
from SOVBAlg import SOVBAlg
from EMAlg import EMAlg

from ParallelVBAlg import ParallelVBAlg
from ParallelMOVBAlg import ParallelMOVBAlg

# from MOVBBirthMergeAlg import MOVBBirthMergeAlg
# from ParallelMOVBMovesAlg import ParallelMOVBMovesAlg
from MemoVBMovesAlg import MemoVBMovesAlg

from GSAlg import GSAlg

from SharedMemWorker import SharedMemWorker

__all__ = ['LearnAlg', 'VBAlg', 'MOVBAlg',
           'SOVBAlg', 'EMAlg',
           'ParallelVBAlg', 'ParallelMOVBAlg',
           'GSAlg', 'SharedMemWorker', 'MemoVBMovesAlg']
