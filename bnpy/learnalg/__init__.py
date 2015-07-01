"""
The:mod:`learnalg' module provides standard learning algorithms such as EM and VB (Variational Bayes)
"""

from LearnAlg import LearnAlg
from VBAlg import VBAlg
from MOVBAlg import MOVBAlg
from STRVBAlg import STRVBAlg
from MOVBBirthMergeAlg import MOVBBirthMergeAlg

from SOVBAlg import SOVBAlg
from EMAlg import EMAlg

from GSAlg import GSAlg

__all__ = ['LearnAlg', 'VBAlg', 'MOVBAlg', 'MOVBBirthMergeAlg',
           'SOVBAlg', 'EMAlg', 
           'GSAlg', 'STRVBAlg']
