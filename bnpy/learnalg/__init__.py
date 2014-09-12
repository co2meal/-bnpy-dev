"""
The:mod:`learnalg' module provides standard learning algorithms such as EM and VB (Variational Bayes)
"""

from LearnAlg import LearnAlg
from VBAlg import VBAlg
from MOVBAlg import MOVBAlg
from SOVBAlg import SOVBAlg
from EMAlg import EMAlg

from GSAlg import GSAlg

__all__ = ['LearnAlg', 'VBAlg', 'MOVBAlg',
           'SOVBAlg', 'EMAlg', 
           'GSAlg']
