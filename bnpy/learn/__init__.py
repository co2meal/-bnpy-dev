"""
The:mod:`learn'  module provides standard learning algorithms such as EM and VB (Variational Bayes)
"""
from .LearnAlg import LearnAlg
from .VBLearnAlg import VBLearnAlg

__all__ = ['LearnAlg', 'VBLearnAlg']
