"""
The:mod:`learnalg' module provides standard learning algorithms such as EM and VB (Variational Bayes)
"""
from .LearnAlg import LearnAlg
from .VBLearnAlg import VBLearnAlg
from .StochasticOnlineVBLearnAlg import StochasticOnlineVBLearnAlg
from .MemoizedOnlineVBLearnAlg import MemoizedOnlineVBLearnAlg
import MergeMove
__all__ = ['LearnAlg', 'VBLearnAlg', 'StochasticOnlineVBLearnAlg', 'MemoizedOnlineVBLearnAlg', 'MergeMove']
