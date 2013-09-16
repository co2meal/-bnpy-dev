"""
The:mod:`learn'  module provides standard learning algorithms such as EM and VB (Variational Bayes)
"""
from .LearnAlg import LearnAlg
from .VBLearnAlg import VBLearnAlg
from .StochasticOnlineVBLearnAlg import StochasticOnlineVBLearnAlg
from .MemoizedOnlineVBLearnAlg import MemoizedOnlineVBLearnAlg

__all__ = ['LearnAlg', 'VBLearnAlg', 'StochasticOnlineVBLearnAlg', 'MemoizedOnlineVBLearnAlg']
