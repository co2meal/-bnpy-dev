"""
The:mod:`learn'  modulerovides standard learning algorithms
  such as EM and Variational Bayesian
"""
from .LearnAlg import LearnAlg

from .VBLearnAlg import VBLearnAlg
from .OnlineVBLearnAlg import OnlineVBLearnAlg

from .VBSMLearnAlg import VBSMLearnAlg
from .OnlineVBSMLearnAlg import OnlineVBSMLearnAlg

from .IncrementalVBLearnAlg import IncrementalVBLearnAlg
from .iVBLearnAlg import iVBLearnAlg
from .iVBSMLearnAlg import iVBSMLearnAlg

from .GibbsSamplerAlg import GibbsSamplerAlg

from .VBInferHeldout import VBInferHeldout
from .VBInferHeldoutEvBound import VBInferHeldoutEvBound

import SplitMove, DeleteMove, MergeMove, BirthMove, CombineMove, FastDeathMove, FastBirthMove

__all__ = ['LearnAlg', 'iVBLearnAlg', 'iVBSMLearnAlg', 'VBLearnAlg', 'VBInferHeldout', 'VBInferHeldoutEvBound', 'VBSMLearnAlg', 'OnlineVBLearnAlg', 'OnlineVBSMLearnAlg', 'GibbsSamplerAlg', 'SplitMove', 'DeleteMove','MergeMove', 'BirthMove', 'CombineMove', 'FastDeathMove','FastBirthMove']
