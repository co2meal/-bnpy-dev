from AllocModel import AllocModel

from mix.MixModel import MixModel
from mix.DPMixModel import DPMixModel
from mix.DPMixPE import DPMixPE

from admix2.LDA import LDA
from admix2.HDPSB import HDPSB
from admix2.HDPPE import HDPPE

AllocModelConstructorsByName = { \
           'MixModel':MixModel,
           'DPMixModel':DPMixModel,
           'DPMixPE':DPMixPE,
           'LDA':LDA,
           'HDPSB':HDPSB,
           'HDPPE':HDPPE,
          }

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = list()
for name in AllocModelConstructorsByName:
  __all__.append(name)
