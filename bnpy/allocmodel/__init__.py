from AllocModel import AllocModel

from mix.MixModel import MixModel
from mix.DPMixModel import DPMixModel
from mix.DPMixPE import DPMixPE
from mix.DPMixFull import DPMixFull

from admix2.LDA import LDA
from admix2.HDPSB import HDPSB
from admix2.HDPPE import HDPPE
from admix2.HDPSBDir import HDPSBDir
from admix2.HDPDir import HDPDir
from admix2.HDPFastRhoFixed import HDPFastRhoFixed
from admix2.HDPFast import HDPFast

AllocModelConstructorsByName = { \
           'MixModel':MixModel,
           'DPMixModel':DPMixModel,
           'DPMixPE':DPMixPE,
           'LDA':LDA,
           'HDPSB':HDPSB,
           'HDPPE':HDPPE,
           'HDPDir':HDPDir,
           'HDPSBDir':HDPSBDir,
           'HDPFastRhoFixed':HDPFastRhoFixed,
           'HDPFast':HDPFast,
           'DPMixFull':DPMixFull,
          }

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = list()
for name in AllocModelConstructorsByName:
  __all__.append(name)
