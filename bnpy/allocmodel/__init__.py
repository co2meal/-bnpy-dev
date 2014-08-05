from AllocModel import AllocModel

from mix.MixModel import MixModel
from mix.DPMixModel import DPMixModel
from mix.HardDPMixModel import HardDPMixModel

from admix.AdmixModel import AdmixModel
from admix.HDPModel import HDPModel
from admix.HDPModel2 import HDPModel2
from admix.HDPPE import HDPPE
from admix.HDPPE2 import HDPPE2
from admix.HDPSoft2Hard import HDPSoft2Hard
from admix.HDPHardMult import HDPHardMult
from admix.HDPRelModel import HDPRelAssortModel
from admix.HDPStickBreak import HDPStickBreak

from hmm.FiniteHMM import FiniteHMM
from hmm.HDPHMM import HDPHMM

AllocModelConstructorsByName = { \
           'MixModel':MixModel,
           'DPMixModel':DPMixModel,
           'HardDPMixModel':HardDPMixModel,
           'AdmixModel':AdmixModel,
           'HDPModel':HDPModel,
           'HDPModel2':HDPModel2,
           'HDPPE':HDPPE,
           'HDPPE2':HDPPE2,
           'HDPSoft2Hard':HDPSoft2Hard,
           'FiniteHMM':FiniteHMM,
           'HDPHardMult':HDPHardMult,
           'HDPRelAssortModel':HDPRelAssortModel,
           'HDPHMM':HDPHMM,
           'HDPStickBreak':HDPStickBreak
          }

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = list()
for name in AllocModelConstructorsByName:
  __all__.append(name)
