from AllocModel import AllocModel

from mix.FiniteMixtureModel import FiniteMixtureModel
from mix.DPMixtureModel import DPMixtureModel

from topics.FiniteTopicModel import FiniteTopicModel
from topics.HDPTopicModel import HDPTopicModel

from hmm.FiniteHMM import FiniteHMM
from hmm.HDPHMM import HDPHMM

from relational.FiniteSMSB import FiniteSMSB
from relational.FiniteMMSB import FiniteMMSB
from relational.FiniteaMMSB import FiniteaMMSB
from relational.HDPaMMSB import HDPaMMSB


AllocModelConstructorsByName = {
    'FiniteMixtureModel': FiniteMixtureModel,
    'DPMixtureModel': DPMixtureModel,
    'FiniteTopicModel': FiniteTopicModel,
    'HDPTopicModel': HDPTopicModel,
    'FiniteHMM': FiniteHMM,
    'HDPHMM': HDPHMM,
    'FiniteSMSB': FiniteSMSB,
    'FiniteMMSB': FiniteMMSB,
    'FiniteaMMSB': FiniteaMMSB,
    'HDPaMMSB': HDPaMMSB,
}

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = list()
for name in AllocModelConstructorsByName:
    __all__.append(name)
