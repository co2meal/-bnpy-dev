from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from MixGaussObsModel import MixGaussObsModel
from ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from AutoRegGaussObsModel import AutoRegGaussObsModel
from MultObsModel import MultObsModel
from BernObsModel import BernObsModel

ObsModelConstructorsByName = {
    'DiagGauss': DiagGaussObsModel,
    'Gauss': GaussObsModel,
    'ZeroMeanGauss': ZeroMeanGaussObsModel,
    'AutoRegGauss': AutoRegGaussObsModel,
    'Mult': MultObsModel,
    'Bern': BernObsModel,
    'MixGauss' : MixGaussObsModel
}

# Make constructor accessible by nickname and fullname
# Nickname = 'Gauss'
# Fullname = 'GaussObsModel'
for val in ObsModelConstructorsByName.values():
    fullname = str(val.__name__)
    ObsModelConstructorsByName[fullname] = val

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
