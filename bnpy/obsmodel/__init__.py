'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
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
}

for key, val in ObsModelConstructorsByName.items():
	expandedkey = key + "ObsModel"
	ObsModelConstructorsByName[expandedkey] = val

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
