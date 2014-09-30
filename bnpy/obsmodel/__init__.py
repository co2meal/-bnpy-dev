'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from AutoRegGaussObsModel import AutoRegGaussObsModel
from MultObsModel import MultObsModel

ObsModelConstructorsByName = { \
           'DiagGauss':DiagGaussObsModel,
           'Gauss':GaussObsModel,
           'ZeroMeanGauss':ZeroMeanGaussObsModel,
           'AutoRegGauss':AutoRegGaussObsModel,
           'Mult':MultObsModel,           
          }

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
