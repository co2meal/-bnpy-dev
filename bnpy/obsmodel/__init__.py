'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from ZMGaussObsModel import ZMGaussObsModel
from MultObsModel import MultObsModel

ObsModelConstructorsByName = { \
           'DiagGauss':DiagGaussObsModel,
           'Gauss':GaussObsModel,
           'ZMGauss':ZMGaussObsModel,
           'Mult':MultObsModel,
          }

ObsModelNameSet = set(ObsModelConstructorsByName.keys())