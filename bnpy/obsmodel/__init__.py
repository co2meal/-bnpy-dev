'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from MultObsModel import MultObsModel

ObsModelConstructorsByName = { \
           'DiagGauss':DiagGaussObsModel,
           'Gauss':GaussObsModel,
#           'ZMGauss':ZMGaussObsModel,
           'Mult':MultObsModel,
#           'BernRel':BernRelObsModel,
          }

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
