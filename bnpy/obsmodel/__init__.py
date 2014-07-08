'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from MultObsModel import MultObsModel

ObsModelConstructorsByName = { \
           'DiagGauss':DiagGaussObsModel,
           'Gauss':GaussObsModel,
           'ZeroMeanGauss':ZeroMeanGaussObsModel,
           'Mult':MultObsModel,
#           'BernRel':BernRelObsModel,
          }

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
