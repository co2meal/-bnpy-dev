'''
'''

from DiagGaussObsModel import DiagGaussObsModel
from GaussObsModel import GaussObsModel
from ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from AutoRegGaussObsModel import AutoRegGaussObsModel
from MultObsModel import MultObsModel
from BernObsModel import BernObsModel
from ZeroMeanFactorAnalyzerObsModel import ZeroMeanFactorAnalyzerObsModel

ObsModelConstructorsByName = { \
           'DiagGauss':DiagGaussObsModel,
           'Gauss':GaussObsModel,
           'ZeroMeanGauss':ZeroMeanGaussObsModel,
           'AutoRegGauss':AutoRegGaussObsModel,
           'Mult':MultObsModel,           
           'Bern':BernObsModel,
           'ZeroMeanFactorAnalyzer':ZeroMeanFactorAnalyzerObsModel,
          }

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
