''' 
MultinomialObsModel.py
'''
import numpy as np
import copy
from scipy.special import digamma, gammaln

from ObsModel import ObsModel


class MultinomialObsModel(ObsModel):

	######################################################### Constructors
  	#########################################################
  	def __init__(self, inferType, obsPrior=None):
  		self.inferType = inferType
  		self.obsPrior = obsPrior

  	@classmethod
  	def CreateWithPrior(cls, inferType, priorArgDict, Data):
  		raise NotImplementedError('TODO')

  	@classmethod
  	def CreateWithAllComps(cls):
  		raise NotImplementedError('TODO')

  	######################################################### Global Params
  	######################################################### M-Step
  	def update_obs_params_EM(self, SS, **kwargs):
  		
