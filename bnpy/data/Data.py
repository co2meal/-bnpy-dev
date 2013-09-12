'''
Data.py

General template for all data objects.
'''

class Data(object):
  
  def __init__(self, *args, **kwargs):
    self.nObs = 0
    self.nObsTotal=0
    
  def add_obs(self, *args, **kwargs):
    pass
    
  def remove_obs(self, *args, **kwargs):
    pass
    
  def select_obs_by_mask(self, *args, **kwargs):
    pass