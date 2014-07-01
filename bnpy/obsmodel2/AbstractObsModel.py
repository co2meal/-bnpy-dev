class AbstractObsModel(object):

  def __init__(self):
    pass

  def calc_local_params(self, Data):
    pass

  def ClearCache(self):
    self.Cache.clear()

  def GetCached(self, key, k=None):
    ''' Retrieved cached function evaluation if possible,
          otherwise compute fresh result and store in cache for later.
    '''
    ckey = key + '-' + str(k)
    try:
      return self.Cache[ ckey ]
    except KeyError:
      Val = getattr(self, '_' + key)(k)
      self.Cache[ ckey ] = Val
      return Val