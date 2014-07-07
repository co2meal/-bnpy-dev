import numpy as np

def as1D(x):
  if type(x) != np.ndarray:
    x = np.asarray(x)
  if x.ndim < 1:
    x = np.asarray([x])
  return x

def as2D(x):
  if type(x) != np.ndarray:
    x = np.asarray(x)
  while x.ndim < 2:
    x = np.asarray([x])
  return x

def as3D(x):
  if type(x) != np.ndarray:
    x = np.asarray(x)
  while x.ndim < 3:
    x = np.asarray([x])
  return x

