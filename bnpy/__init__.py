''' bnpy module __init__ file
'''
import os
import sys

import data
import suffstats
import util

import allocmodel
import obsmodel
from HModel import HModel

import ioutil
load_model = ioutil.ModelReader.load_model
save_model = ioutil.ModelWriter.save_model

import init
import learnalg
import birthmove
import mergemove
#import deletemove

import Run
from Run import run

########################################################### Configure save
###########################################################  location
hasWriteableOutdir = False
if 'BNPYOUTDIR' in os.environ:
  outdir = os.environ['BNPYOUTDIR']
  if os.path.exists(outdir):
    try:
      with open(os.path.join(outdir, 'bnpytest'), 'w') as f:
        pass
    except IOError:
      sys.exit('BNPYOUTDIR not writeable: %s' % (outdir))
    hasWriteableOutdir = True
if not hasWriteableOutdir:
  raise ValueError('Environment variable BNPYOUTDIR not specified. Cannot save results to disk')

########################################################### Configure data
###########################################################  location
root = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
sys.path.append(os.path.join(root, 'datasets/'))
if 'BNPYDATADIR' in os.environ:
  if os.path.exists(os.environ['BNPYDATADIR']):
    sys.path.append(os.environ['BNPYDATADIR'])
  else:
    print "Warning: Environment variable BNPYDATADIR not a valid directory"

########################################################### Optional: viz
###########################################################  package for plots
canPlot = False

try:
  from matplotlib import pylab
  canPlot = True
except ImportError:
  print "Error importing matplotlib. Plotting disabled."
  print "Fix by making sure this produces a figure window on your system"
  print " >>> from matplotlib import pylab; pylab.figure(); pylab.show();"

__all__ = ['run', 'Run', 'learn', 'allocmodel','obsmodel', 'suffstats',
           'HModel', 'init', 'util', 'ioutil', 'mergeutil']

if canPlot:
  import viz
  __all__.append('viz')
