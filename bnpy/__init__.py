''' bnpy module __init__ file
'''

import data
import distr
import util
import suffstats

import allocmodel
import obsmodel
from HModel import HModel

import ioutil
import init

import learn
import Run

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
if canPlot:
  import viz

__all__ = ['Run', 'learn', 'allocmodel','obsmodel', 'suffstats',
           'HModel', 'init', 'util','ioutil','viz','distr']
