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

import learnalg
import Run
from Run import run

########################################################### Config data
###########################################################  location
import os
isValid = False
if 'BNPYDATADIR' in os.environ and os.path.exists(os.environ['BNPYDATADIR']):
  isValid = True
if not isValid:
  root = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
  os.environ['BNPYDATADIR'] = os.path.join(root, 'demodata/')

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

__all__ = ['run', 'Run', 'learn', 'allocmodel','obsmodel', 'suffstats',
           'HModel', 'init', 'util','ioutil','viz','distr']
