"""
"""

import data
import distr
import util
import suffstats

import allocmodel
import obsmodel

import HModel
HModel = HModel.HModel

import ioutil
import init

import learn

canPlot = False
try:
  from matplotlib import pylab
  canPlot = True
except ImportError:
  print "Error importing matplotlib. Plotting disabled."
  print "Fix by making sure 'from matplotlib import pylab; pylab.figure(); pylab.show();' produces a figure on your system."

if canPlot:
  import viz

__all__ = ['learn', 'allocmodel','obsmodel', 'suffstats',
           'HModel', 'init', 'util','ioutil','viz','distr']
