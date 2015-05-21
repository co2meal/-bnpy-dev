'''
PlotParamComparison.py

Usage (command-line)
-------
```
python -m bnpy.viz.PlotParamComparison dataName jobpattern \
    --pvar nu \
    --lvar initname \
    --xvar sF \
    --yvar hamming-distance \
```
Creates a 3 panel plot, where each subplot has one line for each initname,
and each line would trace out the pattern of xvar vs. hamming-distance

Steps
------
Acquire PPListMap given pattern like nickName-K=35-sF=123
Loop over panels
    Loop over lines
        Create x/y pairs for all data points in line
        Plot line

'''
from matplotlib import pylab
import numpy as np
import argparse
import glob
import os
import scipy.io
import copy

from bnpy.ioutil import BNPYArgParser
from JobFilter import makeListOfJPatternsWithSpecificVals
from JobFilter import makePPListMapFromJPattern

import matplotlib
matplotlib.rcParams['text.usetex'] = False

taskidsHelpMsg = "ids of trials/runs to plot from given job." + \
                 " Example: '4' or '1,2,3' or '2-6'."

DefaultLinePlotKwArgs = dict(
    markersize=10, 
    linewidth=1.75, 
    label=None,
    color='k',
    )

DefaultColorList = [
    (1, 0, 0),  # red
    (0, 0, 0),  # black
    (0, 0, 1),  # blue
    (0, 1, 0.25),  # green (darker)
    (1, 0, 1),  # magenta
    (0, 1, 1),  # cyan
    (1, 0.6, 0),  # orange
    ]

LabelMap = dict(laps='num pass thru data',
                iters='num alg steps',
                times='elapsed time (sec)',
                K='num topics K',
                evidence='train objective',
                )
LabelMap['laps-saved-params'] = 'num pass thru data'
LabelMap['hamming-distance'] = 'Hamming dist.'
LabelMap['Keff'] = 'num topics K'


def plotManyPanelsByPVar(jpathPattern='/tmp/', 
        pvar='nu', pvals=None,
        W=5, H=4,
        savefilename=None, doShowNow=False,
        **kwargs):
    ''' Create line plots for jobs matching pattern and provided kwargs
    '''
    prefixfilepath = os.path.sep.join(jpathPattern.split(os.path.sep)[:-1])
    PPListMap = makePPListMapFromJPattern(jpathPattern)
    if pvals is None:
        pvals = PPListMap[pvar]
    else:
        pvals = [p for p in pvals if p in PPListMap[pvar]]
    jpathList = makeListOfJPatternsWithSpecificVals(PPListMap, 
        prefixfilepath=prefixfilepath,
        key=pvar,
        vals=pvals,
        **kwargs)

    nrows = 1
    ncols = len(pvals)
    pylab.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*W, nrows*H))
    axH = None
    for panelID, panel_jobPattern in enumerate(jpathList):
        axH = pylab.subplot(nrows, ncols, panelID+1, sharey=axH, sharex=axH)
        # Only show legend on first plot
        if panelID > 0 and 'loc' in kwargs:
            kwargs['loc'] = None
        plotMultipleLinesByLVar(panel_jobPattern,
            **kwargs)
        pylab.title('%s=%s' % (pvar, pvals[panelID]))

    pylab.subplots_adjust(bottom=0.15)

    if savefilename is not None:
        try:
            pylab.show(block=False)
        except TypeError:
            pass  # when using IPython notebook
        pylab.savefig(savefilename, bbox_inches='tight', pad_inches=0)
    elif doShowNow:
        try:
            pylab.show(block=True)
        except TypeError:
            pass  # when using IPython notebook
    

def plotMultipleLinesByLVar(jpathPattern,
        lvar='initname', lvals=None,
        ColorMap=DefaultColorList,
        loc=None, bbox_to_anchor=None,
        savefilename=None, tickfontsize=None, doShowNow=False,
        **kwargs):
    ''' Create line plots for provided jobs.
    '''
    prefixfilepath = os.path.sep.join(jpathPattern.split(os.path.sep)[:-1])
    PPListMap = makePPListMapFromJPattern(jpathPattern)
    if lvals is None:
        lvals = PPListMap[lvar]
    else:
        lvals = [ll for ll in lvals if ll in PPListMap[lvar]]
    jpathList = makeListOfJPatternsWithSpecificVals(PPListMap, 
        prefixfilepath=prefixfilepath,
        key=lvar,
        vals=lvals,
        **kwargs)

    for lineID, line_jobPattern in enumerate(jpathList):
        line_label = '%s=%s' % (lvar, lvals[lineID])
        if isinstance(ColorMap, dict):
            for label in [line_label, line_jobPattern]:
                try:
                    line_color = ColorMap[label]
                except KeyError:
                    line_color = DefaultColorList[lineID]
        else:
            # Access next elt in ColorMap list
            line_color = ColorMap[lineID]
        plotSingleLineAcrossJobsByXVar(line_jobPattern,
            label=line_label,
            color=line_color,
            lineID=lineID,
            **kwargs)

    if loc is not None and len(jpathList) > 1:
        pylab.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    if tickfontsize is not None:
        pylab.tick_params(axis='both', which='major', labelsize=tickfontsize)
    
    if savefilename is not None:
        try:
            pylab.show(block=False)
        except TypeError:
            pass  # when using IPython notebook
        pylab.savefig(savefilename, bbox_inches='tight', pad_inches=0)
    elif doShowNow:
        try:
            pylab.show(block=True)
        except TypeError:
            pass  # when using IPython notebook


def plotSingleLineAcrossJobsByXVar(jpathPattern, 
        label='',
        xvar='laps',
        xvals=None,
        xlabel=None,
        yvar='evidence',
        lineStyle='.-',
        taskid='.best',
        lineID=0,
        **kwargs):
    ''' Create line plot in current figure for job matching the pattern

    Iterates over each xval in provided list of values.
    Each one corresponds to a single saved job.

    Post Condition
    --------------
    Current axes have one line added.
    '''   
    prefixfilepath = os.path.sep.join(jpathPattern.split(os.path.sep)[:-1])
    PPListMap = makePPListMapFromJPattern(jpathPattern)
    if xvals is None:
        xvals = PPListMap[xvar]
    
    xs = np.zeros(len(xvals))
    ys = np.zeros(len(xvals))
    jpathList = makeListOfJPatternsWithSpecificVals(PPListMap, 
        prefixfilepath=prefixfilepath,
        key=xvar,
        vals=xvals,
        **kwargs)
    for i, jobpath in enumerate(jpathList):
        if not os.path.exists(jobpath):
            raise ValueError("PATH NOT FOUND: %s" % (jobpath))
        x = float(xvals[i])
        y = loadYValFromDisk(jobpath, taskid, yvar=yvar)
        assert isinstance(x, float)
        assert isinstance(y, float)
        xs[i] = x
        ys[i] = y

    plotargs = copy.deepcopy(DefaultLinePlotKwArgs)
    for key in plotargs:
        if key in kwargs:
            plotargs[key] = kwargs[key]
    plotargs['markeredgecolor'] = plotargs['color']
    plotargs['label'] = label
    pylab.plot(xs, ys, lineStyle, **plotargs)

    if lineID == 0:
        if xlabel is None:
            xlabel = xvar
        pylab.xlabel(xlabel)
        pylab.ylabel(LabelMap[yvar])

def loadYValFromDisk(jobpath, taskid, yvar='evidence'):
    ytxtfile = os.path.join(jobpath, taskid, yvar + '.txt')
    if not os.path.isfile(ytxtfile):
        ytxtfile = os.path.join(
            jobpath, taskid, yvar + '-saved-params.txt')
    ys = np.loadtxt(ytxtfile)
    return ys[-1]


def parse_args(xvar='sF', 
        yvar='evidence',
        lvar='initname',
        pvar='nu',
        **kwargs):
    ''' Returns Namespace of parsed arguments retrieved from command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName', type=str, default='AsteriskK8')
    parser.add_argument('jpathPattern', type=str, default='demo*')
    parser.add_argument('--xvar', type=str, default=xvar,
        help="name of x axis variable to plot.")
    parser.add_argument('--yvar', type=str, default=yvar,
        choices=LabelMap.keys(),
        help="name of y axis variable to plot.")
    parser.add_argument('--lvar', type=str, default=lvar,
        help="quantity that varies across lines")
    parser.add_argument('--pvar', type=str, default=None,
        help="quantity that varies across subplots")
    parser.add_argument('--taskid', type=str, default='1',
        help="specify which task to plot (.best, .worst, etc)")
    parser.add_argument(
        '--savefilename', type=str, default=None,
        help="location where to save figure (absolute path directory)")
    args, unkList = parser.parse_known_args()
    argDict = BNPYArgParser.arglist_to_kwargs(unkList)
    argDict.update(args.__dict__)
    argDict.update(kwargs)
    argDict['jpathPattern'] = os.path.join(os.environ['BNPYOUTDIR'],
                                           args.dataName,
                                           args.jpathPattern)
    del argDict['dataName']
    for key in argDict:
        if key.endswith('vals'):
            if not isinstance(argDict[key], list):
                argDict[key] = argDict[key].split(',')
    return argDict

if __name__ == "__main__":
    argDict = parse_args(xvar='sF', yvar='evidence')
    plotManyPanelsByPVar(doShowNow=1, **argDict)
