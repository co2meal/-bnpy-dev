'''
PlotTrace.py

Executable for plotting trace stats of learning algorithm progress, including
* objective function (ELBO) vs laps thru data
* number of active components vs laps thru data
* hamming distance vs laps thru data

Usage (command-line)
-------
python -m bnpy.viz.PlotTrace dataName jobpattern [kwargs]
'''
import numpy as np
import argparse
import glob
import os
import scipy.io

from PlotUtil import pylab
from bnpy.ioutil import BNPYArgParser
from JobFilter import filterJobs

taskidsHelpMsg = "ids of trials/runs to plot from given job." + \
                 " Example: '4' or '1,2,3' or '2-6'."

Colors = [(0, 0, 0),  # black
          (0, 0, 1),  # blue
          (1, 0, 0),  # red
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


def plotJobsThatMatchKeywords(jpathPattern='/tmp/', **kwargs):
    ''' Create line plots for jobs matching pattern and provided kwargs
    '''
    if not jpathPattern.startswith(os.path.sep):
        jpathPattern = os.path.join(os.environ['BNPYOUTDIR'], jpathPattern)
    jpaths, legNames = filterJobs(jpathPattern, **kwargs)
    plotJobs(jpaths, legNames, **kwargs)


def plotJobs(jpaths, legNames, styles=None, density=2,
             xvar='laps', yvar='evidence', loc='upper right',
             xmin=None, xmax=None,
             taskids=None, savefilename=None, tickfontsize=None,
             bbox_to_anchor=None, **kwargs):
    ''' Create line plots for provided jobs.
    '''
    nLines = len(jpaths)
    if nLines == 0:
        raise ValueError('Empty job list. Nothing to plot.')

    nLeg = len(legNames)

    for lineID in xrange(nLines):
        if styles is None:
            curStyle = dict(colorID=lineID)
        else:
            curStyle = styles[lineID]

        plot_all_tasks_for_job(jpaths[lineID], legNames[lineID],
                               xvar=xvar, yvar=yvar,
                               taskids=taskids, density=density, **curStyle)
    
    # Y-axis limit determination
    # If we have "enough" data about the run beyond two full passes of dataset,
    # we zoom in on the region of data beyond lap 2
    if xvar == 'laps' and yvar == 'evidence':
        xmax = 0
        ymin = np.inf
        ymin2 = np.inf
        ymax = -np.inf
        allRunsHaveXBeyond1 = True
        for line in pylab.gca().get_lines():
            xd = line.get_xdata()
            yd = line.get_ydata()
            if xd.size < 3:
                allRunsHaveXBeyond1 = False
                continue
            posLap1 = np.searchsorted(xd, 1.0)
            posLap2 = np.searchsorted(xd, 2.0)
            if posLap1 < xd.size:
                ymin = np.minimum(ymin, yd[posLap1])
                ymax = np.maximum(ymax, yd[posLap1:].max())
            if posLap2 < xd.size:
                ymin2 = np.minimum(ymin2, yd[posLap2])
            xmax = np.maximum(xmax, xd.max())
            if xd.max() <= 1:
                allRunsHaveXBeyond1 = False
        if allRunsHaveXBeyond1 and xmax > 1.5:
            # If all relevant curves extend beyond x=1, only show that part
            xmin = 1.0 - 1e-5
        else:
            xmin = 0
        if allRunsHaveXBeyond1 and ymin2 < ymax:
            range1 = ymax - ymin
            range2 = ymax - ymin2
            if 10 * range2 < range1:
                # Y values jump from lap1 to lap2 is enormous,
                # so let's just show y values from lap2 onward...
                ymin = ymin2
        if (not np.allclose(ymax, ymin)) and allRunsHaveXBeyond1:
            pylab.ylim([ymin, ymax + 0.1 * (ymax - ymin)])
        pylab.xlim([xmin, xmax + .05 * (xmax - xmin)])
    
    if loc is not None and len(jpaths) > 1:
        pylab.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    if tickfontsize is not None:
        pylab.tick_params(axis='both', which='major', labelsize=tickfontsize)

    if savefilename is not None:
        try:
            pylab.show(block=False)
        except TypeError:
            pass  # when using IPython notebook
        pylab.savefig(savefilename, bbox_inches='tight', pad_inches=0)
    else:
        try:
            pylab.show(block=True)
        except TypeError:
            pass  # when using IPython notebook


def plot_all_tasks_for_job(jobpath, label, taskids=None,
                           lineType='.-',
                           color=None,
                           colorID=0,
                           density=2,
                           yvar='evidence',
                           markersize=10,
                           linewidth=2,
                           xvar='laps', **kwargs):
    ''' Create line plot in current figure for each task/run of jobpath
    '''
    if not os.path.exists(jobpath):
        if not jobpath.startswith(os.path.sep):
            jobpath_tmp = os.path.join(os.environ['BNPYOUTDIR'], jobpath)
            if not os.path.exists(jobpath_tmp):
                raise ValueError("PATH NOT FOUND: %s" % (jobpath))
            jobpath = jobpath_tmp
    if color is None:
        color = Colors[colorID % len(Colors)]
    taskids = BNPYArgParser.parse_task_ids(jobpath, taskids)

    if yvar == 'hamming-distance':
        if xvar == 'laps':
            xvar = 'laps-saved-params'

    for tt, taskid in enumerate(taskids):
        try:
            var_ext = ''
            ytxtfile = os.path.join(jobpath, taskid, yvar + '.txt')
            if not os.path.isfile(ytxtfile):
                var_ext = '-saved-params'
                ytxtfile = os.path.join(
                    jobpath, taskid, yvar + var_ext + '.txt')
            ys = np.loadtxt(ytxtfile)

            xtxtfile = os.path.join(jobpath, taskid, xvar + var_ext + '.txt')
            xs = np.loadtxt(xtxtfile)

        except IOError as e:
            try:
                xs, ys = loadXYFromTopicModelFiles(jobpath, taskid)
            except ValueError:
                try:
                    xs, ys = loadXYFromTopicModelSummaryFiles(jobpath, taskid)
                except ValueError:
                    raise e

        if yvar == 'hamming-distance' or yvar == 'Keff':
            if xvar == 'laps-saved-params':
                # fix off-by-one error, if we save an extra dist on final lap
                if xs.size == ys.size - 1:
                    ys = ys[:-1]
                elif ys.size == xs.size - 1:
                    xs = xs[:-1]  # fix off-by-one error, if we quit early
            elif xs.size != ys.size:
                # Try to subsample both time series at laps where they
                # intersect
                laps_x = np.loadtxt(os.path.join(jobpath, taskid, 'laps.txt'))
                laps_y = np.loadtxt(os.path.join(jobpath, taskid,
                                                 'laps-saved-params.txt'))
                assert xs.size == laps_x.size
                if ys.size == laps_y.size - 1:
                    laps_y = laps_y[:-1]
                xs = xs[np.in1d(laps_x, laps_y)]
                ys = ys[np.in1d(laps_y, laps_x)]

        if xs.size != ys.size:
            raise ValueError('Dimension mismatch. len(xs)=%d, len(ys)=%d'
                             % (xs.size, ys.size))

        if xs.size < 3:
            continue

        # Cleanup laps data. Verify that it is sorted, with no collisions.
        if xvar == 'laps':
            diff = xs[1:] - xs[:-1]
            goodIDs = np.flatnonzero(diff >= 0)
            if len(goodIDs) < xs.size - 1:
                print 'WARNING: looks like multiple runs writing to this file!'
                print jobpath
                print 'Task: ', taskid
                print len(goodIDs), xs.size - 1
                xs = np.hstack([xs[goodIDs], xs[-1]])
                ys = np.hstack([ys[goodIDs], ys[-1]])

        if xvar == 'laps' and yvar == 'evidence':
            mask = xs >= 1.0
            xs = xs[mask]
            ys = ys[mask]

        # Force plot density (data points per lap) to desired specification
        # This avoids making plots that have huge file sizes,
        # due to too much content in the given display space
        if xvar == 'laps' and xs.size > 20 and np.sum(xs > 5) > 10:
            if (xs[-1] - xs[9]) != 0:
                curDensity = (xs.size - 10) / (xs[-1] - xs[9])
            else:
                curDensity = density
            while curDensity > density and xs.size > 11:
                # Thin xs and ys data by a factor of 2
                # while preserving the first 10 data points
                xs = np.hstack([xs[:10], xs[10::2]])
                ys = np.hstack([ys[:10], ys[10::2]])
                curDensity = (xs.size - 10) / (xs[-1] - xs[9])

        plotargs = dict(markersize=markersize, linewidth=linewidth, label=None,
                        color=color, markeredgecolor=color)
        plotargs.update(kwargs)
        if tt == 0:
            plotargs['label'] = label

        pylab.plot(xs, ys, lineType, **plotargs)

    pylab.xlabel(LabelMap[xvar])
    pylab.ylabel(LabelMap[yvar])


def loadXYFromTopicModelSummaryFiles(jobpath, taskid, xvar='laps', yvar='K'):
    ''' Load x and y variables for line plots from TopicModel files
    '''
    ypath = os.path.join(jobpath, taskid, 'predlik-' + yvar + '.txt')
    if not os.path.exists(ypath):
        raise ValueError('No TopicModel summary text files found')
    lappath = os.path.join(jobpath, taskid, 'predlik-lapTrain.txt')
    xs = np.loadtxt(lappath)
    ys = np.loadtxt(ypath)
    return xs, ys


def loadXYFromTopicModelFiles(jobpath, taskid, xvar='laps', yvar='K'):
    ''' Load x and y variables for line plots from TopicModel files
    '''
    tmpathList = glob.glob(os.path.join(jobpath, taskid, 'Lap*TopicModel.mat'))
    if len(tmpathList) < 1:
        raise ValueError('No TopicModel.mat files found')
    tmpathList.sort()  # ascending, from lap 0 to lap 1 to lap 100 to ...
    basenames = [x.split(os.path.sep)[-1] for x in tmpathList]
    laps = np.asarray([float(x[3:11]) for x in basenames])
    Ks = np.zeros_like(laps)
    for tt, tmpath in enumerate(tmpathList):
        if yvar == 'K':
            Q = scipy.io.loadmat(tmpath, variable_names=['K', 'probs'])
            try:
                Ks[tt] = Q['K']
            except KeyError:
                Ks[tt] = Q['probs'].size
        else:
            raise ValueError('Unknown yvar type for topic model: ' + yvar)
    return laps, Ks


def parse_args(xvar='laps', yvar='evidence'):
    ''' Returns Namespace of parsed arguments retrieved from command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataName', type=str, default='AsteriskK8')
    parser.add_argument('jpath', type=str, default='demo*')

    parser.add_argument('--xvar', type=str, default=xvar,
                        choices=LabelMap.keys(),
                        help="name of x axis variable to plot.")

    parser.add_argument('--yvar', type=str, default=yvar,
                        choices=LabelMap.keys(),
                        help="name of y axis variable to plot.")

    helpMsg = "ids of trials/runs to plot from given job." + \
              " Example: '4' or '1,2,3' or '2-6'."
    parser.add_argument(
        '--taskids', type=str, default=None, help=helpMsg)
    parser.add_argument(
        '--savefilename', type=str, default=None,
        help="location where to save figure (absolute path directory)")

    args, unkList = parser.parse_known_args()

    argDict = BNPYArgParser.arglist_to_kwargs(unkList, doConvertFromStr=False)
    argDict.update(args.__dict__)
    argDict['jpathPattern'] = os.path.join(os.environ['BNPYOUTDIR'],
                                           args.dataName,
                                           args.jpath)
    del argDict['dataName']
    del argDict['jpath']
    return argDict

plotJobsThatMatch = plotJobsThatMatchKeywords

if __name__ == "__main__":
    argDict = parse_args('laps', 'evidence')
    plotJobsThatMatchKeywords(**argDict)
