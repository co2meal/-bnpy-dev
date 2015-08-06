import numpy as np
import os
import sys
import glob

from PlotUtil import pylab

CELL_WIDTH = 200
SQIMG_WIDTH = 200
WIDEIMG_WIDTH = 600
htmlstart = """
    <html>
    <style>
    td.comment {
        border: 0px;
        width: %dpx;
        text-align: center;
        padding-bottom: 10px;
        padding-left: 10px;
    }
    td.png {
        border: 0px;
        text-align: left;
        padding-bottom: 10px;
        padding-left: 10px;
    }
    tr {
        outline: thin solid black;
    }
    </style>
    <body>
    <div align=center>
    """ % (CELL_WIDTH)

htmlend = """
    </div>
    </body>
    </html>
    """

def plotELBOtermsForProposal(
        curLdict, propLdictList,
        xs=None,
        ymin=-0.5,
        ymax=0.5,
        savefilename=None,
        legendKeys=['Ldata', 'Lentropy', 'Ltotal', 'Lalloc', 'LcDtheta'],
        **kwargs):
    ''' Create trace plot of ELBO gain/loss relative to current model.
    '''
    pylab.figure()
    L = len(propLdictList)
    if xs is None:
        xs = np.arange(1, L+1)
    for key in legendKeys:
        if key.count('total'):
            linewidth= 4
            alpha = 1
            style = '-'
        else:
            linewidth = 3
            alpha = 0.5
            style = '--'
        ys = np.asarray([propLdictList[i][key] for i in range(L)])
        ys -= curLdict[key]
        pylab.plot(xs, 100*ys, style,
                   color=_getLineColorFromELBOKey(key),
                   linewidth=linewidth,
                   alpha=alpha,
                   label=key)
    L = L + 1
    xlims = np.asarray([-0.75*L, L-0.5])
    pylab.xlim(xlims)
    pylab.xticks(np.arange(1, L))
    pylab.plot(xlims, np.zeros_like(xlims), 'k:')
    pylab.xlabel('num proposal steps')
    pylab.ylabel('L gain (proposal - current)')
    pylab.legend(loc='lower left', fontsize=12)
    pylab.subplots_adjust(left=0.2)
    if savefilename is not None:
        pylab.savefig(savefilename, pad_inches=0, bbox_inches='tight')
        pylab.close('all')
        print 'Wrote: %s' % (savefilename)


def plotDocUsageForProposal(docUsageByUID, savefilename=None, **kwargs):
    ''' Make trace plot of doc usage for each component.
    '''
    pylab.figure()
    L = 0
    maxVal = 0
    for k, uid in enumerate(docUsageByUID):
        ys = np.asarray(docUsageByUID[uid])
        xs = np.arange(0, ys.size)
        if k < 6: # only a few labels fit well on a legend
            pylab.plot(xs, ys, label=uid)
        else:
            pylab.plot(xs, ys)
        L = np.maximum(L, ys.size)
        maxVal = np.maximum(maxVal, ys.max())
    # Use big chunk of left-hand side of plot for legend display
    xlims = np.asarray([-0.75*L, L-0.5])
    pylab.xlim(xlims)
    pylab.xticks(np.arange(1, L))
    pylab.ylim([0, 1.1*maxVal])
    pylab.xlabel('num proposal steps')
    pylab.ylabel('num docs using each comp')
    pylab.legend(loc='upper left', fontsize=12)
    pylab.subplots_adjust(left=0.2)
    if savefilename is not None:
        pylab.savefig(savefilename, pad_inches=0, bbox_inches='tight')
        pylab.close('all')
        print 'Wrote: %s' % (savefilename)



def makeSingleProposalHTMLStr(DebugInfo, b_debugOutputDir='', **kwargs):
    ''' Create string representing complete HTML page for one proposal.

    Returns
    -------
    s : string
        hold plain-text HTML content
    '''
    htmlstr = htmlstart
    htmlstr += "<table>"
    # Row 1: original comps    
    htmlstr += "<tr>"
    htmlstr += '<td class="comment">Original model. Before proposal.</td>'
    htmlstr += '<td class="png">%s</td>' % (
        makeImgTag("OrigComps.png"))
    htmlstr += "</tr>\n"
    # Row : status report
    htmlstr += "<tr>"
    htmlstr += '<td class="comment">Proposal outcome:</td>'
    htmlstr += '<td>%s</td>' % (DebugInfo['status'])
    htmlstr += "</tr>\n"
    
    # Row 2: ELBO trace
    htmlstr += "<tr>"
    htmlstr += '<td class="comment">ELBO terms at each refinement step.</td>'
    htmlstr += '<td class="png">%s</td>' % (
        makeImgTag("ProposalTrace_ELBO.png"))
    htmlstr += "</tr>\n"
    # Row 3: Doc usage trace
    htmlstr += "<tr>"
    htmlstr += '<td class="comment">Number of documents used by each' + \
               " new comp at each refinement step</td>"
    htmlstr += '<td class="png">%s</td>' % (
        makeImgTag("ProposalTrace_DocUsage.png"))
    htmlstr += "</tr>\n"

    htmlstr += "<tr>"
    htmlstr += '<td class="comment">Proposed initial topics.</td>'
    htmlstr += '<td class="png">%s</td>' % (
        makeImgTag("NewComps_Init.png"))
    htmlstr += "</tr>\n"

    fnames = glob.glob(os.path.join(b_debugOutputDir,"NewComps_Step*.png"))
    mnames = glob.glob(os.path.join(b_debugOutputDir,"NewComps_AfterM*.png"))
    for stepID in range(len(fnames)):
        if stepID == len(fnames) - 1 and len(mnames) > 0:
            basenameWithPNG = "NewComps_AfterMerge.png"
            htmlstr += "<tr>"
            htmlstr += '<td class="comment">After merge cleanup.</td>'
            htmlstr += '<td class="png">%s</td>' % (
                makeImgTag(basenameWithPNG))
            htmlstr += "</tr>\n"

        basenameWithPNG = "NewComps_Step%d.png" % (stepID+1)
        htmlstr += "<tr>"
        htmlstr += '<td class="comment">After refinement step %d.</td>' % (
            stepID+1)
        htmlstr += '<td class="png">%s</td>' % (
            makeImgTag(basenameWithPNG))
        htmlstr += "</tr>\n"

    mnames = glob.glob(os.path.join(b_debugOutputDir,"MergeComps_*.png"))
    for mergeID in range(len(mnames)):
        basenameWithPNG = "MergeComps_%d.png" % (mergeID+1)
        htmlstr += "<tr>"
        htmlstr += '<td class="comment">Cleanup Phase: Merged Pair %d</td>' %(
            mergeID+1)
        htmlstr += '<td class="png">%s</td>' % (
            makeImgTag(basenameWithPNG))
        htmlstr += "</tr>\n"


    htmlstr += "</table>"    
    htmlstr += htmlend
    return htmlstr


def makeImgTag(imgbasename="ELBOGain"):
    if imgbasename.count("Trace"):
        width = SQIMG_WIDTH
    else:
        width = WIDEIMG_WIDTH
    htmltag = "<img src=%s width=%d>" % (imgbasename, width)
    return htmltag


def _getLineColorFromELBOKey(key):
    ''' Helper method to assign line colors by ELBO term name

    Returns
    -------
    s : str representing a color value to matplotlib
    '''
    if key.count('total'):
        return 'k'
    elif key.count('data'):
        return 'b'
    elif key.count('entrop'):
        return 'r'
    elif key.count('alloc'):
        return 'c'
    else:
        return 'm'

