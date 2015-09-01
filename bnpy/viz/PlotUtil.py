ExportInfo = dict(
    doExport=False,
    dpi=100,
    W_in=4,
    H_in=4,
)

def ConfigPylabDefaults(pylab):
    rcParams = pylab.rcParams
    rcParams['ps.fonttype'] = 42  # Make fonts export as text (not bitmap)
    rcParams['text.usetex'] = False
    rcParams['legend.fontsize'] = 16
    rcParams['axes.titlesize'] = 18
    rcParams['axes.labelsize'] = 18
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 16
    rcParams['figure.figsize'] = ExportInfo['W_in'], ExportInfo['H_in']
    rcParams['figure.dpi'] = ExportInfo['dpi']
    rcParams['figure.subplot.left'] = 0.15
    rcParams['figure.subplot.right'] = 0.95
    rcParams['figure.subplot.bottom'] = 0.15
    rcParams['figure.subplot.top'] = 0.9
    rcParams['savefig.dpi'] = ExportInfo['dpi']

from matplotlib import pylab
ConfigPylabDefaults(pylab)
