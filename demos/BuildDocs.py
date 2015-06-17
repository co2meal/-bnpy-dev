'''
BuildDocs.py

Convert all IPython notebook demos into documentation html pages.

Usage
------
To build all notebooks into docs/
$ python BuildDocs.py *.ipynb 

To build only one notebook
$ python BuildDocs.py MyNotebook.ipynb 
'''
import argparse
import os
import sys
import glob
import commands

#from ExecuteNotebook import execute_notebook

def rewritePlotUtilSourceFile(doExport=True):
    fpath = os.path.expandvars('$BNPYROOT/bnpy/viz/PlotUtil.py')
    f = open(fpath, 'r')
    lines = f.readlines()
    for n, line in enumerate(lines):
        if line.count('doExport'):
            if doExport:
                lines[n] = line.replace('doExport=False', 'doExport=True')
            else:
                lines[n] = line.replace('doExport=True', 'doExport=False')
    f.close()   

    with open(fpath, 'w') as f:
        f.writelines(lines)

def convert_ipynb_to_rst_documentation(
        ipynbfilepath='',
        execute=0):
    ''' Convert notebook to .rst file, inside docs/ folder.
    '''
    demooutroot = os.path.expandvars('$BNPYROOT/docs/source/demos/')
    rstoutpath = os.path.join(
        demooutroot,
        ipynbfilepath.replace('.ipynb', '.rst'))
    logfilepath = '.ipynbconversion.stdout'

    # First, execute the notebook
    if execute:
        CMD = "ipython nbconvert %s " + \
            "--ExecutePreprocessor.timeout=180 " + \
            "--execute --to notebook --output %s"
        CMD = CMD  % (ipynbfilepath, ipynbfilepath)
        stdout = commands.getoutput(CMD)
        print stdout

    # Next, convert to rst
    CMD = "ipython nbconvert %s " + \
        "--to rst --output %s"
    CMD = CMD  % (ipynbfilepath, rstoutpath.replace('.rst',''))
    print CMD
    stdout = commands.getoutput(CMD)
    print stdout

    # Fix broken image links
    # Before: ..image directives link to full local filesystem path
    # After: link is relative to documenation root dir (docs/source/)
    fullsrcpath = os.path.expandvars('$BNPYROOT/docs/source/')
    fixedLines = list()
    with open(rstoutpath, 'r') as f:
        for line in f.readlines():
            if line.count(fullsrcpath):
                line = line.replace(fullsrcpath, '/')
            fixedLines.append(line)

    with open(rstoutpath, 'w') as f:
        f.writelines(fixedLines)
    
def main(ipynbfilepattern='*.ipynb', doExportImages=True):
    if doExportImages:
        rewritePlotUtilSourceFile(doExport=True)

    from bnpy.viz.PlotUtil import ExportInfo
    print ExportInfo

    for ipynbfilepath in sorted(glob.glob(ipynbfilepattern)):
        print ipynbfilepath
        
        #print '---------------- >>> execute_notebook STARTED'
        #execute_notebook(ipynbfilepath=ipynbfilepath)
        #print '---------------- >>> execute_notebook FINISHED'

        print '---------------- >>> nbconvert STARTED'
        convert_ipynb_to_rst_documentation(ipynbfilepath=ipynbfilepath)
        print '---------------- <<< nbconvert FINISHED'

    if doExportImages:
        rewritePlotUtilSourceFile(doExport=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ipynbfilepattern',
        default='*.ipynb', 
        type=str, help='Pattern for IPython notebook (*.ipynb)')
    args = parser.parse_args()
    main(**args.__dict__)
