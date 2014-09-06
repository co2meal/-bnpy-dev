""" 
ExecuteNotebook.py

Script to run all cells of Ipython notebook non-interactively from command-line.

Output is written to the same notebook, overwriting the previous plots.

Usage: `ExecuteNotebook.py foo.ipynb` 
"""
import argparse
import io
import os,sys,time
import shutil
import fileinput
from Queue import Empty
import commands
from distutils.dir_util import mkpath

#WEBDIR=/pro/web/web/people/mhughes/research/topic-model-notes/
WEBDIR = '/Users/mhughes/Desktop/'

fields = os.path.abspath(__file__).split(os.path.sep)
assetdir = os.path.sep.join(fields[:-1])

try:
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager import BlockingKernelManager as KernelManager
 
from IPython.nbformat.current import reads, NotebookNode, write

def run_cell(shell, iopub, cell):
    # print cell.input
    shell.execute(cell.input)
    # wait for finish, maximum 20s
    shell.get_msg(timeout=20)
    outs = []
    
    while True:
        try:
            msg = iopub.get_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue
        
        content = msg['content']
        # print msg_type, content
        out = NotebookNode(output_type=msg_type)
        
        if msg_type == 'stream':
            out.stream = content['name']
            out.text = content['data']
        elif msg_type in ('display_data', 'pyout'):
            for mime, data in content['data'].iteritems():
                attr = mime.split('/')[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(out, attr, data)
            if msg_type == 'pyout':
                #out.prompt_number = content['execution_count']
                #TODO: need to find better workaround
                pass
        elif msg_type == 'pyerr':
            out.ename = content['ename']
            out.evalue = content['evalue']
            out.traceback = content['traceback']
        else:
            print "unhandled iopub msg:", msg_type
        
        outs.append(out)
    return outs
    
 
def writeReportForTask(taskpath):
    ipynbfilepath = create_task_report_notebook(taskpath)

    print "Creating Report as IPython Notebook: ", ipynbfilepath
    with open(ipynbfilepath) as f:
        nb = reads(f.read(), 'json')
    km = KernelManager()
    km.start_kernel(extra_arguments=['--pylab=inline'],
                    stderr=open(os.devnull, 'w'))
    try:
        kc = km.client()
        kc.start_channels()
        iopub = kc.iopub_channel
    except AttributeError:
        # IPython 0.13
        kc = km
        kc.start_channels()
        iopub = kc.sub_channel
    shell = kc.shell_channel
    
    # run %pylab inline, because some notebooks assume this
    # even though they shouldn't
    shell.execute("pass")
    shell.get_msg()
    while True:
        try:
            iopub.get_msg(timeout=1)
        except Empty:
            break
    
    nSuccess = 0
    nError = 0
    prompt_number = 1
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            try:
                outs = run_cell(shell, iopub, cell)
            except Exception as e:
                print "failed to run cell:", repr(e)
                print cell.input
                nError += 1
                cell.outputs = [e]
                continue
            
            nSuccess += 1
            print '.',
 
            cell.outputs = outs
            cell.prompt_number = prompt_number
            if cell.outputs:
                cell.outputs[0]['prompt_number'] = prompt_number
            prompt_number += 1
 
    print 'DONE.'
    print "%3i/%3i cells executed correctly." % (nSuccess, nSuccess+nError)
    if nError:
        print "%3i cells raised errors." % nError
    kc.stop_channels()
    km.shutdown_kernel()
    del km
    with io.open(ipynbfilepath, 'w', encoding='utf8') as f:
        write(nb, f, 'json')
    print "Wrote output to file: %s" % ipynbfilepath

    convert_notebook_to_public_html(taskpath)
    print "Converted to HTML: %s" % (ipynbfilepath.replace('.ipynb', '.html'))

def create_task_report_notebook(taskpath):
    ''' Copy TaskReport.ipynb into taskpath, replacing relevant template lines
    '''
    outpath = os.path.join(taskpath, 'TaskReport.ipynb')
    if os.path.exists(outpath):
        return outpath
    try:
        shutil.copy(os.path.join(assetdir,'TaskReport.ipynb'),
                    outpath)

        for line in fileinput.input(outpath, inplace=True):
            print line.replace('$TASKPATH', taskpath), # <-- keep this comma

        return outpath
    except e:
        print str(e)
        return False

def convert_notebook_to_public_html(taskpath):
    ''' Convert TaskReport notebook to HTML
    '''
    ipynbpath = os.path.join(taskpath, 'TaskReport.ipynb')
    htmlpath = os.path.join(taskpath, 'TaskReport') # .html already added in
    logfilepath = os.path.join(taskpath, 'stdout.log')
    CMD = "ipython nbconvert %s --to html --output %s >> %s" \
           % (ipynbpath, htmlpath, logfilepath)
    stdout = commands.getoutput(CMD)
    print stdout

    if os.path.exists(WEBDIR):
      jobpath = taskpath.replace(os.environ['BNPYOUTDIR'], '')
      mkpath(os.path.join(WEBDIR, jobpath))
      shutil.copy(os.path.join(taskpath,'TaskReport.html'),
                  os.path.join(WEBDIR, jobpath, 'TaskReport.html')
                 )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run iPython notebook ' +
                     'non-interactively and save results to new notebook')
    parser.add_argument('input_ipynb', action='store',
                        help='iPython notebook file to run')
    args = parser.parse_args()
    exe_notebook(args.input_ipynb)

