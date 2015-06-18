""" 
ExecuteNotebook.py

Script to execute all cells of Ipython notebook from the command-line.

Output is written to the same notebook, overwriting the previous plots.

Usage: `ExecuteNotebook.py foo.ipynb` 
"""
import argparse
import io
import os
import sys
import time
from Queue import Empty

try:
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager import BlockingKernelManager as KernelManager
 
from IPython.nbformat.current import reads, NotebookNode, write


def execute_notebook(ipynbfilepath=None, runLongwindedCells=False):
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
    starttime = time.time()
    for ws in nb.worksheets:
        nCodeCells = len([c for c in ws.cells if c.cell_type == 'code'])
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            try:
                outs = run_cell(
                    shell, iopub, cell, 
                    runLongwindedCells=runLongwindedCells)
            except Exception as e:
                print '>>>>>>>>>>>>>> FAILED TO RUN CELL'
                print "Error Msg:", str(e)
                print cell.input
                raise ValueError('Cell Execution Failed')
                nError += 1
                cell.outputs = [e]
                continue

            nSuccess += 1
            cell.outputs = outs
            cell.prompt_number = prompt_number
            if cell.outputs:
                cell.outputs[0]['prompt_number'] = prompt_number
            prompt_number += 1

            elapsedtime = time.time() - starttime
            percentComplete = 100 * nSuccess/float(nCodeCells)
            sys.stdout.write("\r%.1f%% cells after %.2f sec" % (
                percentComplete, elapsedtime))
            sys.stdout.flush()

    print "%3i/%3i cells executed correctly." % (nSuccess, nSuccess+nError)
    if nError:
        print "%3i cells raised errors." % nError
    kc.stop_channels()
    km.shutdown_kernel()
    del km
    with io.open(ipynbfilepath, 'w', encoding='utf8') as f:
        write(nb, f, 'json')
    print "Wrote output to file: %s" % ipynbfilepath


def run_cell(shell, iopub, cell, max_timeout_sec=120, runLongwindedCells=0):
    ''' Execute a single cell of an IPython notebook.

    Returns
    -------
    outs : list
        each element is one output of this cell
    '''
    stime = time.time()

    firstline = cell.input.split('\n')[0]
    if firstline.startswith('# ExpectedRunTime='):
        expectedtime = firstline.split('=')[1]
        expectedtime = expectedtime.replace('sec', '')
        expectedtime = float(expectedtime)
        if expectedtime > max_timeout_sec:
            # Skip it!
            return []

    shell.execute(cell.input)
    # wait for finish or timeout (in seconds)
    shell.get_msg(timeout=max_timeout_sec)
    outs = []
    
    elapsedtime = time.time() - stime
    # print ' %.2f sec | cell done.' % (elapsedtime)
    while True:
        try:
            msg = iopub.get_msg(timeout=1.0)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue
        
        content = msg['content']
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ipynbfilepath',
        default='', type=str, help='Filepath to IPython notebook (__.ipynb)')
    args = parser.parse_args()
    execute_notebook(**args.__dict__)
