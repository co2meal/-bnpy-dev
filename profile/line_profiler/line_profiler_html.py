#!/usr/bin/env python

import cPickle
from cStringIO import StringIO
import inspect
import linecache
import optparse
import os
import sys
import jinja2
import time

from _line_profiler import LineProfiler as CLineProfiler

CO_GENERATOR = 0x0020
def is_generator(f):
    """ Return True if a function is a generator.
    """
    isgen = (f.func_code.co_flags & CO_GENERATOR) != 0 
    return isgen

# Code to exec inside of LineProfiler.__call__ to support PEP-342-style
# generators in Python 2.5+.
pep342_gen_wrapper = '''
def wrap_generator(self, func):
    """ Wrap a generator to profile it.
    """
    def f(*args, **kwds):
        g = func(*args, **kwds)
        # The first iterate will not be a .send()
        self.enable_by_count()
        try:
            item = g.next()
        finally:
            self.disable_by_count()
        input = (yield item)
        # But any following one might be.
        while True:
            self.enable_by_count()
            try:
                item = g.send(input)
            finally:
                self.disable_by_count()
            input = (yield item)
    return f
'''

class LineProfiler(CLineProfiler):
    """ A profiler that records the execution times of individual lines.
    """

    def __call__(self, func):
        """ Decorate a function to start the profiler on function entry and stop
        it on function exit.
        """
        self.add_function(func)
        if is_generator(func):
            f = self.wrap_generator(func)
        else:
            f = self.wrap_function(func)
        f.__module__ = func.__module__
        f.__name__ = func.__name__
        f.__doc__ = func.__doc__
        f.__dict__.update(getattr(func, '__dict__', {}))
        return f

    if sys.version_info[:2] >= (2,5):
        # Delay compilation because the syntax is not compatible with older
        # Python versions.
        exec pep342_gen_wrapper
    else:
        def wrap_generator(self, func):
            """ Wrap a generator to profile it.
            """
            def f(*args, **kwds):
                g = func(*args, **kwds)
                while True:
                    self.enable_by_count()
                    try:
                        item = g.next()
                    finally:
                        self.disable_by_count()
                    yield item
            return f

    def wrap_function(self, func):
        """ Wrap a function to profile it.
        """
        def f(*args, **kwds):
            self.enable_by_count()
            try:
                result = func(*args, **kwds)
            finally:
                self.disable_by_count()
            return result
        return f

    def dump_stats(self, filename):
        """ Dump a representation of the data to a file as a pickled LineStats
        object from `get_stats()`.
        """
        lstats = self.get_stats()
        f = open(filename, 'wb')
        try:
            cPickle.dump(lstats, f, cPickle.HIGHEST_PROTOCOL)
        finally:
            f.close()

    def print_stats(self, stream=None):
        """ Show the gathered statistics.
        """
        lstats = self.get_stats()
        show_text(lstats.timings, lstats.unit, stream=stream)

    def run(self, cmd):
        """ Profile a single executable statment in the main namespace.
        """
        import __main__
        dict = __main__.__dict__
        return self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals, locals):
        """ Profile a single executable statement in the given namespaces.
        """
        self.enable_by_count()
        try:
            exec cmd in globals, locals
        finally:
            self.disable_by_count()
        return self

    def runcall(self, func, *args, **kw):
        """ Profile a single function call.
        """
        self.enable_by_count()
        try:
            return func(*args, **kw)
        finally:
            self.disable_by_count()


def show_func(filename, start_lineno, func_name, timings, unit, stream=None):
    """ Show results for a single function.
    """
    if stream is None:
        stream = sys.stdout
    print >>stream, "File: %s" % filename
    print >>stream, "Function: %s at line %s" % (func_name, start_lineno)
    template = '%6s %9s %12s %8s %8s  %-s'
    d = {}
    total_time = 0.0
    linenos = []
    for lineno, nhits, time in timings:
        total_time += time
        linenos.append(lineno)
    print >>stream, "Total time: %g s" % (total_time * unit)
    if not os.path.exists(filename):
        print >>stream, ""
        print >>stream, "Could not find file %s" % filename
        print >>stream, "Are you sure you are running this program from the same directory"
        print >>stream, "that you ran the profiler from?"
        print >>stream, "Continuing without the function's contents."
        # Fake empty lines so we can see the timings, if not the code.
        nlines = max(linenos) - min(min(linenos), start_lineno) + 1
        sublines = [''] * nlines
    else:
        all_lines = linecache.getlines(filename)
        sublines = inspect.getblock(all_lines[start_lineno-1:])
    for lineno, nhits, time in timings:
        d[lineno] = (nhits, time, '%5.1f' % (float(time) / nhits),
            '%5.1f' % (100*time / total_time))
    linenos = range(start_lineno, start_lineno + len(sublines))
    empty = ('', '', '', '')
    header = template % ('Line #', 'Hits', 'Time', 'Per Hit', '% Time', 
        'Line Contents')
    print >>stream, ""
    print >>stream, header
    print >>stream, '=' * len(header)
    for lineno, line in zip(linenos, sublines):
        nhits, time, per_hit, percent = d.get(lineno, empty)
        print >>stream, template % (lineno, nhits, time, per_hit, percent,
            line.rstrip('\n').rstrip('\r'))
    print >>stream, ""

def show_text(stats, unit, stream=None):
    """ Show text for the given timings.
    """
    if stream is None:
        stream = sys.stdout
    print >>stream, 'Timer unit: %g s' % unit
    print >>stream, ''
    for (fn, lineno, name), timings in sorted(stats.items()):
        show_func(fn, lineno, name, stats[fn, lineno, name], unit, stream=stream)

def create_html(save_path, template_path, stats, unit):
    """ Create html presentation for the given timings.
    """
    stats = {k:timings for (k,timings) in stats.items() if sum(t[2] for t in timings) != 0}

    create_index_html(save_path, template_path, stats, unit)
    create_graph_html(save_path, template_path, stats, unit)

    # show_text(stats, unit, stream)
    # print >>stream, 'Timer unit: %g s' % unit
    # print >>stream, ''
    for (fn, lineno, name), timings in sorted(stats.items()):
        create_func_html(save_path, template_path, fn, lineno, name, stats[fn, lineno, name], unit)

def create_index_html(save_path, template_path, stats, unit):
    """ Create index.html for the given timings
    """
    stream = open(save_path + 'index.html', mode='w')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = env.get_template('index.template')
    functions = [{'name':name, 
                  'fn': fn, 
                  'fn-replace': fn.replace('/','-'), 
                  'calls': max([0] + [t[1] for t in timings]),
                  'total_time': float('%.2f' % (sum(t[2] for t in timings) * unit))} 
                 for (fn, lineno, name), timings in stats.items()]
    functions = sorted(functions, key=lambda f: f['total_time'], reverse=True)
    for f in functions:
        if f['total_time'] < 0.001:
            f['total_time'] = '--'
        else:
            f['total_time'] = str(f['total_time']) + ' s'
    
    index = template.render(timestamp = time.strftime('%X %x %Z'), tunit=('%g' % unit), functions=functions)
    print >>stream, index

def create_graph_html(save_path, template_path, stats, unit):
    """ Create graph.html for the given timings.
    """
    stream = open(save_path + 'graph.html', mode='w')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = env.get_template('graph.template')
    graph = template.render()
    print >>stream, graph


def create_func_html(save_path, template_path, filename, start_lineno, func_name, timings, unit):
    """ Create html presentation of a given function's results.
    """

    # create directory if doesn't exist
    if not os.path.exists(save_path + 'functions'):
        os.makedirs(save_path + 'functions')

    stream = open(save_path + 'functions/' + filename.replace('/', '-') + '-' + func_name + '.html', mode='w')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    template = env.get_template('single_func_page.template')
    parent_dir = '../../../'

    if not os.path.exists(filename):
        print ""
        print "Could not find file %s" % filename
        print "Continuing without the function's contents."
        # Fake empty lines so we can see the timings, if not the code.
        nlines = max(linenos) - min(min(linenos), start_lineno) + 1
        sublines = [''] * nlines
    else:
        all_lines = linecache.getlines(filename)
        sublines = inspect.getblock(all_lines[start_lineno-1:])

    # line-by-line information and coverage results
    max_len = 30
    total_time = sum(t[2] for t in timings)
    lines = []
    num_code = 0
    num_noncode = 0
    num_run= 0
    num_notrun = 0

    sorted_timings = sorted([(t * unit if not t is None else 0) for (_,_,t) in timings], reverse=True)
    if len(sorted_timings) >= 3:
        threshold = sorted_timings[0]
    else:
        threshold = 0

    d = dict()
    for lineno, nhits, t in timings:
        d[lineno] = (nhits, t * unit, '%5.1f' % (100*t / total_time))

    empty = (0, 0, 0.0)
    linenos = range(start_lineno, start_lineno + len(sublines))
    for lineno, line in zip(linenos, sublines):
        nhits, t, percent = d.get(lineno, empty)
        l = dict()
        l['lineno'] = lineno
        l['code_full'] = line.rstrip('\n').rstrip('\r')
        if len(l['code_full']) == 0:
            l['code_full'] = ' ' # make sure code has at least one character
        if len(l['code_full']) > max_len:
            l['code'] = l['code_full'][:max_len-3] + '...'
        else:
            l['code'] = l['code_full']
        l['calls'] = nhits
        if t == 0:
            l['time'] = '--'
        elif t < 0.01:
            l['time'] = '< 0.01 s'
        else:
            l['time'] = '%.2g s' % t
        l['percent'] = percent
        if t >= threshold and t != 0:
            l['warn'] = 'danger'
        else:
            l['warn'] = ''
        lines.append(l)

        # numbers used for coverage result
        code = l['code'].strip()
        if len(code) == 0:
            num_noncode += 1
        elif (code[0] == '#' or code[0] == '@'):
            num_noncode += 1
        else:
            num_code += 1
            if l['calls'] == 0:
                num_notrun += 1
            else:
                num_run += 1

    # full code listing
    listing = ''
    time_width = max([len(l['time']) for l in lines])
    time_width = max(time_width, len('time')) + 4
    calls_width = max([len(str(l['calls'])) for l in lines])
    calls_width = max(calls_width, len('calls')) + 4
    lineno_width = max([len(str(l['lineno'])) for l in lines])
    lineno_width = max(lineno_width, len('line')) + 4
    leading_spaces = min([(len(l['code_full']) - len(l['code_full'].lstrip())) for l in lines])
    max_code_width = 80
    
    fmt = '{0:<%d}{1:<%d}{2:<%d}{3}\n' % (time_width, calls_width, lineno_width)
    header = fmt.format('time', 'calls', 'line', 'code')
    listing = header + '\n'
    for l in lines:
        l['code_full'] = l['code_full'][leading_spaces:]
        if len(l['code_full']) < max_code_width:
            listing += fmt.format(l['time'], l['calls'], l['lineno'], l['code_full'])
        else:
            indent = len(l['code_full']) - len(l['code_full'].lstrip())
            listing += fmt.format(l['time'], l['calls'], l['lineno'], l['code_full'][:max_code_width])
            listing += ' ' * indent
            listing += fmt.format('', '', '', l['code_full'][max_code_width:])

    func = template.render(timestamp = time.strftime('%X %x %Z'),
                           name = func_name,
                           num_calls = max([0] + [t[1] for t in timings]),
                           seconds = '%g' % (sum(t[2] for t in timings) * unit),
                           link = filename,
                           parent_dir=parent_dir,
                           lines = lines,
                           code_full = listing,
                           total_lines = len(sublines),
                           non_code_lines = num_noncode,
                           code_lines = num_code,
                           run = num_run,
                           not_run = num_notrun,
                           run_percent = '%5.1f' % (100*float(num_run)/float(num_code)))
    print >>stream, func

# A %lprun magic for IPython.
def magic_lprun(self, parameter_s=''):
    """ Execute a statement under the line-by-line profiler from the
    line_profiler module.

    Usage:
      %lprun -f func1 -f func2 <statement>

    The given statement (which doesn't require quote marks) is run via the
    LineProfiler. Profiling is enabled for the functions specified by the -f
    options. The statistics will be shown side-by-side with the code through the
    pager once the statement has completed.

    Options:
    
    -f <function>: LineProfiler only profiles functions and methods it is told
    to profile.  This option tells the profiler about these functions. Multiple
    -f options may be used. The argument may be any expression that gives
    a Python function or method object. However, one must be careful to avoid
    spaces that may confuse the option parser. Additionally, functions defined
    in the interpreter at the In[] prompt or via %run currently cannot be
    displayed.  Write these functions out to a separate file and import them.

    One or more -f options are required to get any useful results.

    -D <filename>: dump the raw statistics out to a pickle file on disk. The
    usual extension for this is ".lprof". These statistics may be viewed later
    by running line_profiler.py as a script.

    -T <filename>: dump the text-formatted statistics with the code side-by-side
    out to a text file.

    -r: return the LineProfiler object after it has completed profiling.
    """
    # Local imports to avoid hard dependency.
    from distutils.version import LooseVersion
    import IPython
    ipython_version = LooseVersion(IPython.__version__)
    if ipython_version < '0.11':
        from IPython.genutils import page
        from IPython.ipstruct import Struct
        from IPython.ipapi import UsageError
    else:
        from IPython.core.page import page
        from IPython.utils.ipstruct import Struct
        from IPython.core.error import UsageError

    # Escape quote markers.
    opts_def = Struct(D=[''], T=[''], f=[])
    parameter_s = parameter_s.replace('"',r'\"').replace("'",r"\'")
    opts, arg_str = self.parse_options(parameter_s, 'rf:D:T:', list_all=True)
    opts.merge(opts_def)

    global_ns = self.shell.user_global_ns
    local_ns = self.shell.user_ns

    # Get the requested functions.
    funcs = []
    for name in opts.f:
        try:
            funcs.append(eval(name, global_ns, local_ns))
        except Exception, e:
            raise UsageError('Could not find function %r.\n%s: %s' % (name, 
                e.__class__.__name__, e))

    profile = LineProfiler(*funcs)

    # Add the profiler to the builtins for @profile.
    import __builtin__
    if 'profile' in __builtin__.__dict__:
        had_profile = True
        old_profile = __builtin__.__dict__['profile']
    else:
        had_profile = False
        old_profile = None
    __builtin__.__dict__['profile'] = profile

    try:
        try:
            profile.runctx(arg_str, global_ns, local_ns)
            message = ''
        except SystemExit:
            message = """*** SystemExit exception caught in code being profiled."""
        except KeyboardInterrupt:
            message = ("*** KeyboardInterrupt exception caught in code being "
                "profiled.")
    finally:
        if had_profile:
            __builtin__.__dict__['profile'] = old_profile

    # Trap text output.
    stdout_trap = StringIO()
    profile.print_stats(stdout_trap)
    output = stdout_trap.getvalue()
    output = output.rstrip()

    if ipython_version < '0.11':
        page(output, screen_lines=self.shell.rc.screen_length)
    else:
        page(output)
    print message,

    dump_file = opts.D[0]
    if dump_file:
        profile.dump_stats(dump_file)
        print '\n*** Profile stats pickled to file',\
              `dump_file`+'.',message

    text_file = opts.T[0]
    if text_file:
        pfile = open(text_file, 'w')
        pfile.write(output)
        pfile.close()
        print '\n*** Profile printout saved to text file',\
              `text_file`+'.',message

    return_value = None
    if opts.has_key('r'):
        return_value = profile

    return return_value


def load_ipython_extension(ip):
    """ API for IPython to recognize this module as an IPython extension.
    """
    ip.define_magic('lprun', magic_lprun)


def load_stats(filename):
    """ Utility function to load a pickled LineStats object from a given
    filename.
    """
    f = open(filename, 'rb')
    try:
        lstats = cPickle.load(f)
    finally:
        f.close()
    return lstats



def main():
    usage = "usage: %prog profile.lprof"
    parser = optparse.OptionParser(usage=usage, version='%prog 1.0b2')

    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error("Must provide a file containing profiling results, a directory containing templates, and a directory to save profiling results.")
    lstats = load_stats(args[0])
    # show_text(lstats.timings, lstats.unit)
    create_html(args[2], args[1], lstats.timings, lstats.unit)

if __name__ == '__main__':
    main()
