'''
RunWithProfiler.py

Usage
-------
python RunWithProfiler.py [same args as bnpy.Run]
'''

import decorate_for_profiling
import undecorate_for_profiling
import os
import sys
import subprocess
import datetime

nowobj = datetime.datetime.now()
nowstr = '%4d%02d%02d-%02d%02d-%06d' % (nowobj.year,
                                        nowobj.month,
                                        nowobj.day,
                                        nowobj.hour,
                                        nowobj.minute,
                                        nowobj.microsecond)

outroot = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-1])
outputhtmldir = os.path.join(outroot, "reports/%s/" % (nowstr))
# important that this DOESNOT have a trailing slash
outputlinkdir = os.path.join(outroot, "reports/latest")

try:
    import bnpy
    bnpyroot = os.path.sep.join(
        os.path.abspath(
            bnpy.__file__).split(
            os.path.sep)[
                :-
                1])
except:
    bnpyroot = os.path.abspath('../bnpy/')

fparts = bnpyroot.split(os.path.sep)[:-1]
fparts.append('third-party')
thirdpartyroot = os.path.sep.join(fparts)

# Decorate codebase
print "Decorating ...",
undecorate_for_profiling.main(bnpyroot)  # Remove previous decos, if any
decorate_for_profiling.main(bnpyroot)
undecorate_for_profiling.main(thirdpartyroot)
decorate_for_profiling.main(thirdpartyroot)
print '[DONE]'

# Run line-by-line profiler
if 'PYTHONEXE' in os.environ:
    pycmdstr = os.environ['PYTHONEXE']
else:
    pycmdstr = 'python'

bnpyRunPath = os.path.join(bnpyroot, 'Run.py')
kwargsStr = ' '.join(sys.argv[1:])
ProfileCMD = "%s line_profiler/kernprof.py -o %s.lprof --line-by-line %s %s" \
    % (pycmdstr, nowstr, bnpyRunPath, kwargsStr)
subprocess.call(ProfileCMD, shell=True)


print "Building HTML ...",
# Convert output to HTML
toHTMLCMD = "python line_profiler/line_profiler_html.py " + \
    nowstr + ".lprof assets/templates/ " + outputhtmldir
subprocess.call(toHTMLCMD, shell=True)
print '[DONE]'

# Undecorate codebase
print "Undecorating ...",
undecorate_for_profiling.main(bnpyroot)
undecorate_for_profiling.main(thirdpartyroot)
print '[DONE]'

print "Wrote HTML to %s/index.html" % (outputhtmldir)

# Clean up extra files (Run.py.lprof)
extrafilename = '%s.lprof' % (nowstr)
if os.path.exists(extrafilename):
    os.remove(extrafilename)

# Make symbolic link to "latest" folder"
if os.path.islink(outputlinkdir):
    os.unlink(outputlinkdir)
os.symlink(outputhtmldir, outputlinkdir)
