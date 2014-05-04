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

outroot = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-1])
outputhtmldir = os.path.join(outroot, "reports/MyProfile/")

try:
  import bnpy
  bnpyroot = os.path.sep.join(os.path.abspath(bnpy.__file__).split(os.path.sep)[:-1])
except:
  bnpyroot = os.path.abspath('../bnpy/')

fparts = bnpyroot.split(os.path.sep)[:-1]
fparts.append('third-party')
thirdpartyroot = os.path.sep.join(fparts)
print bnpyroot
print thirdpartyroot

# Decorate codebase
print "Decorating ...",
undecorate_for_profiling.main(bnpyroot) # Remove previous decos, if any
decorate_for_profiling.main(bnpyroot)
undecorate_for_profiling.main(thirdpartyroot)
decorate_for_profiling.main(thirdpartyroot)
print '[DONE]'

# Run line-by-line profiler
if 'PYTHONEXE' in os.environ:
  pycmdstr = os.environ['PYTHONEXE']
else:
  pycmdstr = 'python'

ProfileCMD = "%s line_profiler/kernprof.py --line-by-line %s %s" \
        % (pycmdstr, os.path.join(bnpyroot,'Run.py'), ' '.join(sys.argv[1:]))
print "Running script with profiling enabled ...",
subprocess.call(ProfileCMD, shell=True)
print '[DONE]'

print "Building HTML ...",
# Convert output to HTML
toHTMLCMD = "python line_profiler/line_profiler_html.py Run.py.lprof assets/templates/ " + outputhtmldir 
subprocess.call(toHTMLCMD, shell=True)
print '[DONE]'

# Undecorate codebase
print "Undecorating ...",
undecorate_for_profiling.main(bnpyroot)
undecorate_for_profiling.main(thirdpartyroot)
print '[DONE]'

print "Wrote HTML to %s/index.html" % (outputhtmldir)

# Clean up extra files (Run.py.lprof)
extrafilename = 'Run.py.lprof'
if os.path.exists(extrafilename):
  os.remove(extrafilename)
