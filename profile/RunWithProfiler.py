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

try:
  import bnpy
  bnpyroot = os.path.sep.join(os.path.abspath(bnpy.__file__).split(os.path.sep)[:-1])
except:
  bnpyroot = os.path.abspath('../bnpy/')
print bnpyroot

# Decorate codebase
print "Decorating ...",
decorate_for_profiling.main(bnpyroot)
print '[DONE]'

# Run line-by-line profiler
ProfileCMD = "python line_profiler/kernprof.py --line-by-line %s %s" \
        % (os.path.join(bnpyroot,'Run.py'), ' '.join(sys.argv[1:]))
print "Running script with profiling enabled ...",
subprocess.call(ProfileCMD, shell=True)
print '[DONE]'

print "Building HTML ...",
# Convert output to HTML
toHTMLCMD = "python line_profiler/line_profiler_html.py Run.py.lprof assets/templates/ reports/" 
subprocess.call(toHTMLCMD, shell=True)
print '[DONE]'

# Undecorate codebase
print "Undecorating ...",
undecorate_for_profiling.main(bnpyroot)
print '[DONE]'
