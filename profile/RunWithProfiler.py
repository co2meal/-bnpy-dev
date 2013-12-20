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
decorate_for_profiling.main(bnpyroot)

# Run line-by-line profiler
ProfileCMD = "python line_profiler/kernprof.py --line-by-line %s %s" \
        % (os.path.join(bnpyroot,'Run.py'), ' '.join(sys.argv[1:]))
print ProfileCMD
subprocess.call(ProfileCMD, shell=True)

# Convert output to HTML
toHTMLCMD = "python line_profiler/line_profiler_html.py Run.py.lprof assets/templates/ reports/" 
subprocess.call(toHTMLCMD, shell=True)

# Undecorate codebase
undecorate_for_profiling.main(bnpyroot)

