'''
undecorate_for_profiling.py

Explore all the python functions in the user-specified directory,
and remove decoration @profile from appropriate functions
'''
import os

def main(bnpyrootdir):
  list_of_files = {}
  for (dirpath, contentdirs, contentfiles) in os.walk(bnpyrootdir):
    for fname in contentfiles:
      if fname[-3:] == '.py':
        fullpathkey = os.sep.join([dirpath,fname])
        list_of_files[fullpathkey] = fname

  for origPath in list_of_files.keys():
    profPath = origPath + 'CLEAN'
    profFileObj = open(profPath, 'w')
    with open(origPath, 'r') as f:
      for line in f.readlines():
        if line.count( '@profile' ) == 0:
          profFileObj.write( line )
    profFileObj.close()
    os.rename(profPath, origPath)

