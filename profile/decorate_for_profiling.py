'''
decorate_for_profiling.py

Explore all the python functions in the user-specified directory
and decorate the appropriate functions with @profile
'''

import os

def main(rootdir):
  with open('func_names.txt','r') as f:
    TARGET_FUNCS = [ fname.strip() for fname in f.readlines()]

  # Discard empty entries (like blank lines)
  if len(TARGET_FUNCS) > 0:
    keepList = []
    for aa in range(len(TARGET_FUNCS)):
      curName = TARGET_FUNCS[aa]
      if len(curName.strip()) > 0:
        keepList.append(curName)
    TARGET_FUNCS = keepList

  list_of_files = {}
  for (dirpath, contentdirs, contentfiles) in os.walk(rootdir):
    for fname in contentfiles:
      if fname[-3:] == '.py':
        fullpathkey = os.sep.join([dirpath,fname])
        list_of_files[fullpathkey] = fname

  for origpath in list_of_files.keys():
    profpath = origpath + 'PROFILE'
    profFileObj = open(profpath, 'w')
    with open(origpath, 'r') as f:
      for line in f.readlines():
        sline = line.strip()
        if sline.startswith( 'def' ):
          funcname = sline.split( ' ' )[1]
          for targetname in TARGET_FUNCS:
            if funcname.startswith(targetname):
              nspaces = len(line) -len(line.lstrip())
              profFileObj.write( ' '*nspaces +'@profile\n' )
              break
        profFileObj.write( line )
    profFileObj.close()
    # NOW, REPLACE .py files with their .pyPROFILE counterparts
    os.rename(profpath, origpath)
