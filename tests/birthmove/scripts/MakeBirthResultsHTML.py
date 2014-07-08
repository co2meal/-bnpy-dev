import argparse
import glob
import os
import sys

import MakePlotsForSingleBirthExperiment as Plotter

def MakeTranslationsForJob(TemplateLines, path, task):
  if type(task) == int:
    task = str(task)
  Translations = dict(
    ELBO1=os.path.join('1', 'TraceELBO.png'),
    ELBO2=os.path.join('2', 'TraceELBO.png'),
    ELBO3=os.path.join('3', 'TraceELBO.png'),
    ELBO4=os.path.join('4', 'TraceELBO.png'),
    ORIGTOPICWORDCLOUD=os.path.join(task, 'OriginalWordCloud.png'),
    INITTOPICWORDCLOUD=os.path.join(task, 'InitWordCloud.png'),
    FINALTOPICWORDCLOUD=os.path.join(task, 'FinalWordCloud.png'),
    TARGETWORDCLOUD=os.path.join(task, 'TargetDataWordCloud.png'),
    ORIGTOPICNOW=os.path.join(task, 'OriginalTopics.png'),
    INITTOPICNOW=os.path.join(task, 'InitTopics.png'),
    FINALTOPICNOW=os.path.join(task, 'FinalTopics.png'),
    ALLTOPICNOW=os.path.join(task, 'Topics.png'),
    TARGETNOW=os.path.join(task, 'TargetData.png'),
    STATSNOW=os.path.join(task, 'TraceStats.png'),
    ELBONOW=os.path.join(task, 'TraceELBO.png'),
    SELECTNOW=os.path.join(task, 'TargetSelection.png'),
    )

  MyLines = list()
  for line in TemplateLines:
    doSkipThisLine = 0
    for key, dest in Translations.items():
      fullpath = os.path.join(path, dest)
      if line.count(key) > 0:    
        if not os.path.exists(fullpath) and fullpath.count(os.path.sep+task):
          doSkipThisLine = 1
      line = line.replace(key, dest)
    if not doSkipThisLine:
      MyLines.append(line)
  return MyLines

def MakeHTMLForJob(TemplateLines, path):
  for task in [1, 2, 3, 4]:
    Plotter.MakePlotsForSavedBirthMove(os.path.join(path, str(task)))

    MyLines = MakeTranslationsForJob(TemplateLines, path, task)
    fpath = os.path.join(path, '%d.html'%(task))
    print fpath
    with open(fpath, 'w') as f:
      for line in MyLines:
        f.write(line)  

if __name__ == '__main__':
  root = '/final-results/birth-results/'
  parser = argparse.ArgumentParser()
  parser.add_argument('data')
  parser.add_argument('initName')
  parser.add_argument('--pattern', default='*')
  args = parser.parse_args()

  if args.data.count('Bars') > 0:
    with open('BirthResultsTemplate.html','r') as f:
      TemplateLines = f.readlines()
  else:
    with open('BirthResultsTemplateRealData.html','r') as f:
      TemplateLines = f.readlines()

  if args.pattern.count('*') == 0:
    args.pattern = '*' + args.pattern + '*'

  pattern = os.path.join(root, args.data, args.initName+args.pattern)
  fList = glob.glob(pattern)
  for f in fList:
    MakeHTMLForJob(TemplateLines, f)

  #jstr ='/final-results/birth-results/BarsK10V900/K=8-best-xspectral-Kfresh5/'
  #MakeHTMLForJob(jstr)