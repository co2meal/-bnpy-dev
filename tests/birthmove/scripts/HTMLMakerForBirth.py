import os
import sys
import MakeBirthPlots as MBP

def MakeHTMLForBirth(savepath, BirthResults, CurResults, Data, planID):
  if savepath.lower().count('bars'):
    with open('Template-Birth-Bars.html','r') as f:
      TemplateLines = f.readlines()
  else:
    with open('Template-Birth-Real.html','r') as f:
      TemplateLines = f.readlines()

  MBP.MakePlots(BirthResults, CurResults, Data,
                 os.path.join(savepath, str(planID)))

  MyLines = fillHTMLTemplate(TemplateLines, BirthResults, savepath, planID)
  fpath = os.path.join(savepath, '%d.html'%(planID))
  with open(fpath, 'w') as f:
    for line in MyLines:
      f.write(line)  
  print '... HTML OUTPUT:', fpath

def fillHTMLTemplate(TemplateLines, BirthResults, path, task):
  if type(task) == int:
    task = str(task)
  targetparts = path.split(os.path.sep)
  targethtmlfile = targetparts[-2] + '.html'
  Translations = dict(
    ELBO1=os.path.join('1', 'ELBO.png'),
    ELBO2=os.path.join('2', 'ELBO.png'),
    ELBO3=os.path.join('3', 'ELBO.png'),
    ELBO4=os.path.join('4', 'ELBO.png'),
    ELBONOW=os.path.join(task, 'ELBO.png'),
    STATSNOW=os.path.join(task, 'TraceStats.png'),
    TDLINK=targethtmlfile,
    TARGETDATA=os.path.join('../', 'TargetData.png'),
    INITTOPICS=os.path.join(task, 'InitTopics.png'),
    FINALTOPICS=os.path.join(task, 'FinalTopics.png'),
    MINIMALTOPICS=os.path.join(task, 'MinimalFinalTopics.png'),
    )

  MyLines = list()
  for line in TemplateLines:
    doSkipThisLine = 0
    for key, dest in Translations.items():
      fullpath = os.path.join(path, dest)
      if line.count(key) > 0:    
        parts = fullpath.split(os.path.sep)[-2:]
        basepath = os.path.sep.join(parts)
        if not os.path.exists(fullpath) and basepath.count(os.path.sep+task):
          doSkipThisLine = 1
          print 'SKIPPING:', basepath
      line = line.replace(key, dest)
    if not doSkipThisLine:
      MyLines.append(line)
  return MyLines