import os
import sys
import MakeTargetPlots as MTP

def MakeHTMLForPlan(savepath, Plan, planID):
  if savepath.count('Bars'):
    with open('Template-Target-Bars.html','r') as f:
      TemplateLines = f.readlines()
  else:
    with open('Template-Target-Real.html','r') as f:
      TemplateLines = f.readlines()

  MTP.MakePlotsForTargetDataPlan(Plan, os.path.join(savepath, str(planID)))

  MyLines = fillHTMLTemplate(TemplateLines, Plan, savepath, planID)
  fpath = os.path.join(savepath, '%d.html'%(planID))
  print fpath
  with open(fpath, 'w') as f:
    for line in MyLines:
      f.write(line)  

def fillHTMLTemplate(TemplateLines, Plan, path, task):
  if type(task) == int:
    task = str(task)
  Translations = dict(
    ICON1=os.path.join('1', 'TargetIcon.png'),
    ICON2=os.path.join('2', 'TargetIcon.png'),
    ICON3=os.path.join('3', 'TargetIcon.png'),
    ICON4=os.path.join('4', 'TargetIcon.png'),
    ICONNOW=os.path.join(task, 'TargetIcon.png'),
    TARGETDATANOW=os.path.join(task, 'TargetData.png'),
    ORIGTOPICNOW=os.path.join(task, 'OriginalTopics.png'),
    LOGNOW='\n'.join([line for line in Plan['log']])
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


"""
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
"""
