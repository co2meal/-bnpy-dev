
from PlotELBO import plotJobsThatMatchKeywords, parse_args

if __name__ == "__main__":
  argDict = parse_args()
  plotJobsThatMatchKeywords(yvar='K', **argDict)