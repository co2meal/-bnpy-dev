from PlotTrace import plotJobsThatMatchKeywords, parse_args

if __name__ == "__main__":
  argDict = parse_args(xvar='laps', yvar='K')
  plotJobsThatMatchKeywords(**argDict)
