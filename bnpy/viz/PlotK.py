
from PlotELBO import main, parse_args

if __name__ == "__main__":
  argDict = parse_args()
  main(yvar='K', **argDict)