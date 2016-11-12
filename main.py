from Analyzer import Analyzer

analyzer = Analyzer()

analyzer.loadData()
# analyzer.improveData(percentage for ap occurence, set values <-85 to -85)
analyzer.improveData(70, True)
analyzer.predictFloor()
analyzer.predictLocation()

