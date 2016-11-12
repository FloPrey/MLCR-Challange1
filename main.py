from Analyzer import Analyzer

analyzer = Analyzer()

analyzer.loadData()
analyzer.improveData(80, True)
analyzer.createSets()
analyzer.predictFloor()
analyzer.predictLocation()
