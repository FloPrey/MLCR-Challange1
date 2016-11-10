from Analyzer import Analyzer

analyzer = Analyzer()

analyzer.loadData()
analyzer.improveData()
analyzer.splitData()
analyzer.classifyBayesGausch()
#analyzer.classifyKNearest()
#analyzer.classifySVC()
analyzer.convert()
