from Analyzer import Analyzer

analyzer = Analyzer()

analyzer.loadData()
analyzer.improveData()
analyzer.splitData()
analyzer.classifyBayesGausch()
analyzer.classifyNaiveBayes()
#analyzer.classifyKNearest()
#analyzer.classifySVC()
analyzer.convert()
