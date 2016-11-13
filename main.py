from Analyzer import Analyzer

analyzer = Analyzer()

randomForest = "RandomForest"
knn = "KNN"
naiveBayes = "NaiveBayes"
svm = "SVM"

analyzer.loadData()

# you can change the optimization value - but we recommend 80%
# more details can be found in the readme
analyzer.improveData(80, True)
analyzer.createSets()

# please pass on the classifier of your choice - they can be different
analyzer.predictFloor(randomForest)
analyzer.predictLocation(randomForest)
