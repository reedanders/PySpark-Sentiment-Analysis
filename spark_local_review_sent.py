# MLlibs
# from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from numpy import array

# Exclude for AWS implementation
conf = SparkConf()
sc = SparkContext(conf=conf)

# 0 128:51 129:159 130:253 131:159 [Label KEY:VALUE]
training_data = MLUtils.loadLibSVMFile(sc, "train_libsvm.txt")

# 231547 128:51 129:159 130:253 131:159 [PhraseId KEY:VALUE]
testing_data = MLUtils.loadLibSVMFile(sc, "test_libsvm.txt")

# Build the model
model = DecisionTree.trainClassifier(training_data, numClasses=5, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=100)

# Evaluating the model on testing_data data
labelsAndPreds = testing_data.map(lambda p: (p.label, model.predict(p.features)))

test = {}

for i in labelsAndPreds.collect():
	test[i[1]] = i[0]

# Write predictions
o = DictWriter(open('pred.csv', 'w'), ['PhraseId', 'Sentiment'])
o.writeheader()
for ii in sorted(test):
	o.writerow({'PhraseId': ii, 'Sentiment': test[ii]})