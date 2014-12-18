# MLlibs
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from numpy import array

conf = SparkConf()
sc = SparkContext(conf=conf)

# 0 128:51 129:159 130:253 131:159 [Label KEY:VALUE]
training_data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

# # 231547 128:51 129:159 130:253 131:159 [PhraseId KEY:VALUE]
# testing_data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

# Build the model
model = LogisticRegressionWithSGD.train(training_data)

# Evaluating the model on training_data data
labelsAndPreds = training_data.map(lambda p: (p.label, model.predict(p.features)))
print training_data.count()

# # Evaluating the model on testing_data data
# labelsAndPreds = testing_data.map(lambda p: (p.label, model.predict(p.features)))
# print testing_data.count()

test = {}

for i in labelsAndPreds.collect():
	test[i[1]] = i[0]

# Write predictions
o = DictWriter(open('pred.csv', 'w'), ['PhraseId', 'Sentiment'])
o.writeheader()
for ii in sorted(test):
	o.writerow({'PhraseId': ii, 'Sentiment': test[ii]})