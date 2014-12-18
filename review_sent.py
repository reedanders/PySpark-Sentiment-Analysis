import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from csv import DictReader, DictWriter

# Local classifier and csc-matrix vectorizer

# The most common class, Neutral , includes more than 50 percent of the instances.
# Accuracy will not be an informative performance measure for this problem, as a
# degenerate classifier that predicts only Neutral can obtain an accuracy near 0.5.
# Approximately one quarter of the reviews are positive or somewhat positive, and
# approximately one fifth of the reviews are negative or somewhat negative.

# df = pd.read_csv('train.tsv', header=0, delimiter='\t')
# print df['Sentiment'].describe()
# print df['Sentiment'].value_counts()/df['Sentiment'].count()

def main():
	
	pipeline = Pipeline([
		('vect', TfidfVectorizer(stop_words='english')),
		('clf', LogisticRegression())
		])
	
	parameters = {
		'vect__max_df': (0.25, 0.5),
		'vect__ngram_range': ((1,1), (1,2)),
		'vect__use_idf': (True, False),
		'clf__C': (0.1, 1, 10),
	}
	
	# Train classifier
	df = pd.read_csv('data/train.tsv', header=0, delimiter='\t')
	
	X, y = df['Phrase'], df['Sentiment'].as_matrix()
	# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
	grid_search.fit(X, y)

	# numpy export
	from tempfile import TemporaryFile
	outfile = TemporaryFile()
	np.savez_compressed(outfile, grid_search)

	# Test classifier
	d_test = pd.read_csv('data/test.tsv', header=0, delimiter='\t')
	testFeatures = d_test['Phrase'].as_matrix()

	print testFeatures
	print dir(testFeatures)	
	# predictions = grid_search.predict(testFeatures)

	# # Write predictions
	# test = {}
	# for ii in range(len(predictions)):
	# 	test[d_test['PhraseId'][ii]] = predictions[ii]

	# o = DictWriter(open('predictions.csv', 'w'), ['PhraseId', 'Sentiment'])
	# o.writeheader() 

	# for ii in sorted(test):
	# 	o.writerow({'PhraseId': ii, 'Sentiment': test[ii]})

if __name__ == '__main__':
	main()
