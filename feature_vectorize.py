import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file

# Read CSV and organize columns in 2D array by 'Answer'
df = pd.read_csv('train.csv', usecols=['Phrase', 'Sentiment'])
df_t = pd.read_csv('test.csv', usecols=['Phrase', 'PhraseId'])

print 'Vectorizing tfidf bag of words ...'
vectorizer = TfidfVectorizer()
df_train = vectorizer.fit_transform(df.Phrase)
df_t_test = vectorizer.transform(df_t.Phrase)

dump_svmlight_file(df_train,df.Sentiment,'train_libsvm.txt')
dump_svmlight_file(df_t_test,df_t.PhraseId,'test_libsvm.txt')

# # Create SGDClassifier instance and train model
# print 'Training classifier ...'
# classifier = SGDClassifier(n_iter=2000)
# classifier.fit(df_train, df.Answer)
# predictions = classifier.predict(df_t_test)

# test = {}

# for i, pred in enumerate(predictions):
# 	test[df_t.Question_ID[i]] = pred

# # Write predictions
# o = DictWriter(open('pred_2.csv', 'w'), ['Question ID', 'Answer'])
# o.writeheader()
# for ii in sorted(test):
# 	o.writerow({'Question ID': ii, 'Answer': test[ii]})

# print 'Complete!'