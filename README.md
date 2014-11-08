PySpark-Sentiment-Analysis
==========================

Reed Anderson, Anas Salamah 

About
===========

The goal of our project is to apply Natural Language Processing techniques in a cluster computing environment. We intend to classify movie review sentiment using Apache Spark’s MLlip, specifically focusing on Naive Bayes, and will benchmark our progress as we compete in the related Kaggle Competition (link). The competition requires the sentences be labelled as either: negative, somewhat negative, neutral, somewhat positive, or positive.
	
The training and test data needed for this project has already been supplied by Kaggle, and our main needed resource will be AWS to train the classifier. A stretch goal will be to build a simple web app, REST API, and web server, similar to OpenALPR, where a user could input a sentence and receive sentiment analysis as one of the five labels. Finally, we’ll present the results of our classifier in a paper, and with our position on the Kaggle leaderboard (although the final results of the competition will not be known until 28 Feb 2015). 
	
Potential challenges for our project might include understanding how to use the data structure of the training data (a sentiment treebank) as it relates to Spark MLlip LabeledPoint data structure, and how to correctly apply naive bayes for text classification, which we are both studying in Jordan Boyd-Grabers’ NLP course. That we’re not totally certain how to do these things is exactly why we would like to do this project.
