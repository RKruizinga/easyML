from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from function.basic import metrics
from function.printer import Printer

from text.features import TextFeatures
from text.tokenizer import TextTokenizer

from nltk.corpus import stopwords as sw

class KNeighbors:
  X_train = []
  Y_train = []
  X_development = []
  Y_development = []
  X_test = []

  Y_predicted = []

  labels = []

  features = []

  def __init__(self, data):
    self.X_train = data.X_train
    self.Y_train = data.Y_train

    self.X_development = data.X_development
    self.Y_development = data.Y_development

    self.X_test = data.X_test

    self.labels = data.labels

  def classify(self, features, classifier=None):
    if len(features) < 1:
      feature_union = ('feats', FeatureUnion([
             #('wordCount', CustomFeatures.wordCount()),
             #('characterCount', CustomFeatures.characterCount()),
            #  ('userMentions', CustomFeatures.userMentions()),
            #  ('urlMentions', CustomFeatures.urlMentions()),
            #  ('instagramMentions', CustomFeatures.instagramMentions()),
             #('hashtagUse', CustomFeatures.hashtagUse()),
	 					 #('char', TfidfVectorizer(tokenizer=Tokenizer.tweetIdentity, lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
	 					 ('word', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeTweet, lowercase=False, analyzer='word', stop_words=sw.words('english'), ngram_range=(1,20), min_df=1)),#, max_features=100000)),
      ]))
    else:
      feature_union = ('feats', FeatureUnion(
        features
      ))

    if classifier == None:
      classifier = KNeighborsClassifier(n_neighbors=40)
      
    self.classifier = Pipeline([
      feature_union,
      ('classifier', classifier)
    ])

    self.printer = Printer('Model Fitting')
    self.classifier.fit(self.X_train, self.Y_train)  
    self.printer.duration()

  def evaluate(self):
    if self.X_development:
      self.Y_development_predicted = self.classifier.predict(self.X_development)
    if self.X_test:
      self.Y_test_predicted = self.classifier.predict(self.X_test)

    self.accuracy, self.precision, self.recall, self.f1score = metrics(self.Y_development, self.Y_development_predicted, self.labels)

  def printBasicEvaluation(self):    
    self.printer.evaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    self.printer.classEvaluation(self.Y_development, self.Y_development_predicted, self.labels)





