from sklearn.linear_model import SGDClassifier

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion

from _function.basic import metrics
from _class.printer import Printer

class SVM:
  X_train = []
  Y_train = []
  X_development = []
  Y_development = []
  X_test = []

  Y_predicted = []

  labels = []

  features = []

  def __init__(self, data, show_fitting):

    self.X_train = data.X_train
    self.Y_train = data.Y_train

    self.X_development = data.X_development
    self.Y_development = data.Y_development

    self.X_test = data.X_test

    self.labels = data.labels

    self.show_fitting =show_fitting

  def classify(self, features, classifier=None):
  
    feature_union = ('feats', FeatureUnion(
      features
    ))

    if classifier == None:
      classifier = SGDClassifier(loss='hinge', random_state=42, max_iter=50, tol=None)
      
    self.classifier = Pipeline([
      feature_union,
      ('classifier', classifier)
    ])

    self.printer = Printer('Model Fitting', self.show_fitting)
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



