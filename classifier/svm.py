from sklearn.linear_model import SGDClassifier

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion

from _function.basic import classificationMetrics
from _function.basic import regressionMetrics
from _function.basic import printProbabilities

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

  def __init__(self, data, predict_method, show_fitting):

    self.X_train = data.X_train
    self.Y_train = data.Y_train

    self.X_development = data.X_development
    self.Y_development = data.Y_development

    self.X_test = data.X_test

    self.labels = data.labels

    self.predict_method = predict_method
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
      #self.Y_development_predicted_proba = self.classifier.predict_proba(self.X_development)
    if self.X_test:
      self.Y_test_predicted = self.classifier.predict(self.X_test)

      #self.Y_test_predicted_proba = self.classifier.predict_proba(self.X_test)

    if self.predict_method == 'classification':
      self.accuracy, self.precision, self.recall, self.f1score = classificationMetrics(self.Y_development, self.Y_development_predicted, self.labels)
      
    elif self.predict_method == 'regression':
      self.mean_abs_err, self.mean_squ_err, self.r2score, self.kl_divergence = regressionMetrics(self.Y_development, self.Y_development_predicted, self.labels)

  def printBasicEvaluation(self):
    if self.predict_method == 'classification':
      self.printer.evaluation(self.accuracy, self.precision, self.recall, self.f1score, "Classification Evaluation")
    elif self.predict_method == 'regression':
      self.printer.regressionEvaluation(self.mean_abs_err, self.mean_squ_err, self.r2score, self.kl_divergence, "Regression Evaluation")

    #printProbabilities(self.Y_development, self.Y_test_predicted_proba)

  def printClassEvaluation(self):
    self.printer.classEvaluation(self.Y_development, self.Y_development_predicted, self.labels)



