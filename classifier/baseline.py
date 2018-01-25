import numpy as np

import sklearn
from _function.basic import metrics, keyCounter
from _class.printer import Printer

class Baseline:

  X_train = []
  Y_train = []
  X_development = []
  Y_development = []
  X_test = []

  labels = []

  features = []

  def __init__(self, data, show_fitting):
    self.X_train = data.X_train
    self.Y_train = data.Y_train

    self.X_development = data.X_development
    self.Y_development = data.Y_development

    self.X_test = data.X_test

    self.labels = data.labels

    self.show_fitting = show_fitting

    self.classifier = Classifier()

  def classify(self, features, classifier=None):
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

class Classifier:
  def __init__(self):
    pass

  def fit(self, X, y):
    label_distribution = keyCounter(y)
    highest_amount = 0
    for label in label_distribution:
      if label_distribution[label] > highest_amount or highest_amount == 0:
        highest_amount = label_distribution[label]
        self.most_frequent_class = label
  
  def predict(self, X):
    Y_predicted = []
    for doc in X:
      Y_predicted.append(self.most_frequent_class) 
    return Y_predicted


