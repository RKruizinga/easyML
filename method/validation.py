from sklearn.model_selection import KFold
import numpy as np

from function.printer import Printer
from function.basic import classifier as selectClassifier
from function.basic import avg

class KFoldValidation:
    accuracy = []
    precision = []
    recall = []
    f1score = []

    def __init__(self, k, method, data, features, new_classifier):
      self.k = k
      self.kf = KFold(n_splits=self.k)

      self.printer = Printer(str(self.k)+'-Fold validation')

      self.method = method
      self.data = data
      self.features = features
      self.new_classifier = new_classifier

      self.validation()
    def validation(self):
      i = 0
      for train_index, test_index in self.kf.split(self.data.X):
        i += 1
        n_printer = Printer(str(self.k)+'-Fold, Run: '+str(i))
        X_train, X_development = list(np.array(self.data.X)[train_index]), list(np.array(self.data.X)[test_index])
        Y_train, Y_development = list(np.array(self.data.Y)[train_index]), list(np.array(self.data.Y)[test_index])
        self.data.initialize(X_train, Y_train, X_development, Y_development)

        classifier = selectClassifier(self.method, self.data)
        classifier.classify(self.features, self.new_classifier)
        classifier.evaluate()

        self.accuracy.append(classifier.accuracy)
        self.precision.append(classifier.precision)
        self.recall.append(classifier.recall)
        self.f1score.append(classifier.f1score)

        # classifier.printBasicEvaluation()
        # classifier.printClassEvaluation()

        # writeResults(options.args.predict_languages, classifier.Y_development, classifier.Y_development_predicted, 'development')
        # printer.confusionMatrix(classifier.Y_development, classifier.Y_development_predicted, data.labels)
        
        n_printer.duration()
      
    def printBasicEvaluation(self):
      self.printer.evaluation(
        avg(self.accuracy),
        avg(self.precision),
        avg(self.recall),
        avg(self.f1score),
        str(self.k) + "-Fold Cross Validation Evaluation"
      )

      self.printer.duration()
