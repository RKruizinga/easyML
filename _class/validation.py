from sklearn.model_selection import KFold
import numpy as np

from _class.printer import Printer
from _function.basic import classifier as selectClassifier
from _function.basic import avg

class KFoldValidation:
    accuracy = []
    precision = []
    recall = []
    f1score = []

    def __init__(self, k, method, data, features, new_classifier, print_details, show_fitting):
      self.k = k
      self.kf = KFold(n_splits=self.k)

      self.print_details = print_details
      self.show_fitting = show_fitting

      self.printer = Printer(str(self.k)+'-Fold validation')

      self.method = method
      self.data = data
      self.features = features
      self.new_classifier = new_classifier

      self.validation()
    def validation(self):
      i = 0
      for train_index, test_index in self.kf.split(self.data.X[:(self.data.amount_train+self.data.amount_development)]):
        i += 1

        if self.print_details >= 4:
          n_printer = Printer(str(self.k)+'-Fold, Run: '+str(i))
        X_train, X_development = list(np.array(self.data.X)[train_index]), list(np.array(self.data.X)[test_index])
        Y_train, Y_development = list(np.array(self.data.Y)[train_index]), list(np.array(self.data.Y)[test_index])
        self.data.initialize(X_train, Y_train, X_development, Y_development)

        classifier = selectClassifier(self.method, self.data, self.show_fitting)
        classifier.classify(self.features, self.new_classifier)
        classifier.evaluate()

        self.accuracy.append(classifier.accuracy)
        self.precision.append(classifier.precision)
        self.recall.append(classifier.recall)
        self.f1score.append(classifier.f1score)

        if self.print_details >= 5:
          classifier.printBasicEvaluation()
        
        if self.print_details >= 6:
          classifier.printClassEvaluation()

        if self.print_details >= 7:
          n_printer.confusionMatrix(classifier.Y_development, classifier.Y_development_predicted, self.data.labels)
        # writeResults(options.args.predict_languages, classifier.Y_development, classifier.Y_development_predicted, 'development')
        
        if self.print_details >= 4:
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
