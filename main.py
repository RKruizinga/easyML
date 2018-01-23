
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from options import Options
from data import Data
from constants import Constants

from functions.printer import Printer
from functions.basicFunctions import classifier
from functions.customFunctions import writeResults, getLanguages

### Get all constants
con = Constants()

# Read arguments
options = Options(description='system parameters')
options.add(name=con.title['name'], _type=con.title['type'], _default=con.title['default'], _help=con.title['help'])
options.add(name=con.method['name'], _type=con.method['type'], _default=con.method['default'], _help=con.method['help'])
options.add(name=con.random_seed['name'], _type=con.random_seed['type'], _default=con.random_seed['default'], _help=con.random_seed['help'])
options.add(name=con.data_method['name'], _type=con.data_method['type'], _default=con.data_method['default'], _help=con.data_method['help'])
options.add(name=con.predict_label['name'], _type=con.predict_label['type'], _default=con.predict_label['default'], _help=con.predict_label['help'])
options.add(name=con.predict_languages['name'], _type=con.predict_languages['type'], _default=con.predict_languages['default'], _help=con.predict_languages['help'])
options.add(name=con.avoid_skewness['name'], _type=con.avoid_skewness['type'], _default=con.avoid_skewness['default'], _help=con.avoid_skewness['help'])
options.add(name=con.KFold['name'], _type=con.KFold['type'], _default=con.KFold['default'], _help=con.KFold['help'])

options.parse()

### change for random seed
random.seed(options.args.random_seed)

options.args.predict_languages = getLanguages(options.args.predict_languages)
printer = Printer('System')
printer.system(options.args_dict)
data = Data(options.args.predict_languages, options.args.avoid_skewness)

#data.subset(50000, 5000) #to make a subset of the data (train_size, test_size)
printer.labelDistribution(data.Y_train, 'Training Set')

if len(labels) > 1: #otherwise, there is nothing to train

  classifier = classifier(options.args.method, data)

  classifier.classify()
  classifier.evaluate()
  classifier.printBasicEvaluation()
  classifier.printClassEvaluation()

  #writeResults(predict_languages, classifier.Y_dev)
  #BasicFunctions.writeConfusionMatrix(classifier.Y_test, classifier.Y_predicted)

  printer.duration()
  
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))