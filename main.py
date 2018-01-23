
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
options.add(name=con.dataset['name'], _type=con.dataset['type'], _default=con.dataset['default'], _help=con.dataset['help'])
options.add(name=con.random_seed['name'], _type=con.random_seed['type'], _default=con.random_seed['default'], _help=con.random_seed['help'])
options.add(name=con.avoid_skewness['name'], _type=con.avoid_skewness['type'], _default=con.avoid_skewness['default'], _help=con.avoid_skewness['help'])
options.add(name=con.KFold['name'], _type=con.KFold['type'], _default=con.KFold['default'], _help=con.KFold['help'])

options.add(name=con.data_folder['name'], _type=con.data_folder['type'], _default=con.data_folder['default'], _help=con.data_folder['help'])

#specific arguments for each situation
options.add(name='predict_languages', _type=str, _default='english', _help='specify which language you want to predict')
options.add(name='predict_label', _type=str, _default='response variable', _help='specify which variable you want to predict')

options.parse()

### change for random seed
random.seed(options.args.random_seed)

printer = Printer('System')
printer.system(options.args_dict)

data = Data(options.args.avoid_skewness, options.args.data_folder)

### THIS PART IS CASE SPECIFIC
#specify training, development and test files

if options.args.predict_languages:
  data.file_train = 'eng-train.pickle'
  data.file_development = 'eng-trial.pickle'
  data.file_test = 'eng-test.pickle'
else: 
  data.file_train = 'es-train.pickle'
  data.file_development = 'es-trial.pickle'
  data.file_test = 'es-test.pickle'

data.train = data.load(data.file_train, format='pickle')
if data.file_development != '':
  data.development = data.load(data.file_development, format='pickle')
if data.file_test != '':
  data.test = data.load(data.file_test, format='pickle')

#specify the format from the loaded files to transform it to X and Y
textPreprocessing = ['replaceTwitterInstagram', 'replaceTwitterURL', 'replaceSpecialCharacters', 'maxCharacterSequence']
data.transform(_type='YXrow', preprocessing=textPreprocessing) #> now we got X, Y and X_train, Y_train, X_development, Y_development and X_test


#data.subset(50000, 5000) #to make a subset of the data (train_size, test_size)
printer.labelDistribution(data.Y_train, 'Training Set')

if len(data.labels) > 1: #otherwise, there is nothing to train

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