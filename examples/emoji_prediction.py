
import random

from options import Options
from data import Data
from constants import Constants

from function.printer import Printer

from function.basic import classifier
from function.custom import writeResults

from method.validation import KFoldValidation

from classifier.features import ClassifierFeatures

from text.features import TextFeatures
from text.tokenizer import TextTokenizer

## Import custom functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords as sw

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
#specify training, development and test files (if any)

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

#data.subset(500, 50, 50) #to make a subset of the data (train_size, development_size, test_size)

#only for sklearn: if wanted, you can edit your features here to specify features for all classifiers. Otherwise, you can just edit the classifier files.
features = ClassifierFeatures()
#features.add('wordCount', TextFeatures.wordCount())
features.add('word', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeTweet, lowercase=False, analyzer='word', stop_words=sw.words('english'), ngram_range=(1,20), min_df=1)),#, max_features=100000)),

#if you want, you can even specify your own classifier (such as SVC() with paramaters)
#new_classifier = LinearSVC()
new_classifier = None

printer.labelDistribution(data.Y_train, 'Training Set')

if len(data.labels) > 1: #otherwise, there is nothing to train
  if options.args.k > 1:
    kfold = KFoldValidation(options.args.k, options.args.method, data, features._list, new_classifier)
    kfold.printBasicEvaluation()

  else:
    classifier = classifier(options.args.method, data)
    classifier.classify(features._list, new_classifier)
    classifier.evaluate()
    classifier.printBasicEvaluation()
    classifier.printClassEvaluation()

    writeResults(options.args.predict_languages, classifier.Y_development, classifier.Y_development_predicted, 'development')
    printer.confusionMatrix(classifier.Y_development, classifier.Y_development_predicted, data.labels)
  printer.duration()
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))