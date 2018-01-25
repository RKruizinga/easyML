
#Step 1: Import all mandatory functions
import random

#Step 1.1: Import all classes
from _class.options import Options
from _class.data import Data
from _class.constants import Constants
from _class.printer import Printer

#Step 1.2: Import all functions
from _function.basic import classifier, run
from _function.custom import writeResults, languages

#Step 1.3: Import classifier
from classifier.features import ClassifierFeatures

#Step 2: Import custom functions
from text.features import TextFeatures
from text.tokenizer import TextTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords as sw

#Step 3: Get all constants
con = Constants()

#Step 4: Get options and read all system arguments
options = Options('System parameters', con)

#Step 5: Read all custom arguments/options
options.add(name='predict_languages', _type=str, _default='english', _help='specify which language you want to predict')

#Step 6: Parse arguments
options.parse()

#Use random seed
random.seed(options.args.random_seed)

#Print system
printer = Printer('System')
printer.system(options.args_dict)

#Step 7: Create data with default arguments
data = Data(options.args.avoid_skewness, options.args.data_folder, options.args.predict_label, options.args.data_method

#Step 8: Add all datasources and transform them to row(Y, X) format
#Custom, should be self-made!

#Step 8.1: Add the files or folders the data is preserved in (only if available)
if options.args.predict_languages == 'english':
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

#Step 8.2: Formulate the preprocessing steps which have to be done
textPreprocessing = ['replaceTwitterInstagram', 'replaceTwitterURL', 'replaceSpecialCharacters', 'maxCharacterSequence']

#Step 8.3: Transform the data to our desired format
data.transform(_type='YXrow', preprocessing=textPreprocessing) #> now we got X, Y and X_train, Y_train, X_development, Y_development and X_test

#Step 8.4: For training purposes, we can specify what our subset will look like (train_size, development_size, test_size)
#data.subset(500, 50, 50)

#Step 9: Specify the features to use, this part is merely for sklearn.
features = ClassifierFeatures()
#features.add('wordCount', TextFeatures.wordCount())
features.add('word', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeTweet, lowercase=False, analyzer='word', stop_words=sw.words('english'), ngram_range=(1,20), min_df=1)),#, max_features=100000)),

#Step 10: Specify the classifier you want to use (additionaly!)
#new_classifier = LinearSVC()
new_classifier = None

if options.args.print_details >= 2:
  printer.labelDistribution(data.Y_train, 'Training Set')

#Step 11: Run our system.
if len(data.labels) > 1: #otherwise, there is nothing to train
  run(options.args.k, options.args.method, data, features._list, printer, new_classifier, options.args.print_details, options.args.show_fitting)

  printer.duration()
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))