
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
from static.classifier.features import ClassifierFeatures

#Step 2: Import custom functions
from static.text.features import TextFeatures
from static.text.tokenizer import TextTokenizer

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, FeatureHasher
from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC, SVR
from sklearn.linear_model import BayesianRidge
#from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

from nltk.corpus import stopwords as sw
from _function.basic import printProbabilities


import pprint

pp = pprint.PrettyPrinter(indent=2)

#Step 3: Get all constants
con = Constants()

#Step 4: Get options and read all system arguments
options = Options('System parameters', con)

#Step 5: Read all custom arguments/options
#options.add(name='predict_languages', _type=str, _default='esdi', _help='specify which language you want to predict')

#Step 6: Parse arguments
options.parse()

#Use random seed
random.seed(options.args.random_seed)

#Custom function to read language from input

#Print system
printer = Printer('System')
printer.system(options.args_dict)

#Step 7: Create data with default arguments
data = Data(options.args.avoid_skewness, options.args.data_folder, options.args.predict_label, options.args.data_method)

#Step 8: Add all datasources and transform them to row(Y, X) format
#Custom, should be self-made!

#Step 8.1: Add the files or folders the data is preserved in (only if available)

file_name = 'conversion_product'
#data.file_train = 'conversion_chance.pickle'
data.file_train = file_name+'.pickle'
#data.file_train = 'conversion_path.pickle'

#Custom function

#Load data into a file
data.train = data.load(data.file_train, format='pickle')
counter = {}

for row in data.train:
  if row[0] not in counter:
    counter[row[0]] = 0
  counter[row[0]] += 1

new_train= []
for row in data.train:
  if counter[row[0]] > 5:
    new_train.append(row)
data.train = new_train

#Step 8.2: Formulate the preprocessing steps which have to be done
textPreprocessing = []
#Step 8.3: Transform the data to our desired format
data.transform(_type='YXrow', preprocessing=textPreprocessing) #> now we got X, Y and X_train, Y_train, X_development, Y_development and X_test

#Step 8.4: For training purposes, we can specify what our subset will look like (train_size, development_size, test_size)
#data.subset(500, 50, 50)
#Step 9: Specify the features to use, this part is merely for sklearn.
features = ClassifierFeatures()
features.add('pageviews', TfidfVectorizer(tokenizer=TextTokenizer.tokenized, lowercase=False, analyzer='word', ngram_range=(1,3), min_df=1), 'pageviews'),#, max_features=100000)),
# features.add('page_level_1', TfidfVectorizer(tokenizer=TextTokenizer.tokenized, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'page_level_1'),#, max_features=100000)),
# features.add('page_level_2', TfidfVectorizer(tokenizer=TextTokenizer.tokenized, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'page_level_2'),#, max_features=100000)),
# features.add('page_level_3', TfidfVectorizer(tokenizer=TextTokenizer.tokenized, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'page_level_3'),#, max_features=100000)),
# features.add('page_level_4', TfidfVectorizer(tokenizer=TextTokenizer.tokenized, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'page_level_4'),#, max_features=100000)),
#features.add('pageviews_conversion_rate_avg', StandardScaler(), 'pageviews_conversion_rate_avg'),
#Step 10: Specify the classifier you want to use (additionaly!)
new_classifier = SGDClassifier()

#these are for the conversion probability task
new_classifier = LogisticRegression()
#new_classifier = RandomForestClassifier()
# new_classifier = GradientBoostingClassifier()
#new_classifier =  SVC(kernel='linear', probability=True)

#new_classifier = LinearRegression()
#new_classifier = Ridge()

if options.args.print_details >= 1:
  printer.labelDistribution(data.Y_train, 'Training Set')

#Step 11: Run our system.
if len(data.labels) > 1: #otherwise, there is nothing to train
  classifier = run(options.args.k, options.args.method, data, features._list, printer, options.args.predict_method, new_classifier, options.args.print_details, options.args.show_fitting)

  classifier.Y_development_predicted_proba = classifier.classifier.predict_proba(classifier.X_test)
  joblib.dump(classifier.classifier, options.args.data_folder+file_name+'_model.pickle') 

  # for i, x in enumerate(classifier.X_test['pageviews']):
  #   if classifier.Y_development[i] == 1 and classifier.Y_development_predicted_proba[i][1] < 0.2:
  #     pp.pprint([classifier.X_development['pageviews'][i], classifier.Y_development[i], classifier.Y_development_predicted_proba[i]])
  #printProbabilities(classifier.Y_development, classifier.Y_development_predicted_proba)

  printer.duration()
else:
  print('The combination of the language <> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(args.predict_label))