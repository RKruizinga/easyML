
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

from nltk.corpus import stopwords as sw

#Step 3: Get all constants
con = Constants()

#Step 4: Get options and read all system arguments
options = Options('System parameters', con)

#Step 5: Read all custom arguments/options
options.add(name='predict_languages', _type=str, _default='esdi', _help='specify which language you want to predict')

#Step 6: Parse arguments
options.parse()

#Use random seed
random.seed(options.args.random_seed)

#Custom function to read language from input
options.args.predict_languages = languages(options.args.predict_languages)

#Print system
printer = Printer('System')
printer.system(options.args_dict)

#Step 7: Create data with default arguments
data = Data(options.args.avoid_skewness, options.args.data_folder, options.args.predict_label, options.args.data_method)

#Step 8: Add all datasources and transform them to row(Y, X) format
#Custom, should be self-made!

#Step 8.1: Add the files or folders the data is preserved in (only if available)
data.file_train = 'impression_data.csv'

#Custom function
data.languages = options.args.predict_languages

#Load data into a file
data.train = data.load(data.file_train, format='complex_file')

#Step 8.2: Formulate the preprocessing steps which have to be done
textPreprocessing = []

#Step 8.3: Transform the data to our desired format
data.transform(_type='YXrow', preprocessing=textPreprocessing) #> now we got X, Y and X_train, Y_train, X_development, Y_development and X_test

#Step 8.4: For training purposes, we can specify what our subset will look like (train_size, development_size, test_size)
#data.subset(500, 50, 50)

#Step 9: Specify the features to use, this part is merely for sklearn.
features = ClassifierFeatures()
features.add('headline', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'headline'),#, max_features=100000)),
# features.add('headline_words', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'headline'),#, max_features=100000)),
# features.add('headline1_words', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'headline1'),#, max_features=100000)),
# features.add('headline2_words', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1,1), min_df=1), 'headline2'),#, max_features=100000)),
# features.add('keyword_words', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1, 1), min_df=1), 'keyword'),#, max_features=100000)),
# #features.add('term_words', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1, 2), min_df=1), 'term'),#, max_features=100000)),
# features.add('display_url', TfidfVectorizer(tokenizer=TextTokenizer.tokenizeText, lowercase=False, analyzer='word', ngram_range=(1, 1), min_df=1), 'display_url'),#, max_features=100000)),
# features.add('quality_score', StandardScaler(), 'quality_score'),
# features.add('position', StandardScaler(), 'position'),
# features.add('cost', StandardScaler(), 'cost'),
#features.add('month', OneHotEncoder(), 'month'),

#features.add('device', OneHotEncoder(), 'device'),

#Step 10: Specify the classifier you want to use (additionaly!)
new_classifier = LogisticRegression()
#new_classifier = SVR(kernel='linear')
#new_classifier = LinearRegression(normalize=True)
#new_classifier = Ridge()

if options.args.print_details >= 2:
  printer.labelDistribution(data.Y_train, 'Training Set')

#Step 11: Run our system.
if len(data.labels) > 1: #otherwise, there is nothing to train
  clf = run(options.args.k, options.args.method, data, features._list, printer, options.args.predict_method, new_classifier, options.args.print_details, options.args.show_fitting)

  coefs = clf.classifier.named_steps['classifier'].coef_
  feature_names = clf.classifier.named_steps['feats'].get_params()['transformer_list'][0][1].named_steps['feature'].get_feature_names()

  coef_dict = {}
  
  for coef, feat in zip(coefs,feature_names):
    coef_dict[feat] = coef
  print(coef_dict)

  # print(coef_dict)
  
  # for word in coef_dict:
  #   print(len(coef_dict[word]))
  printer.duration()
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))