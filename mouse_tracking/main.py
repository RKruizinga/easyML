
from flask import Flask, request
import pickle
import sklearn
from sklearn.externals import joblib
from flask_cache import Cache
from flask_cors import CORS
import numpy 
import nltk
import json

class TextTokenizer:
	def tokenized(arg):
		return arg

# [start config]
app = Flask(__name__, static_url_path='')
cache = Cache(app,config={'CACHE_TYPE': 'simple'})
CORS(app)

@app.route('/user-intention',  methods=["GET"])
def user_intention():
	with open('test_model.pickle', 'rb') as curr_file:
		clf = joblib.load(curr_file)
	xy_section = request.args.getlist('xy_section')[0].replace('-', '/').split(',')
	xy_area = request.args.getlist('xy_area')[0].replace('-', '/').split(',')
	xy_element = request.args.getlist('xy_element')[0].replace('-', '/').split(',')

	x = {
		'xy_section': [xy_section],
		'xy_area': [xy_area],
		'xy_element': [xy_element]
	}
	#print([x])
	prediction = clf.predict(x)
	prediction_proba = clf.predict_proba(x)
	# print(y)

	interest_probabilities = {}

	for i, interest in enumerate(clf.classes_):
		interest_probabilities[interest] = round(prediction_proba[0][i], 2)

	response = json.dumps({ 'response': {
			'predicted_class': prediction[0],
			'prediction_probabilities': interest_probabilities
		}
	})
	return response
