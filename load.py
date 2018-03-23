  
import pickle
from sklearn.externals import joblib
model = joblib.load('conversion_chance_model.pickle')
print(model)

