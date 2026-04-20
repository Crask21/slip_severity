
#import the five classifiers
import joblib
classifiers = {}
for finger in ['thumb', 'index', 'middle', 'ring', 'little']:
    classifiers[finger] = joblib.load(f'{finger}_classifier.pkl')

