import pickle
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

class testModel:
    def predictions(self, best_model, data):
        saved_model = pickle.load(open(f'{best_model}_model', 'rb'))
        preds = saved_model.predict(data)
        print(preds)
        
data = pd.read_csv('cleaned_test_dataset.csv').drop('Output', axis = 1)
testModel().predictions('RandomForestClassifier(max_depth=8, max_features=None, n_estimators=64)', data)
