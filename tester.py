import pickle as pickle
import pandas as pd
from build_model import TextClassifier, get_data

def test_code():

    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)

    X, y = get_data('data/articles.csv')

    print ("Accuracy:", model.score(X, y))
    print ("Predictions:", model.predict(X))

if __name__ == '__main__':
    test_code()
