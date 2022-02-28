'''This script allows you to load the model and classify a review(s)
'''

# Import required libraries.
import pickle
import numpy as np

# Load model.
with open('sentiment_analysis.pkl', 'rb') as f:
    model = pickle.load(f)

def get_class(statement):
    '''
    This function will take as input a string or list of strings and give the classification.
    '''
    results = model.predict([statement])
    pos = np.where(results[1][0] == np.amax(results[1][0]))
    pos = int(pos[0])
    sentiment_dict = {0:'positive', 1:'negative', 2:'neutral'}
    print(sentiment_dict[pos])
    return sentiment_dict[pos]
    