'''
This is a script to train a BERT-based classification model.
'''

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import transformers
from pylab import rcParams
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split



# Set the convas.
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# Set random seed.
RANDOM_SEED = 10
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Check availability of GPU or use CPU if unavailable.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data loading from multiple sources
df_app = pd.read_csv('app_reviews.csv', usecols = ['content', 'score'])
df_absa = pd.read_csv('bank_reviews.csv', usecols = ['review_content', 'review_rating'],nrows=16000)
df_hotels = pd.read_csv('tripadvisor_hotel_reviews.csv')
df_rain = pd.read_csv('rain.csv')
df_dotsure =  pd.read_csv('dotsure_reviews.csv')
df_discoverybank = pd.read_csv('discoverybank.csv')
df_discoveryinsure = pd.read_csv('discoveryinsure.csv')
df_african = pd.read_csv('africanbank.csv')
df_app = df_app.rename(columns = {'content': 'review_content', 'score': 'review_rating'})
df_hotels = df_hotels.rename(columns = {'Review': 'review_content', 'Rating': 'review_rating'})

# Combine all the dataframes to one big one with a variety of reviews.
df_reviews = pd.concat([df_hotels, df_absa, df_app, df_rain, df_dotsure, df_discoverybank,\
                            df_discoveryinsure, df_african]).reset_index()

df_reviews = df_reviews[['review_content', 'review_rating']]
print(df_reviews.tail())
print(df_reviews.shape)

# Set conditions for classification.
conditions = [
    (df_reviews['review_rating'] <= 2),
    (df_reviews['review_rating'] > 2) & (df_reviews['review_rating'] <= 3),
    (df_reviews['review_rating'] > 3) & (df_reviews['review_rating'] <= 4),
    (df_reviews['review_rating'] > 4)
    ]

# create a list of the values we want to assign for each condition.
values = ['negative', 'neutral', 'neutral', 'positive']

# create a new column and use np.select to assign values to it using our lists as arguments.
df_reviews['sentiment'] = np.select(conditions, values)
print(df_reviews.head())

# For typing convenience.
df = df_reviews
# Rename columns for convenience.
df = df.rename(columns = {'review_content':'statement',\
                'review_rating':'rating', 'sentiment':'review'})

# Check for duplicate reviews.
df.drop_duplicates(subset = ['statement'], keep = 'first', inplace = True)
print(df.info())

# Check class distribution visually.
plt.figure(figsize = (6, 4))
sns.countplot(x = df.review)
plt.show()

# Check the number of classes numerically.
df['review'].value_counts()

# Dataset is imbalanced. Make this a 3-class dataset to restore some balance.
train, eva = train_test_split(df, test_size = 0.2)

# Specify the base model that will be re-trained to suit out case.
model = ClassificationModel('bert', 'bert-base-cased', num_labels  = 3,\
    args = {'reprocess_input_data': True, 'overwrite_output_dir': True}, use_cuda = False)

# 0, 1, 2 : positive, negative, neutral.
def making_label(string):
    '''
    The function "making_label" will assign each sentiment to an integer.
    '''
    if string=='positive':
        return 0
    if string=='neutral':
        return 2
    return 1

# Sort out the training and evaluation labels.
train['label'] = train['review'].apply(making_label)
eva['label'] = eva['review'].apply(making_label)
print('\n Shape and size of training data: \n', train.shape)

# Training dataset.
train_df = pd.DataFrame({'text': train['statement'][:5000].replace(r'\n', ' ', regex=True),\
                        'label': train['label'][:5000]})
# Evaluation dataset.
eval_df = pd.DataFrame({'text': eva['statement'][-1000:].replace(r'\n', ' ', regex=True),\
                        'label': eva['label'][-1000:]})

# Train the model.
model.train_model(train_df)

# Evaluate the model.
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print('\n Result \n:', result)
print('\n Model outputs:\n', model_outputs)

# Create a container. It will hold the highest integer (0, 1, 2), i.e the classification.
lst = []
for arr in model_outputs:
    lst.append(np.argmax(arr))
true = eval_df['label'].tolist()
predicted = lst

# Check the metrics visually.
mat = sklearn.metrics.confusion_matrix(true, predicted)
print(mat)

df_cm = pd.DataFrame(mat, range(3), range(3))
plt.figure(figsize = (8, 4))
sns.heatmap(df_cm, annot = True)
plt.show()

# Analyse the metrics further.
sklearn.metrics.classification_report(true, predicted,\
                                        target_names = ['positive','neutral','negative'])

sklearn.metrics.accuracy_score(true,predicted)

def get_class(statement):
    '''
    This function will take as input a string or list of strings and give the classification.
    This is a test function.
    '''
    results = model.predict([statement])
    pos = np.where(results[1][0] == np.amax(results[1][0]))
    pos = int(pos[0])
    sentiment_dict = {0:'positive', 1:'negative', 2:'neutral'}
    print(sentiment_dict[pos])
    return sentiment_dict[pos]

# Sample test.
get_class('The app is fantastic')
get_class('The app is terrible')
get_class('The app is OK')

# Save model.
with open('sentiment_analysis.pkl', 'wb') as f:
    model = pickle.dump(f)
    f.close()
