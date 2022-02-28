'''
This script will scrap the relevent review data from 'HelloPeter'.
It scraps the reviews as well as the ratings in 'star' format.
'''

# Import the required libraries
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Scrap multiple pages.
# Cycle through pages using a 'for' loop.
# Specify how many you want.
# Do not use 'zero' indexing. Start at page 1.
# Page 0 and Page 1 are the same pages in web development.
# Dotsure has 1365 pages.
pages = []
LAST_PAGE = 4 # Avoid magic numbers. Set number of pages to scrap here.
REVIEWS_PER_PAGE = 11 # Avoid magic numbers. Set number of review per page.
for i in range(1, LAST_PAGE + 1):
    PAGE = 'https://api.hellopeter.com/api/consumer/business/whatsapp/reviews?page={}'.format(i)
    pages.append(PAGE)

# This runs until an error is encountered but still saves the scraped pages \
# up until the errorneous page.
jsondata_all_review_content = []
for current_page in pages:
    print(current_page)
    page = requests.get('{}'.format(current_page))
    jsondata=json.loads(page.text)
    jsondata_all_review_content.append(jsondata)

# Check how many pages were successfully scrapped and use that to avoid 'out-of-index' error.
SCRAPED_PAGES = len(jsondata_all_review_content)

# Gather both the reviews and ratings to one variable.
all_data = []
for i in range(SCRAPED_PAGES):
    data_dict = jsondata_all_review_content[i]['data']
    all_data.append(data_dict)

print('\n Total number of pages be scraped: \n', len(all_data))

# Get the ratings as a list.
ratings_only = []
for i in range(SCRAPED_PAGES):
    for j in range(REVIEWS_PER_PAGE):
        ratings_only.append(all_data[i][j]['review_rating'])
        #print(all_data[i][j]['review_rating'])
print('\n The number of ratings is:\n', len(ratings_only))

# Get the reviews as a list.

reviews_only = []
for i in range(SCRAPED_PAGES):
    for j in range(REVIEWS_PER_PAGE):
        reviews_only.append(all_data[i][j]['review_content'])
        #print(all_data[i][j]['review_content'])
print('\n The number of reviews is:\n', len(reviews_only))

# Convert and combine the two lists to dataframe.
dotsure_reviews = pd.DataFrame(np.column_stack([reviews_only, ratings_only]),\
                                columns=['review_content', 'review_rating'])
print(dotsure_reviews.tail())

# Check the dataset for classes.
# Naturally, the dataset is imbalanced.
plt.hist(dotsure_reviews['review_rating'])

# Save the dataset of 'reviews' and 'ratings' to a CSV file.

dotsure_reviews.to_csv('whatsapp_reviews.csv', sep = ',', index=False, encoding='utf-8')

# Reload if you want to check that the CSV is not corrupted.
#df = pd.read_csv("dotsure_reviews.csv")
#print(df)
