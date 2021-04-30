import os
import sys
import requests

import numpy
import ast
from sklearn.feature_extraction.text import TfidVectorizer

r = requests.get('https://t4g-dl-api.herokuapp.com/api/datasets/proj2/')
#sys.stdout = open("data.txt", "w")
#print(r.text)
#sys.stdout.close()

data = ast.literal_eval(r.text)
print("Question ID: ", data['data'][1]['_id'])
print("Title words: ", data['data'][1]['title'])
print("Query words: ", data['data'][1]['query'])
print("Indicated label: ", data['data'][1]['label'])

