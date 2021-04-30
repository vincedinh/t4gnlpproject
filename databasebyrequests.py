import os
import sys
import requests
import ast

import pickle
from io import BytesIO

TRAINING_SIZE = 16798

r = requests.get('https://t4g-dl-api.herokuapp.com/api/datasets/proj2/')
sys.stdout = open("data.txt", "w")
print(r.text)
sys.stdout.close()

full_data = ast.literal_eval(r.text)

