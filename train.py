import ast
import pickle
import numpy as np

from read_data import *

# with open("data.txt") as file:
#     database = file.read()

# database = ast.literal_eval(database)
# test_data, train_data = split_data(database)

with open("test_data.pkl", "rb") as file:
    test_data = pickle.load(file)

with open("categories.pkl", "rb") as file:
    categories = pickle.load(file)

with open("wordlist.pkl", "rb") as file:
    wordlist = pickle.load(file)

with open("weights.pkl", "rb") as file:
    weights = pickle.load(file)

weights = np.ones(weights.shape)
# test_data, train_data = split_data
# wordlist = build_word_list(train_data)
# categories = build_category_vecs(wordlist, train_data)

test_model(weights, test_data, categories, wordlist)

# training = gradient_descent(train_data, categories, wordlist, .1, 30)

# Jwith open("weights.pkl", "wb") as file:
#     pickle.dump(training[0], file)
