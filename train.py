import ast
import pickle
import numpy as np

from read_data import *

# with open("data.txt") as file:
#     database = file.read()

# database = ast.literal_eval(database)
# test_data, train_data = split_data(database)

# with open("train_data.pkl", "wb") as file:
#     pickle.dump(train_data, file)

# with open("test_data.pkl", "wb") as file:
#     pickle.dump(test_data, file)


with open("train_data.pkl", "rb") as file:
    train_data = pickle.load(file)

with open("categories.pkl", "rb") as file:
    categories = pickle.load(file)

with open("wordlist.pkl", "rb") as file:
    wordlist = pickle.load(file)

test_data, train_data = split_data
wordlist = build_word_list(train_data)
categories = build_category_vecs(wordlist, train_data)
print(cost(train_data, categories, wordlist))

with open("train_data.pkl", "rb") as file:
    train_data = pickle.load(file)

with open("categories.pkl", "rb") as file:
    categories = pickle.load(file)

with open("wordlist.pkl", "rb") as file:
    wordlist = pickle.load(file)

with open("test_data.pkl", "wb") as file:
    pickle.dump(test_d)
