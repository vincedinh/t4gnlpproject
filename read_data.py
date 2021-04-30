import ast
import pickle

import numpy as np


TRAINING_SIZE = 16798

# Read in raw database
with open("data.txt") as file:
    database = file.read()

database = ast.literal_eval(database)

def split_data():
    # Split data
    test_data = []
    train_data = []

    for i, item in enumerate(database['data']):
        if i < TRAINING_SIZE:
           train_data.append(item)
        else:
            test_data.append(item)
 
    # Write to file
    with open("train_data.pkl", "wb") as file:
        pickle.dump(train_data, file)

    with open("test_data.pkl", "wb") as file:
        pickle.dump(test_data, file)


def build_word_list():
    with open("train_data.pkl", "rb") as file:
        train_data = pickle.load(file)
    
    # Build word list from data
    words = []
    for item in train_data:
        for key in item.keys():
            if key == '_id':
                continue
            else:
                for word in item[key]:
                    if (word not in words):
                        words.append(word)
    words.sort()
    
    # Export to pickle
    with open("wordlist.pkl", "wb") as file:
        pickle.dump(words, file)

def build_category_vecs():
    with open("wordlist.pkl", "rb") as file:
        wordlist = pickle.load(file)
    
    with open("train_data.pkl", "rb") as file:
        train_data = pickle.load(file)
    
    categories = {} # Wordlists for each tag
    for item in train_data:

        tags = [] # Tags in the current query
        for tag in item["label"].split(): # Each string is its own label
            tags.append(tag)

        # Build array of new word frequencies
        words = np.zeros(len(wordlist))
        for word in item["title"] + item["query"]:
            words[wordlist.index(word)] += 1

        # Add array to each tag
        for tag in tags:
            if not tag in categories.keys():
                categories[tag] = np.zeros(len(wordlist))
            categories[tag] += words
    
    with open("categories.pkl", "wb") as file:
        pickle.dump(categories, file)

def test():
    with open("categories.pkl", "rb") as file:
        categories = pickle.load(file)
    for key in categories.keys():
        print(key)

if __name__ == '__main__':
    build_category_vecs()
    test()