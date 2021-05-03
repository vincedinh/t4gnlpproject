import ast
import pickle

import numpy as np


TRAINING_RATIO = 10 # One out of every TRAINING_RATIO data gets included in test data set, rest used for training


def split_data(dataset):
    # Split data
    test_data = []
    train_data = []

    for i, item in enumerate(dataset['data']):
        if i % TRAINING_RATIO == 0:
           test_data.append(item)
        else:
            train_data.append(item)
 
    return (test_data, train_data)


def build_word_vector(query, wordlist):

    vector = np.zeros(len(wordlist))
    for word in (query["title"] + query["query"]):
        vector[wordlist.index(word)] += 1

    return vector


def build_word_list(train_data):
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

    return words


def build_category_vecs(wordlist, train_data):  
    categories = {} # Wordlists for each tag
    for item in train_data:

        # Build array of new word frequencies
        words = build_word_vector(item, wordlist)
        
        # Update tag word frequencies
        tag = item["label"]
        if not tag in categories.keys():
            categories[tag] = words
        else:
            categories[tag] += words
    
    return categories


def predict(categories, word_vector):
    similarities = []
    for tag in categories.keys():
        similarity = np.dot(categories[tag], word_vector) / (np.linalg.norm(categories[tag]) * np.linalg.norm(word_vector))
        similarities.append((similarity, tag))
    similarities.sort()
    return similarities

def cost(queries, categories, wordlist):
    cost = 0
    for query in queries:
        prediction = predict(categories, build_word_vector(query, wordlist))
        correct = query["label"]
        for tag in prediction:
            if tag[1] == correct:
                cost += (1 - tag[0]) ** 2
        if numpy.isnan(cost):
            print(query)
    cost /= len(queries)
    return cost

