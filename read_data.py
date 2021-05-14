import ast
import pickle

import numpy as np
from gensim.models import Doc2Vec

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
        try:
            vector[wordlist.index(word)] += 1
        except ValueError:
            pass
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


# returns a list of tuples cosine similarties for each query in ascending order from least likely to most likely category
def predict(categories, word_vector, weights):
    similarities = []
    word_vector = word_vector * weights
    for tag in categories.keys():
        similarity = np.dot(categories[tag], word_vector) / (np.linalg.norm(categories[tag]) * np.linalg.norm(word_vector))
        similarities.append((similarity, tag))
    return similarities

def cost(queries, categories, wordlist, weights, reg_weight):
    cost = 0
    num_queries = 0
    for query in queries:
        # Check for empty query
        if query["query"] == []:
            continue            
        prediction = predict(categories, build_word_vector(query, wordlist), weights)
        # Update cost
        for category in prediction:
            if category[1] == query["label"]:
                cost += (category[0] - 1) ** 2
            else:
                cost += (category[0] + 1) ** 2
            cost += reg_weight + np.sum(np.square(weights))
        num_queries += 1
    return cost / (2 * num_queries)

def gradient_descent(train_data, categories, wordlist, learn_rate, num_iters, reg_weight, weights):
    cost_history = []
    word_vectors = [build_word_vector(query, wordlist) for query in train_data]
    for i in range(num_iters):
        print(str(i*100/num_iters) + "% complete")
        new_weights = np.zeros(weights.shape)
        for j in range(weights.shape[0]):
            gradient = 0
            predictions = [predict(categories, word_vector, weights) if np.count_nonzero(word_vector) != 0 else [] for word_vector in word_vectors]
            for k, word_vector in enumerate(word_vectors):
                if (word_vector[j] == 0):
                    continue
                for category in predictions[k]:
                    if category == train_data[i]["label"]:
                        gradient += (category[0] - 1) * word_vector[j]
                    else:
                        gradient += (category[0] + 1) * word_vector[j]
                    gradient += 2 * reg_weight * weights[j]
            gradient /= (len(train_data) * len(categories))
            new_weights[j] = weights[j] - (learn_rate * gradient)
        weights = new_weights.copy()
        curr_cost = cost(train_data, categories, wordlist, weights)
        cost_history.append(curr_cost)
        print(curr_cost)
    return weights, cost_history

def test_model(weights, test_data, categories, wordlist):
    num_correct = 0
    total = 0
    word_vectors = [build_word_vector(query, wordlist) for query in test_data]
    for i in range(len(test_data)):
        if word_vectors[i] == []:
            continue
        if max(predict(categories, word_vectors[i], weights))[1] == test_data[i]["label"]:
            num_correct += 1
        total += 1
    print("Accuracy:", num_correct/total)

