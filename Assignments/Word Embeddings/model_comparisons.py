import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import csv

model = 'regularized'

# Loading the embeddings
vec_file = open(f"embeddings/{model}/vecs.tsv")
read_vec = csv.reader(vec_file, delimiter="\t")
word_file = open(f"embeddings/{model}/meta.tsv")
read_word = csv.reader(word_file, delimiter = '\t')

word_to_vec = {}

for word, row in zip(read_word, read_vec):
    vec = np.array([float(r) for r in row])
    word_to_vec[str(word[0])] = vec


def cosine_similarity(u, v):
    dot = u.dot(v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot/(norm_u*norm_v)
    return cosine_similarity

def complete_analogy(a, b, c, word_to_vec):
    
    a, b, c = a.lower(), b.lower(), c.lower()
    e_a, e_b, e_c = word_to_vec[a], word_to_vec[b], word_to_vec[c]
    
    words = word_to_vec.keys()
    max_cosine_sim = -100
    best_word = None

    input_words_set = set([a, b, c])
    
    for w in words:        
        if w in input_words_set:
            continue
        cosine_sim = cosine_similarity(e_b-e_a,word_to_vec[w]-e_c)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
    
    return best_word

def concept_categorization(words):
    # Perform k means clustering on the list of words
    pass

def compactness_score(word, words):
    n = len(words)
    W = [w in words if w != word]
    
    score = 0

    for w1 in W:
        for w2 in [word in W if word!=w1]:
            score += cosine_similarity(word_to_vec[word1], word_to_vec[word2])

    score /= n*(n-1)

    return score 

def outlier_detection(words):
    scores = np.array([compactness_score(w, words) for w in words])
    
    index = np.argmin(scores)
    return words[index]


if __name__ == "__main__":
    woman = word_to_vec['woman']
    girl = word_to_vec['girl']
    fat = word_to_vec['fat']
    food = word_to_vec['food']
            
    print(cosine_similarity(woman, girl))
    print(cosine_similarity(fat, food))

    print(complete_analogy('comedy', 'fun', 'action', word_to_vec))