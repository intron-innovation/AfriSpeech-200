import math
import numpy as np
import pandas as pd
from itertools import islice
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


def sort_dictionary(dict):

    keys = list(dict.keys())
    values = list(dict.values())
    sorted_value_index = np.argsort(values) # in asceding order 
    sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
    return sorted_dict


def compute_distances(points, centroids, k):
    distances = { accent:math.dist(points, list(cent)) for accent, cent in centroids.iterrows()}
    sorted_distances = sort_dictionary(distances)
    top_k = list(sorted_distances.keys())[:k]

    return top_k
##cosine sim
def compute_cosine_sim(points, centroids, k):
    distances = {}
    for accent, cent in centroids.iterrows():
        distance= cosine_similarity(np.array(points).reshape(1, -1), np.array(list(cent)).reshape(1, -1))[0][0]
        distances[accent] =distance
    sorted_distances = sort_dictionary(distances)
    top_k = list(sorted_distances.keys())[-k:]
    top_k.reverse()
    top_k = [t[0] for t in top_k]

    return top_k


