import math
import ast
import numpy as np
import pandas as pd
from itertools import islice
from numpy.linalg import norm
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity


def sort_dictionary(dict):

    keys = list(dict.keys())
    values = list(dict.values())
    sorted_value_index = np.argsort(values) # in ascending order 
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

    return top_k

   
def distance_two_coords(coord1,coord2):
    # use geopy package
    # Note: might need to convert the coords from string. could use ast.literal_eval for that
    coord1, coord2 = ast.literal_eval(coord1), ast.literal_eval(coord2)
    return geodesic(coord1,coord2).miles

def compute_geo_proximity(accent,other_accents,accent_info_df,k):
    # get the country coord of accent
    # calculate distance between country coord in `other_accents` and `accent`. Save the result in another column
    # Sort by distance and take k accents.
    assert len(accent)==1
    accent = accent[0]
    accent_country_coord = accent_info_df.loc[accent].country_coord
    accents_df = accent_info_df.loc[other_accents]
    accents_df['geo_proximity'] = accents_df['country_coord'].apply(lambda x: distance_two_coords(accent_country_coord, x))
    accents_df_sorted = accents_df.sort_values('geo_proximity')

    return [accent] + accents_df_sorted.index.values.tolist()[:k]

