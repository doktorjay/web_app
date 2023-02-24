"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import random

import pandas as pd
import numpy as np
from utils import movies

def recommend_random(k=3):
    return movies['title'].sample(k).to_list()



# collaborative filtering = look at ratings only!
def recommend_with_NMF(query, model, ratings, k=3):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model
    OUTPUT
    - a list of movieIds

    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    
    # construct a user vector
    data = list(query.values()) # ratings of new user
    row_ind = [0]*len(data) # single row of zeroes 
    col_ind = list(query.keys())  # movieId as columns
    # Initialize a sparse user-item rating matrix
    # user_vec = csr_matrix((data, (row_ind, col_ind)), shape = (1, ratings.movieId.max()))
    # more general: shape as number of features in model:
    user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, model.n_features_in_))
   
    # 2. scoring
    
    # calculate the score with the NMF model
    # user_vec -> encoding -> p_user_vec -> decoding -> user_vec_hat
    scores = model.inverse_transform(model.transform(user_vec))
    # convert to a pandas series
    scores = pd.Series(scores[0])
    
    # 3. ranking
    
    # filter out movies allready seen by the user
    # give a zero score to movies the user has allready seen
    scores[query.keys()] = 0
    # sort the scores from high to low
    scores = scores.sort_values(ascending = False)
    # return the top-k highst rated movie ids
    recommendations = list(scores.head(k).index)
    
    return recommendations



# collaborative filtering = look at ratings only!
def recommend_neighborhood(query, model, ratings, k = 10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    
    # construct a user vector
    data = list(query.values()) # ratings of new user
    row_ind = [0]*len(data) # single row of zeroes 
    col_ind = list(query.keys())  
    # Initialize a sparse user-item rating matrix
    user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, model.n_features_in_))
   
    # 2. scoring
    
    # find n neighbors
    userIds = model.kneighbors([user_vec], n_neighbors = 5, return_distance = False)[0]
    # calculate their average rating
    scores = ratings.set_index('userId').loc[userIds].groupby('movieId')['rating'].sum()
    
    # 3. ranking
    
    # filter out movies allready seen by the user
    # give a zero score to movies the user has already seen
    already_seen = scores.index.isin(query.keys())
    scores.loc[already_seen] = 0
    # return the top-k highst rated movie ids
    recommendations = list(scores.head(k).index)
    
    return recommendations



