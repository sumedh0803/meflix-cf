# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:28:33 2020

@author: sumed
"""


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

dataset = pd.read_csv("ratings.csv")
moviesdataset = pd.read_csv("movies.csv")

users = dataset["userId"].unique()
movies = dataset["movieId"].unique()
genres = []
for i in range(moviesdataset.shape[0]):
    genres += moviesdataset.iloc[i,2].split("|")
genres = list(set(genres))

#getting initial ratings from user
user_watch_list = []
ratings_pred = pd.DataFrame(np.nan, index = ['0'], columns = movies)
print("Enter data in following format: movieId ratings(1-5). Type 'END' to stop")
while True:
    ip = input()
    if ip == "END" or ip == "end":
        break
    else:
        ip = ip.split(" ")
        mvid = int(ip[0])
        rating = float(ip[1])
        ratings_pred.iloc[0][mvid] = rating
        user_watch_list.append(mvid)

print("What genre of movies would you like to watch?")        
for g in range(len(genres)):
    print(g,". ",genres[g])
genre_code = int(input())

        
user_mean = ratings_pred.mean(axis = 1)[0]
ratings_pred = ratings_pred.sub(user_mean, axis=1)
ratings_pred = ratings_pred.fillna(0)

print("Analysing your ratings..\n")
#Creating the ratings matrix
ratingsmatrix = pd.DataFrame(columns = list(movies),index = list(users))
for i in range(dataset.shape[0]):
    movieid = dataset.iloc[i,1]
    userid = dataset.iloc[i,0]
    rating = dataset.iloc[i,2]
    ratingsmatrix[movieid][userid] = rating
    
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(ratingsmatrix, test_size=0.01, random_state = 2)

all_users_mean = ratingsmatrix.mean(axis = 1)
rmcentered = ratingsmatrix.sub(ratingsmatrix.mean(axis = 1), axis = 'index')
rmcentered = rmcentered.fillna(0)


print("Finding similar users...\n")
sim = {0:{}}
uid = 0 #may replace with userid of the user. Keep it 0 for now
for j in range(rmcentered.shape[0]):
    uid_1 = rmcentered.iloc[j,:].name
    cosine = cosine_similarity([list(ratings_pred.iloc[0,:]),list(rmcentered.iloc[j,:])])[0][1] 
    sim[uid][uid_1] = cosine
        
        
#rsme = 0        
#for s in sim:   
similarity = sim[0]
sorted_similarity = {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1],reverse = True)}
similar_users = []
i = 0
for j in sorted_similarity:
    if i < 100:
        similar_users.append(j)
        i += 1
    else:
        break
    
ratings_similar_users = pd.DataFrame(columns = list(movies))    
for i in similar_users:
    ratings_similar_users.loc[i] = rmcentered.loc[i]

for i in range(ratings_similar_users.shape[0]):
    idx = ratings_similar_users.iloc[i].name
    ratings_similar_users.loc[idx] = ratings_similar_users.loc[idx].add(all_users_mean[idx],axis = 'index')

print("Predicting your favourite movies...\n")
#to improve accuracy, lets try weighted mean
predicted_ratings = []
sum_of_weights = 0
for i in similar_users:
    sum_of_weights += sorted_similarity[i]
    
for i in movies:
    summ = 0
    for j in similar_users:
        summ += ratings_similar_users[i][j] * similarity[j]
    predicted_ratings.append(summ / sum_of_weights)


ratings_pred_mean = pd.Series(predicted_ratings, index = movies)
ratings_test_mean = pd.Series(ratings_pred.iloc[0], index = movies) #here, test is our user's ratings


print("suggestions for user: 0")
c = 0
ratings_pred_mean = ratings_pred_mean.sort_values()
for i in range(len(list(ratings_pred_mean.index))-1,-1,-1):
    if c < 15:
        mvid = list(ratings_pred_mean.index)[i]
        if mvid not in user_watch_list and genres[genre_code] in moviesdataset.loc[moviesdataset.movieId == mvid,'genres'].values[0]:
            print(moviesdataset.loc[moviesdataset.movieId == mvid,'title'].values[0])
            c += 1
    else:
        break
print("\n")



    


