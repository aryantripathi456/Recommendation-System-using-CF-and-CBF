import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from scipy.sparse.linalg import svds
import pickle
import os

print("Loading data...")
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
links = pd.read_csv('data/ml-latest-small/links.csv')

# --- Content-Based Features ---
print("Processing content-based features...")
credits.columns = ['id','twtle','cast','crew']
movies = movies.merge(credits,on='id')

def convert(text):
    if pd.isna(text):
        return []
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert3(text):
    L = []
    counter = 0
    if pd.isna(text):
        return []
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

def fetch_director(text):
    if pd.isna(text):
        return []
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

movies['overview'] = movies['overview'].fillna('')
movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['id','title','tags']].copy()
# Store poster path related stuff or we will fetch dynamically.
# TMDB dataset doesn't have direct poster paths. We can fetch using TMDB API later in streamlt, or just not show images, or use Wikipedia.
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

# Vectorize tags
print("Vectorizing text data...")
cv = TfidfVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Save for Content-Based
with open('data/movies_metadata.pkl', 'wb') as f:
    pickle.dump(new_df, f)
with open('data/vector.pkl', 'wb') as f:
    pickle.dump(vector, f)

# --- Collaborative Filtering Features ---
print("Processing collaborative filtering features...")
links = links.dropna(subset=['tmdbId'])
links['tmdbId'] = links['tmdbId'].astype(int)

ratings = ratings.merge(links[['movieId', 'tmdbId']], on='movieId')

tmdb_ids_in_dataset = set(movies['id'])
ratings = ratings[ratings['tmdbId'].isin(tmdb_ids_in_dataset)]

user_item_matrix = ratings.pivot_table(index='userId', columns='tmdbId', values='rating').fillna(0)
matrix = user_item_matrix.to_numpy()

users_ratings_mean = np.mean(matrix, axis=1)
matrix_demeaned = matrix - users_ratings_mean.reshape(-1, 1)

print(f"Matrix shape for SVD: {matrix_demeaned.shape}")
U, sigma, Vt = svds(matrix_demeaned, k=min(matrix_demeaned.shape)-1 if min(matrix_demeaned.shape) <= 50 else 50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + users_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns)

user_indices = list(user_item_matrix.index)

with open('data/cf_preds.pkl', 'wb') as f:
    pickle.dump(preds_df, f)
with open('data/cf_user_indices.pkl', 'wb') as f:
    pickle.dump(user_indices, f)

print("Training completed and models saved.")
