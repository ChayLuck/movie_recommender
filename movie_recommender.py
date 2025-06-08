# movie_recommender.py

import pandas as pd
import numpy as np
from tqdm import tqdm

# Parametreler
K = 5   # minimum ortak film sayısı
N = 10  # en benzer kullanıcı sayısı
m = 5   # önerilecek film sayısı

# 1️⃣ Verileri yükle
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# 2️⃣ Yardımcı fonksiyonlar

# Ortalama puanı hesapla
def mean_rating(user_ratings):
    return user_ratings['rating'].mean()

# Pearson Correlation Coefficient (PCC) hesapla
def pearson_correlation(u_ratings, v_ratings):
    # Ortak filmleri bul
    common_movies = set(u_ratings['movieId']).intersection(set(v_ratings['movieId']))
    
    if len(common_movies) < K:
        return None  # Yetersiz ortak film
    
    u_common = u_ratings[u_ratings['movieId'].isin(common_movies)].sort_values('movieId')
    v_common = v_ratings[v_ratings['movieId'].isin(common_movies)].sort_values('movieId')
    
    ru = u_common['rating'].values
    rv = v_common['rating'].values
    
    mu = ru.mean()
    mv = rv.mean()
    
    numerator = np.sum((ru - mu) * (rv - mv))
    denominator = np.sqrt(np.sum((ru - mu)**2)) * np.sqrt(np.sum((rv - mv)**2))
    
    if denominator == 0:
        return None
    
    return numerator / denominator

# Tahmini puan hesapla (pred(u, p))
def predict_rating(u_id, p_id, user_mean, similar_users):
    numerator = 0
    denominator = 0
    
    for v_id, rho_uv in similar_users:
        v_ratings = ratings[ratings['userId'] == v_id]
        v_mean = mean_rating(v_ratings)
        
        if p_id in v_ratings['movieId'].values:
            rvp = v_ratings[v_ratings['movieId'] == p_id]['rating'].values[0]
            numerator += rho_uv * (rvp - v_mean)
            denominator += abs(rho_uv)
    
    if denominator == 0:
        return user_mean
    
    return user_mean + numerator / denominator

# 3️⃣ Öneri algoritması
def recommend_movies_for_user(u_id):
    u_ratings = ratings[ratings['userId'] == u_id]
    user_mean = mean_rating(u_ratings)
    
    # Diğer kullanıcılarla benzerlik hesapla
    user_ids = ratings['userId'].unique()
    similarities = []
    
    print(f"Computing similarities for user {u_id}...")
    
    for v_id in tqdm(user_ids):
        if v_id == u_id:
            continue
        
        v_ratings = ratings[ratings['userId'] == v_id]
        rho = pearson_correlation(u_ratings, v_ratings)
        
        if rho is not None and rho > 0:
            similarities.append((v_id, rho))
    
    # En benzer N kullanıcıyı seç
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_users = similarities[:N]
    
    # Aday film listesi oluştur
    candidate_movies = set()
    for v_id, rho in top_similar_users:
        v_ratings = ratings[ratings['userId'] == v_id]
        v_mean = mean_rating(v_ratings)
        
        # Favori filmler (ortalamanın üstünde)
        fav_movies = v_ratings[v_ratings['rating'] > v_mean]['movieId'].values
        
        # Kullanıcının zaten izlediği filmleri çıkar
        watched_movies = set(u_ratings['movieId'])
        new_movies = set(fav_movies) - watched_movies
        
        candidate_movies.update(new_movies)
        
        if len(candidate_movies) >= m * 3:
            break
    
    # Aday filmleri tahmin et ve sırala
    movie_predictions = []
    
    print(f"\nPredicting ratings for candidate movies...")
    
    for p_id in tqdm(candidate_movies):
        pred = predict_rating(u_id, p_id, user_mean, top_similar_users)
        movie_predictions.append((p_id, pred))
    
    movie_predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_predictions[:m]
    
    # Sonuçları yazdır
    print(f"\nTop {m} recommended movies for user {u_id}:\n")
    for p_id, score in top_movies:
        movie_title = movies[movies['movieId'] == p_id]['title'].values[0]
        print(f"{movie_title} (predicted rating: {score:.2f})")

# 4️⃣ Kullanım

if __name__ == "__main__":
    target_user_id = int(input("Enter target user ID: "))
    recommend_movies_for_user(target_user_id)
