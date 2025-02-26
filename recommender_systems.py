import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class RecommenderSystem:
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Process and cache movie genres
        self.genre_to_movies = defaultdict(list)
        self.movie_to_genres = {}
        for _, movie in self.movies_df.iterrows():
            genres = movie['genres'].split('|')
            self.movie_to_genres[movie['movieId']] = genres
            for genre in genres:
                self.genre_to_movies[genre].append(movie['movieId'])
        
        # Process and cache user ratings
        self.movie_ratings = defaultdict(list)
        self.movie_avg_ratings = {}
        for _, rating in self.ratings_df.iterrows():
            self.movie_ratings[rating['movieId']].append(rating['rating'])
        
        for movie_id, ratings in self.movie_ratings.items():
            self.movie_avg_ratings[movie_id] = np.mean(ratings)
    
    def get_similar_movies(self, movie_id, n=5):
        """Content-based movie recommendation"""
        if movie_id not in self.movie_to_genres:
            return []
        
        target_genres = set(self.movie_to_genres[movie_id])
        similar_movies = defaultdict(float)
        
        # Calculate genre and rating based similarity
        for genre in target_genres:
            for similar_movie_id in self.genre_to_movies[genre]:
                if similar_movie_id != movie_id:
                    similar_genres = set(self.movie_to_genres[similar_movie_id])
                    
                    # Calculate Jaccard similarity
                    genre_similarity = len(target_genres.intersection(similar_genres)) / len(target_genres.union(similar_genres))
                    
                    # Calculate rating similarity
                    if similar_movie_id in self.movie_avg_ratings:
                        rating_score = self.movie_avg_ratings[similar_movie_id] / 5.0
                        
                        # Popularity factor
                        popularity = len(self.movie_ratings[similar_movie_id]) / len(self.ratings_df)
                        
                        # Combined score (genre: 0.4, rating: 0.4, popularity: 0.2)
                        similar_movies[similar_movie_id] = (
                            0.4 * genre_similarity + 
                            0.4 * rating_score + 
                            0.2 * popularity
                        )
        
        # Select top n movies
        top_movies = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_movies 