import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

class MovieAnalytics:
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
    def genre_correlation_analysis(self):
        """Analyze correlations between movie genres"""
        # Convert movie genres to one-hot encoded matrix
        genres_list = []
        for genres in self.movies_df['genres'].str.split('|'):
            genres_list.extend(genres)
        unique_genres = list(set(genres_list))
        
        genre_matrix = np.zeros((len(self.movies_df), len(unique_genres)))
        for i, genres in enumerate(self.movies_df['genres'].str.split('|')):
            for genre in genres:
                genre_matrix[i, unique_genres.index(genre)] = 1
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(genre_matrix.T)
        
        # Create heatmap with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=unique_genres,
            y=unique_genres,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title={
                'text': 'Genre Correlation Analysis',
                'font': {'size': 24}
            },
            xaxis_title={
                'text': 'Genre',
                'font': {'size': 16}
            },
            yaxis_title={
                'text': 'Genre',
                'font': {'size': 16}
            },
            font=dict(family='Arial'),
            template='plotly_white'
        )
        
        return {
            'genre_correlation_plot': fig,
            'correlation_matrix': correlation_matrix.tolist(),
            'genres': unique_genres
        }
    
    def genre_popularity_trends(self):
        """Analyze genre popularity trends over time"""
        # Extract years from titles
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # Calculate yearly averages for each genre
        genre_trends = []
        for _, movie in self.movies_df.iterrows():
            year = movie['year']
            if pd.notna(year):
                genres = movie['genres'].split('|')
                movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie['movieId']]
                if not movie_ratings.empty:
                    avg_rating = movie_ratings['rating'].mean()
                    rating_count = len(movie_ratings)
                    for genre in genres:
                        genre_trends.append({
                            'year': int(year),
                            'genre': genre,
                            'avg_rating': avg_rating,
                            'rating_count': rating_count
                        })
        
        trend_df = pd.DataFrame(genre_trends)
        
        # Rating trends visualization
        fig_ratings = px.line(
            trend_df.groupby(['year', 'genre'])['avg_rating'].mean().reset_index(),
            x='year',
            y='avg_rating',
            color='genre',
            title='Average Ratings by Genre Over Time'
        )
        fig_ratings.update_layout(
            title={
                'text': 'Temporal Evolution of Genre Ratings (1995-2023)',
                'font': {'size': 24}
            },
            xaxis_title={
                'text': 'Year',
                'font': {'size': 16}
            },
            yaxis_title={
                'text': 'Average Rating',
                'font': {'size': 16}
            },
            font=dict(family='Arial'),
            template='plotly_white',
            legend_title_text='Genre'
        )
        
        # View count trends
        fig_counts = px.line(
            trend_df.groupby(['year', 'genre'])['rating_count'].sum().reset_index(),
            x='year',
            y='rating_count',
            color='genre',
            title='Genre Popularity Trends'
        )
        fig_counts.update_layout(
            title={
                'text': 'Genre Popularity Evolution (1995-2023)',
                'font': {'size': 24}
            },
            xaxis_title={
                'text': 'Year',
                'font': {'size': 16}
            },
            yaxis_title={
                'text': 'Number of Ratings',
                'font': {'size': 16}
            },
            font=dict(family='Arial'),
            template='plotly_white',
            legend_title_text='Genre'
        )
        
        return {
            'genre_rating_trends': fig_ratings,
            'genre_count_trends': fig_counts
        }
    
    def user_segmentation(self, n_clusters=4):
        """Perform user segmentation analysis"""
        # Calculate user features
        user_features = self.ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        }).fillna(0)
        
        user_features.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(user_features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Visualize results
        user_features['cluster'] = clusters
        
        # Scatter plot
        fig_scatter = px.scatter(
            user_features,
            x='rating_count',
            y='avg_rating',
            color='cluster',
            title='User Segments Distribution'
        )
        fig_scatter.update_layout(
            title={
                'text': 'User Segmentation Analysis',
                'font': {'size': 24}
            },
            xaxis_title={
                'text': 'Number of Ratings',
                'font': {'size': 16}
            },
            yaxis_title={
                'text': 'Average Rating',
                'font': {'size': 16}
            },
            font=dict(family='Arial'),
            template='plotly_white',
            legend_title_text='Segment'
        )
        
        # Segment profiles
        segment_profiles = user_features.groupby('cluster').mean()
        
        # Radar chart
        categories = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
        fig_radar = go.Figure()
        
        for cluster in range(n_clusters):
            values = segment_profiles.loc[cluster, categories].values
            values = (values - values.min()) / (values.max() - values.min())  # Min-max normalization
            values = np.append(values, values[0])  # Complete the circle
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=f'Segment {cluster}'
            ))
        
        fig_radar.update_layout(
            title={
                'text': 'User Segment Characteristics',
                'font': {'size': 24}
            },
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            font=dict(family='Arial'),
            template='plotly_white'
        )
        
        return {
            'segment_scatter': fig_scatter,
            'segment_radar': fig_radar,
            'segment_profiles': segment_profiles.to_dict('index')
        }
    
    def genre_success_analysis(self):
        """Analyze success metrics by genre"""
        # Calculate success metrics for each genre
        genre_metrics = []
        
        for _, movie in self.movies_df.iterrows():
            genres = movie['genres'].split('|')
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie['movieId']]
            
            if not movie_ratings.empty:
                avg_rating = movie_ratings['rating'].mean()
                rating_count = len(movie_ratings)
                
                for genre in genres:
                    genre_metrics.append({
                        'genre': genre,
                        'avg_rating': avg_rating,
                        'rating_count': rating_count
                    })
        
        genre_df = pd.DataFrame(genre_metrics)
        genre_summary = genre_df.groupby('genre').agg({
            'avg_rating': ['mean', 'std'],
            'rating_count': 'sum'
        }).round(2)
        
        genre_summary.index = genre_summary.index.tolist()
        return genre_summary
    
    def temporal_analysis(self):
        """Analyze temporal patterns in ratings"""
        # Calculate rating trends by date
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
        self.ratings_df['year'] = self.ratings_df['timestamp'].dt.year
        self.ratings_df['month'] = self.ratings_df['timestamp'].dt.month
        
        temporal_trends = self.ratings_df.groupby(['year', 'month'])['rating'].agg([
            'mean', 'count', 'std'
        ]).reset_index()
        
        return temporal_trends
    
    def user_behavior_analysis(self):
        """Analyze user rating behavior patterns"""
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        })
        
        user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
        return user_stats
    
    def generate_visualizations(self):
        """Generate all visualizations for the analysis"""
        visualizations = {}
        
        try:
            # Genre success analysis plot
            genre_stats = self.genre_success_analysis()
            fig_genre = go.Figure(data=[
                go.Bar(
                    name='Average Rating',
                    x=list(genre_stats.index),
                    y=genre_stats[('avg_rating', 'mean')].tolist(),
                    error_y=dict(
                        type='data',
                        array=genre_stats[('avg_rating', 'std')].tolist(),
                        visible=True
                    )
                )
            ])
            fig_genre.update_layout(
                title={
                    'text': 'Genre Performance Analysis',
                    'font': {'size': 24}
                },
                xaxis_title={
                    'text': 'Genre',
                    'font': {'size': 16}
                },
                yaxis_title={
                    'text': 'Average Rating',
                    'font': {'size': 16}
                },
                font=dict(family='Arial'),
                template='plotly_white'
            )
            visualizations['genre_plot'] = json.loads(fig_genre.to_json())
            print("Genre plot created successfully")
        except Exception as e:
            print(f"Genre plot error: {str(e)}")
        
        try:
            # Temporal trend plot
            temporal_data = self.temporal_analysis()
            fig_temporal = px.line(
                temporal_data,
                x='year',
                y='mean',
                title='Rating Trends Over Time'
            )
            fig_temporal.update_layout(
                title={
                    'text': 'Temporal Evolution of Ratings (1995-2023)',
                    'font': {'size': 24}
                },
                xaxis_title={
                    'text': 'Year',
                    'font': {'size': 16}
                },
                yaxis_title={
                    'text': 'Average Rating',
                    'font': {'size': 16}
                },
                font=dict(family='Arial'),
                template='plotly_white'
            )
            visualizations['temporal_plot'] = json.loads(fig_temporal.to_json())
            print("Temporal plot created successfully")
        except Exception as e:
            print(f"Temporal plot error: {str(e)}")
        
        try:
            # User behavior distribution
            user_stats = self.user_behavior_analysis()
            fig_user = px.scatter(
                user_stats.reset_index(),
                x='rating_count',
                y='avg_rating',
                title='User Rating Behavior Analysis'
            )
            fig_user.update_layout(
                title={
                    'text': 'User Rating Distribution Analysis',
                    'font': {'size': 24}
                },
                xaxis_title={
                    'text': 'Number of Ratings',
                    'font': {'size': 16}
                },
                yaxis_title={
                    'text': 'Average Rating',
                    'font': {'size': 16}
                },
                font=dict(family='Arial'),
                template='plotly_white'
            )
            visualizations['user_behavior_plot'] = json.loads(fig_user.to_json())
            print("User behavior plot created successfully")
        except Exception as e:
            print(f"User behavior plot error: {str(e)}")
        
        try:
            # Genre correlations
            genre_correlations = self.genre_correlation_analysis()
            visualizations['genre_correlation_plot'] = json.loads(genre_correlations['genre_correlation_plot'].to_json())
            print("Genre correlation plot created successfully")
        except Exception as e:
            print(f"Genre correlation plot error: {str(e)}")
        
        try:
            # Genre trends
            genre_trends = self.genre_popularity_trends()
            visualizations['genre_rating_trends'] = json.loads(genre_trends['genre_rating_trends'].to_json())
            visualizations['genre_count_trends'] = json.loads(genre_trends['genre_count_trends'].to_json())
            print("Genre trends plots created successfully")
        except Exception as e:
            print(f"Genre trends plot error: {str(e)}")
        
        try:
            # User segments
            user_segments = self.user_segmentation()
            visualizations['segment_scatter'] = json.loads(user_segments['segment_scatter'].to_json())
            visualizations['segment_radar'] = json.loads(user_segments['segment_radar'].to_json())
            visualizations['segment_profiles'] = user_segments['segment_profiles']
            print("User segment plots created successfully")
        except Exception as e:
            print(f"User segment plot error: {str(e)}")
        
        return visualizations
    
    def statistical_tests(self):
        """Perform statistical analysis"""
        # ANOVA test for genre differences
        genre_ratings = []
        genre_names = []
        
        for _, movie in self.movies_df.iterrows():
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie['movieId']]['rating']
            if not movie_ratings.empty:
                primary_genre = movie['genres'].split('|')[0]
                genre_ratings.append(movie_ratings.values)
                genre_names.extend([primary_genre] * len(movie_ratings))
        
        f_stat, p_value = stats.f_oneway(*[group for group in genre_ratings])
        
        return {
            'anova_results': {
                'f_statistic': f_stat,
                'p_value': p_value
            }
        }
    
    def calculate_metrics(self, predictions, actual):
        """Calculate model performance metrics"""
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        return {
            'mae': mae,
            'rmse': rmse
        } 