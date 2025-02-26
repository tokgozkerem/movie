from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import gc
import requests
from recommender_systems import RecommenderSystem
from movie_analytics import MovieAnalytics
from advanced_analytics import AdvancedAnalytics
import logging
import json
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='movie_recommender.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TMDB API settings
TMDB_API_KEY = "2876fcf774c5a43d2971bacef9f3b68f"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Load links.csv to match MovieLens and TMDB IDs
logger.info("Loading links.csv file...")
links_df = pd.read_csv('links.csv')
links_dict = links_df.set_index('movieId')['tmdbId'].to_dict()

logger.info("Loading and preprocessing data...")

# Load and preprocess data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Process and cache movie genres
genre_to_movies = defaultdict(list)
movie_to_genres = {}
for _, movie in movies_df.iterrows():
    genres = movie['genres'].split('|')
    movie_to_genres[movie['movieId']] = genres
    for genre in genres:
        genre_to_movies[genre].append(movie['movieId'])

# Process and cache user ratings
movie_ratings = defaultdict(list)
movie_avg_ratings = {}
for _, rating in ratings_df.iterrows():
    movie_ratings[rating['movieId']].append(rating['rating'])

for movie_id, ratings in movie_ratings.items():
    movie_avg_ratings[movie_id] = np.mean(ratings)

logger.info("Data preprocessing completed!")

def get_movie_poster(movie_id):
    """Get movie poster URL from TMDB"""
    try:
        # Convert MovieLens ID to TMDB ID
        tmdb_id = links_dict.get(movie_id)
        if not tmdb_id:
            return None
            
        # Get movie information from TMDB
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{int(tmdb_id)}",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US"
            }
        )
        data = response.json()
        
        # Return poster URL
        if data.get("poster_path"):
            return f"{POSTER_BASE_URL}{data['poster_path']}"
    except Exception as e:
        logger.error(f"Poster error for movie {movie_id}: {str(e)}")
    return None

def get_similar_movies(movie_id, n=5):
    """Gelişmiş film önerisi sistemi"""
    if movie_id not in movie_to_genres:
        return []

    target_genres = set(movie_to_genres[movie_id])
    similar_movies = defaultdict(float)
    
    # Tür ve puan bazlı benzerlik hesaplama
    for genre in target_genres:
        for similar_movie_id in genre_to_movies[genre]:
            if similar_movie_id != movie_id:
                similar_genres = set(movie_to_genres[similar_movie_id])
                
                # Jaccard benzerliği hesapla
                genre_similarity = len(target_genres.intersection(similar_genres)) / len(target_genres.union(similar_genres))
                
                # Puan benzerliği hesapla
                if similar_movie_id in movie_avg_ratings:
                    rating_score = movie_avg_ratings[similar_movie_id] / 5.0
                    
                    # Popülerlik faktörü
                    popularity = len(movie_ratings[similar_movie_id]) / len(ratings_df)
                    
                    # Kombine skor (tür: 0.4, puan: 0.4, popülerlik: 0.2)
                    similar_movies[similar_movie_id] = (
                        0.4 * genre_similarity + 
                        0.4 * rating_score + 
                        0.2 * popularity
                    )

    # En iyi n film seç
    top_movies = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)[:n]
    
    recommendations = []
    for movie_id, score in top_movies:
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        poster_url = get_movie_poster(movie_id)
        recommendations.append({
            'id': int(movie_id),
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'similarity_score': f"{score:.2f}",
            'avg_rating': f"{movie_avg_ratings[movie_id]:.1f}",
            'poster_url': poster_url
        })
    
    return recommendations

# Initialize RecommenderSystem class
recommender = RecommenderSystem(ratings_df, movies_df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search_movies():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])
    
    matching_movies = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
    results = []
    
    for _, movie in matching_movies.head(10).iterrows():
        poster_url = get_movie_poster(movie['movieId'])
        results.append({
            'id': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'],
            'poster_url': poster_url
        })
    
    return jsonify(results)

@app.route('/recommend')
def recommend_movies():
    movie_id = request.args.get('movieId', type=int)
    if not movie_id:
        return jsonify({'error': 'Movie ID not specified'})
    
    try:
        recommended_movies = get_similar_movies(movie_id)
        
        selected_movie = None
        if movie_id:
            selected_movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            selected_movie = {
                'id': int(selected_movie['movieId']),
                'title': selected_movie['title'],
                'genres': selected_movie['genres']
            }
        
        gc.collect()
        return jsonify({
            'selected_movie': selected_movie,
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/evaluate')
def evaluate_systems():
    """Evaluate recommendation system performance"""
    try:
        metrics = recommender.evaluate_recommendations()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/analytics')
def analytics():
    """Data analysis page"""
    try:
        logger.info("Starting analytics page generation")
        
        # Initialize MovieAnalytics class
        analytics = MovieAnalytics(ratings_df, movies_df)
        
        # Basic statistics
        basic_stats = {
            'total_movies': len(movies_df),
            'total_ratings': len(ratings_df),
            'total_users': len(ratings_df['userId'].unique()),
            'avg_rating': float(ratings_df['rating'].mean()),
            'rating_distribution': ratings_df['rating'].value_counts().sort_index().to_dict()
        }
        
        logger.info("Basic statistics calculated")
        
        # Visualizations
        try:
            visualizations = analytics.generate_visualizations()
            logger.info("Visualizations created successfully")
        except Exception as viz_error:
            logger.error(f"Visualization error: {str(viz_error)}")
            visualizations = {}
        
        # Statistical tests
        try:
            statistical_results = analytics.statistical_tests()
            logger.info("Statistical tests completed")
        except Exception as stat_error:
            logger.error(f"Statistical test error: {str(stat_error)}")
            statistical_results = {'anova_results': {'f_statistic': 0.0, 'p_value': 1.0}}
        
        # Model metrics
        try:
            test_predictions = ratings_df['rating'].head(1000).values
            test_actual = ratings_df['rating'].tail(1000).values
            metrics = analytics.calculate_metrics(test_predictions, test_actual)
            logger.info("Model metrics calculated")
        except Exception as metric_error:
            logger.error(f"Metric calculation error: {str(metric_error)}")
            metrics = {'mae': 0.0, 'rmse': 0.0}
        
        # Combine all statistics
        stats = {
            **basic_stats,
            'genre_plot': visualizations.get('genre_plot'),
            'temporal_plot': visualizations.get('temporal_plot'),
            'user_behavior_plot': visualizations.get('user_behavior_plot'),
            'genre_correlation_plot': visualizations.get('genre_correlation_plot'),
            'genre_rating_trends': visualizations.get('genre_rating_trends'),
            'genre_count_trends': visualizations.get('genre_count_trends'),
            'segment_scatter': visualizations.get('segment_scatter'),
            'segment_radar': visualizations.get('segment_radar'),
            'segment_profiles': visualizations.get('segment_profiles'),
            'anova_results': {
                'f_statistic': float(statistical_results['anova_results']['f_statistic']),
                'p_value': float(statistical_results['anova_results']['p_value'])
            },
            'metrics': {
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse'])
            }
        }
        
        logger.info("All data prepared, rendering template")
        return render_template('analytics.html', stats=stats)
        
    except Exception as e:
        logger.error(f"Analytics general error: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/academic-dashboard')
def academic_dashboard():
    """Academic analysis dashboard page"""
    try:
        logger.info("Starting academic dashboard generation")
        
        # Initialize AdvancedAnalytics class
        advanced_analytics = AdvancedAnalytics(ratings_df, movies_df)
        
        # Generate comprehensive report
        report = advanced_analytics.generate_comprehensive_report()
        
        # Log report structure for debugging
        logger.info(f"Report type: {type(report)}")
        if isinstance(report, dict):
            logger.info(f"Report structure: {report.keys()}")
        else:
            logger.error(f"Report is not a dictionary: {report}")
            report = {
                'anomaly_analysis': {
                    'anomaly_stats': {
                        'total_users': 0,
                        'anomaly_count': 0,
                        'normal_count': 0,
                        'anomaly_percentage': 0.0
                    },
                    'anomaly_plot': None
                },
                'pattern_analysis': {
                    'pattern_network': None,
                    'top_rules': []
                },
                'clustering_analysis': {
                    'dbscan_plot': None,
                    'kmeans_plot': None,
                    'clustering_metrics': {
                        'kmeans_silhouette': 0.0,
                        'dbscan_clusters': 0,
                        'kmeans_clusters': 0
                    },
                    'cluster_distributions': {
                        'dbscan': {},
                        'kmeans': {}
                    }
                },
                'distribution_analysis': {
                    'distribution_plot': None,
                    'distribution_stats': {
                        'count': 0,
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'q1': 0.0,
                        'median': 0.0,
                        'q3': 0.0,
                        'max': 0.0
                    },
                    'normality_test': {
                        'p_value': 0.0,
                        'is_normal': False
                    }
                }
            }
        
        # Ensure all sections exist
        required_sections = ['anomaly_analysis', 'pattern_analysis', 'clustering_analysis', 'distribution_analysis']
        for section in required_sections:
            if section not in report:
                report[section] = {}
        
        return render_template('academic_dashboard.html', report=report)
        
    except Exception as e:
        logger.error(f"Academic dashboard error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return empty report structure in case of error
        empty_report = {
            'anomaly_analysis': {
                'anomaly_stats': {
                    'total_users': 0,
                    'anomaly_count': 0,
                    'normal_count': 0,
                    'anomaly_percentage': 0.0
                },
                'anomaly_plot': None
            },
            'pattern_analysis': {
                'pattern_network': None,
                'top_rules': []
            },
            'clustering_analysis': {
                'dbscan_plot': None,
                'kmeans_plot': None,
                'clustering_metrics': {
                    'kmeans_silhouette': 0.0,
                    'dbscan_clusters': 0,
                    'kmeans_clusters': 0
                },
                'cluster_distributions': {
                    'dbscan': {},
                    'kmeans': {}
                }
            },
            'distribution_analysis': {
                'distribution_plot': None,
                'distribution_stats': {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'q1': 0.0,
                    'median': 0.0,
                    'q3': 0.0,
                    'max': 0.0
                },
                'normality_test': {
                    'p_value': 0.0,
                    'is_normal': False
                }
            }
        }
        return render_template('academic_dashboard.html', report=empty_report)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 