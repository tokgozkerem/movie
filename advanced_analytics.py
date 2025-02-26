import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import networkx as nx
from sklearn.metrics import silhouette_score
import warnings
import gc
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.scaler = StandardScaler()
        
    def detect_rating_anomalies(self):
        """
        Detects anomalous rating behaviors.
        Memory-efficient implementation.
        """
        try:
            print("Starting anomaly detection...")
            
            # Limit sample size
            user_counts = self.ratings_df['userId'].value_counts()
            print(f"Total unique users: {len(user_counts)}")
            
            active_users = user_counts[user_counts >= 20].index
            print(f"Active users (>=20 ratings): {len(active_users)}")
            
            filtered_ratings = self.ratings_df[self.ratings_df['userId'].isin(active_users)]
            print(f"Filtered ratings shape: {filtered_ratings.shape}")
            
            # Calculate user features
            user_features = filtered_ratings.groupby('userId').agg({
                'rating': ['count', 'mean', 'std'],
                'movieId': 'nunique'
            }).fillna(0)
            
            user_features.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
            print(f"User features shape: {user_features.shape}")
            print(f"User features columns: {user_features.columns}")
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(user_features)
            print(f"Scaled features shape: {features_scaled.shape}")
            
            # Detect anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(features_scaled)
            
            # Convert anomalies to boolean for better visualization
            user_features['is_anomaly'] = anomalies == -1
            
            # Count anomalies
            anomaly_count = sum(anomalies == -1)
            normal_count = sum(anomalies == 1)
            print(f"Anomaly count: {anomaly_count}")
            print(f"Normal count: {normal_count}")
            
            # Create visualization
            fig = go.Figure()
            
            # Add normal points
            normal_data = user_features[~user_features['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=normal_data['rating_count'],
                y=normal_data['avg_rating'],
                mode='markers',
                name='Normal',
                marker=dict(
                    color='blue',
                    size=8,
                    opacity=0.6
                ),
                hovertemplate=
                "<b>User</b>: %{customdata}<br>" +
                "<b>Ratings</b>: %{x}<br>" +
                "<b>Average</b>: %{y:.2f}<br>" +
                "<b>Std Dev</b>: %{text:.2f}<br>" +
                "<extra></extra>",
                customdata=normal_data.index,
                text=normal_data['rating_std']
            ))
            
            # Add anomaly points
            anomaly_data = user_features[user_features['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=anomaly_data['rating_count'],
                y=anomaly_data['avg_rating'],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color='red',
                    size=10,
                    opacity=0.7,
                    symbol='x'
                ),
                hovertemplate=
                "<b>User</b>: %{customdata}<br>" +
                "<b>Ratings</b>: %{x}<br>" +
                "<b>Average</b>: %{y:.2f}<br>" +
                "<b>Std Dev</b>: %{text:.2f}<br>" +
                "<extra></extra>",
                customdata=anomaly_data.index,
                text=anomaly_data['rating_std']
            ))
            
            fig.update_layout(
                title={
                    'text': 'User Rating Behavior Analysis',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis_title={
                    'text': 'Number of Ratings',
                    'font': dict(size=16)
                },
                yaxis_title={
                    'text': 'Average Rating',
                    'font': dict(size=16)
                },
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                ),
                hovermode='closest',
                template='plotly_white',
                width=800,
                height=600
            )
            
            # Create result dictionary
            result = {
                'anomaly_plot': fig.to_dict(),
                'anomaly_stats': {
                    'total_users': int(len(user_features)),
                    'anomaly_count': int(anomaly_count),
                    'normal_count': int(normal_count),
                    'anomaly_percentage': float(round(anomaly_count / len(user_features) * 100, 2))
                }
            }
            
            print("Anomaly detection completed successfully")
            print(f"Result keys: {result.keys()}")
            print(f"Anomaly stats: {result['anomaly_stats']}")
            
            return result
            
        except Exception as e:
            print(f"Error in detect_rating_anomalies: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'anomaly_plot': None,
                'anomaly_stats': {
                    'total_users': 0,
                    'anomaly_count': 0,
                    'normal_count': 0,
                    'anomaly_percentage': 0.0
                },
                'error': str(e)
            }
    
    def extract_movie_patterns(self):
        """
        Extracts movie watching patterns using Apriori algorithm.
        Memory-efficient implementation.
        """
        try:
            # Select popular movies (en az 50 değerlendirme alan filmler)
            movie_counts = self.ratings_df['movieId'].value_counts()
            popular_movies = movie_counts[movie_counts >= 50].index[:100]  # Top 100 popular movies
            
            filtered_ratings = self.ratings_df[self.ratings_df['movieId'].isin(popular_movies)]
            
            # Select active users (en az 10 film değerlendiren kullanıcılar)
            user_counts = filtered_ratings['userId'].value_counts()
            active_users = user_counts[user_counts >= 10].index[:1000]  # Top 1000 active users
            
            filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(active_users)]
            
            print(f"Filtered data shape: {filtered_ratings.shape}")
            
            # Create user-movie matrix (4 ve üzeri puanları pozitif olarak kabul et)
            user_movie_matrix = pd.pivot_table(
                filtered_ratings,
                index='userId',
                columns='movieId',
                values='rating',
                fill_value=0
            )
            
            print(f"Matrix shape before binary conversion: {user_movie_matrix.shape}")
            
            # Convert to binary matrix (4 ve üzeri puanları 1, diğerlerini 0 yap)
            binary_matrix = (user_movie_matrix >= 4).astype(np.int8)
            
            # Clear memory
            del user_movie_matrix
            gc.collect()
            
            print(f"Binary matrix shape: {binary_matrix.shape}")
            
            # Apply Apriori algorithm with lower support threshold
            frequent_itemsets = apriori(binary_matrix, 
                                      min_support=0.1,  # Daha düşük support değeri
                                      use_colnames=True,
                                      max_len=3)  # En fazla 3'lü kombinasyonlar
            
            if len(frequent_itemsets) == 0:
                print("No frequent itemsets found.")
                return {
                    'pattern_network': None,
                    'top_rules': [],
                    'error': 'No frequent itemsets found with current parameters.'
                }
            
            # Generate rules with lower thresholds
            rules = association_rules(frequent_itemsets, 
                                    metric="lift",
                                    min_threshold=1.2)  # Daha düşük lift threshold
            
            # Clear memory
            del binary_matrix
            gc.collect()
            
            if len(rules) == 0:
                print("No association rules found.")
                return {
                    'pattern_network': None,
                    'top_rules': [],
                    'error': 'No association rules found with current parameters.'
                }
            
            # Convert numpy values to Python types
            for col in rules.select_dtypes(include=[np.number]).columns:
                rules[col] = rules[col].astype(float)
            
            # Add movie titles
            movie_id_to_title = self.movies_df.set_index('movieId')['title'].to_dict()
            
            rules['antecedents_names'] = rules['antecedents'].apply(
                lambda x: [movie_id_to_title.get(i, str(i)) for i in x]
            )
            rules['consequents_names'] = rules['consequents'].apply(
                lambda x: [movie_id_to_title.get(i, str(i)) for i in x]
            )
            
            # Filter rules by confidence and lift
            rules = rules[
                (rules['confidence'] >= 0.5) &  # En az %50 confidence
                (rules['lift'] >= 1.2)  # En az 1.2 lift
            ]
            
            # Sort by multiple criteria
            rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
            
            # Select diverse top rules (farklı filmler içeren kuralları seç)
            top_rules = []
            seen_movies = set()
            
            for _, rule in rules.iterrows():
                antecedents = set(rule['antecedents_names'])
                consequents = set(rule['consequents_names'])
                
                # Eğer bu kural yeni filmler içeriyorsa veya çok yüksek lift değerine sahipse
                if len(seen_movies.intersection(antecedents.union(consequents))) < 2 or rule['lift'] > 2:
                    seen_movies.update(antecedents)
                    seen_movies.update(consequents)
                    
                    top_rules.append({
                        'antecedents': [str(x) for x in rule['antecedents_names']],
                        'consequents': [str(x) for x in rule['consequents_names']],
                        'lift': float(rule['lift']),
                        'confidence': float(rule['confidence'])
                    })
                    
                    if len(top_rules) >= 15:  # En fazla 15 kural
                        break
            
            # Create network visualization
            G = nx.Graph()
            
            # Add edges for top rules
            for rule in top_rules:
                for ant in rule['antecedents']:
                    for cons in rule['consequents']:
                        G.add_edge(ant, cons, weight=rule['lift'])
            
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            edge_weights = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([float(x0), float(x1), None])
                edge_y.extend([float(y0), float(y1), None])
                edge_weights.extend([edge[2]['weight'], edge[2]['weight'], None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                mode='lines',
                text=[f'Lift: {w:.2f}' if w is not None else '' for w in edge_weights]
            )

            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(float(x))
                node_y.append(float(y))
                node_text.append(str(node))

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=15,
                    line_width=2,
                    color='#1f77b4'
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                             title={
                                 'text': 'Movie Association Network',
                                 'y': 0.95,
                                 'x': 0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top',
                                 'font': dict(size=24)
                             },
                             showlegend=False,
                             hovermode='closest',
                             margin=dict(b=20,l=5,r=5,t=40),
                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             width=800,
                             height=600
                         ))
            
            return {
                'pattern_network': fig.to_dict(),
                'top_rules': top_rules
            }
            
        except Exception as e:
            print(f"Error in extract_movie_patterns: {str(e)}")
            return {
                'pattern_network': None,
                'top_rules': [],
                'error': str(e)
            }
    
    def perform_advanced_clustering(self):
        """
        Performs advanced clustering analysis.
        Memory-efficient implementation.
        """
        try:
            # Limit sample size
            user_counts = self.ratings_df['userId'].value_counts()
            active_users = user_counts[user_counts >= 20].index
            
            filtered_ratings = self.ratings_df[self.ratings_df['userId'].isin(active_users)]
            
            # Calculate user features
            user_features = filtered_ratings.groupby('userId').agg({
                'rating': ['count', 'mean', 'std'],
                'movieId': 'nunique'
            }).fillna(0)
            
            user_features.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_movies']
            features_scaled = self.scaler.fit_transform(user_features)
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_clusters = dbscan.fit_predict(features_scaled)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans_clusters = kmeans.fit_predict(features_scaled)
            
            # Reduce to 2D using PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
            
            # Create DBSCAN visualization
            dbscan_df = pd.DataFrame({
                'PC1': features_2d[:, 0],
                'PC2': features_2d[:, 1],
                'Cluster': [f'Cluster {c}' if c != -1 else 'Noise' for c in dbscan_clusters]
            })
            
            dbscan_fig = px.scatter(
                dbscan_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title='DBSCAN Clustering Results'
            )
            
            dbscan_fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                showlegend=True,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            
            # Create K-means visualization
            kmeans_df = pd.DataFrame({
                'PC1': features_2d[:, 0],
                'PC2': features_2d[:, 1],
                'Cluster': [f'Cluster {c}' for c in kmeans_clusters]
            })
            
            kmeans_fig = px.scatter(
                kmeans_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title='K-means Clustering Results'
            )
            
            kmeans_fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                showlegend=True,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            
            # Calculate metrics
            kmeans_silhouette = silhouette_score(features_scaled, kmeans_clusters)
            
            results = {
                'dbscan_plot': dbscan_fig.to_dict(),
                'kmeans_plot': kmeans_fig.to_dict(),
                'clustering_metrics': {
                    'kmeans_silhouette': float(kmeans_silhouette),
                    'dbscan_clusters': int(len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)),
                    'kmeans_clusters': int(len(set(kmeans_clusters)))
                },
                'cluster_distributions': {
                    'dbscan': {str(k): int(v) for k, v in zip(*np.unique(dbscan_clusters, return_counts=True))},
                    'kmeans': {str(k): int(v) for k, v in zip(*np.unique(kmeans_clusters, return_counts=True))}
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in perform_advanced_clustering: {str(e)}")
            return {
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
                },
                'error': str(e)
            }
    
    def analyze_rating_distributions(self):
        """
        Performs detailed statistical analysis of rating distributions.
        Memory-efficient implementation.
        """
        try:
            # Take sample
            sample_size = min(100000, len(self.ratings_df))
            ratings_sample = self.ratings_df['rating'].sample(n=sample_size, random_state=42)
            
            # Calculate basic statistics
            rating_stats = ratings_sample.describe()
            
            # Normality test
            _, normality_pvalue = stats.normaltest(ratings_sample)
            
            # Create distribution visualization
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=ratings_sample,
                name='Actual Distribution',
                opacity=0.75,
                nbinsx=20
            ))
            
            # Add theoretical normal distribution
            x = np.linspace(ratings_sample.min(), ratings_sample.max(), 100)
            y = stats.norm.pdf(x, ratings_sample.mean(), ratings_sample.std())
            fig.add_trace(go.Scatter(
                x=x,
                y=y*len(ratings_sample)/5,
                name='Theoretical Normal Distribution',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title='Rating Distribution Analysis',
                title_x=0.5,
                title_font_size=20,
                xaxis_title='Rating',
                yaxis_title='Frequency',
                barmode='overlay',
                showlegend=True,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            
            results = {
                'distribution_plot': fig.to_dict(),
                'distribution_stats': {
                    'count': int(rating_stats['count']),
                    'mean': float(rating_stats['mean']),
                    'std': float(rating_stats['std']),
                    'min': float(rating_stats['min']),
                    'q1': float(rating_stats['25%']),
                    'median': float(rating_stats['50%']),
                    'q3': float(rating_stats['75%']),
                    'max': float(rating_stats['max'])
                },
                'normality_test': {
                    'p_value': float(normality_pvalue),
                    'is_normal': bool(normality_pvalue > 0.05)
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in analyze_rating_distributions: {str(e)}")
            return {
                'distribution_plot': None,
                'distribution_stats': {},
                'normality_test': {
                    'p_value': 0.0,
                    'is_normal': False
                },
                'error': str(e)
            }
    
    def _convert_to_serializable(self, obj):
        """
        Converts numpy types and other special types to JSON serializable types.
        """
        try:
            if obj is None:
                return None
            elif isinstance(obj, (bool, str, int, float)):
                return obj
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return self._convert_to_serializable(obj.tolist())
            elif isinstance(obj, pd.Series):
                return self._convert_to_serializable(obj.tolist())
            elif isinstance(obj, pd.DataFrame):
                return self._convert_to_serializable(obj.to_dict('records'))
            elif isinstance(obj, (list, tuple)):
                return [self._convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(key): self._convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (go.Figure, px.Figure)):
                return obj.to_dict()
            else:
                return str(obj)
        except Exception as e:
            print(f"Error converting {type(obj)} to JSON serializable: {str(e)}")
            return str(obj)

    def _prepare_stats_for_json(self, stats_dict):
        """
        Prepares statistics dictionary for JSON serialization.
        """
        if stats_dict is None:
            return {}
            
        try:
            return self._convert_to_serializable(stats_dict)
        except Exception as e:
            print(f"Error preparing stats for JSON: {str(e)}")
            return {'error': str(e)}

    def generate_comprehensive_report(self):
        """
        Generates a comprehensive report containing all analyses.
        """
        try:
            print("Starting comprehensive report generation...")
            
            # Initialize empty report structure
            report = {
                'anomaly_analysis': None,
                'pattern_analysis': None,
                'clustering_analysis': None,
                'distribution_analysis': None
            }
            
            # Generate individual analyses
            print("Generating anomaly analysis...")
            anomaly_results = self.detect_rating_anomalies()
            if anomaly_results and isinstance(anomaly_results, dict):
                report['anomaly_analysis'] = anomaly_results
            
            print("Generating pattern analysis...")
            pattern_results = self.extract_movie_patterns()
            if pattern_results and isinstance(pattern_results, dict):
                report['pattern_analysis'] = pattern_results
            
            print("Generating clustering analysis...")
            clustering_results = self.perform_advanced_clustering()
            if clustering_results and isinstance(clustering_results, dict):
                report['clustering_analysis'] = clustering_results
            
            print("Generating distribution analysis...")
            distribution_results = self.analyze_rating_distributions()
            if distribution_results and isinstance(distribution_results, dict):
                report['distribution_analysis'] = distribution_results
            
            # Convert numpy types to Python native types
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, go.Figure):
                    try:
                        return obj.to_dict()
                    except:
                        return None
                elif str(type(obj)) == "<class 'plotly.graph_objs._figure.Figure'>":
                    try:
                        return obj.to_dict()
                    except:
                        return None
                return obj
            
            # Convert all results to JSON serializable format
            try:
                report = convert_numpy(report)
            except Exception as e:
                print(f"Error during numpy conversion: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
            
            # Ensure all sections have required structure
            if 'anomaly_analysis' not in report or not report['anomaly_analysis']:
                report['anomaly_analysis'] = {
                    'anomaly_stats': {
                        'total_users': 0,
                        'anomaly_count': 0,
                        'normal_count': 0,
                        'anomaly_percentage': 0.0
                    },
                    'anomaly_plot': None
                }
            
            if 'pattern_analysis' not in report or not report['pattern_analysis']:
                report['pattern_analysis'] = {
                    'pattern_network': None,
                    'top_rules': []
                }
            
            if 'clustering_analysis' not in report or not report['clustering_analysis']:
                report['clustering_analysis'] = {
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
                }
            
            if 'distribution_analysis' not in report or not report['distribution_analysis']:
                report['distribution_analysis'] = {
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
            
            print("Report generation completed successfully")
            print(f"Report keys: {report.keys()}")
            return report
            
        except Exception as e:
            print(f"Error in generate_comprehensive_report: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            # Return empty report structure in case of error
            return {
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