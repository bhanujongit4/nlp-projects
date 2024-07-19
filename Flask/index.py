from flask import Flask, request, jsonify
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load and preprocess your movie data here (replace with your data loading logic)
movies = pd.read_csv('Top_10000_Movies_IMDb.csv')
movies = movies.drop(['Metascore'], axis=1)
movies = movies.rename(columns={'Movie Name': 'moviename'})
movies['Rating'] = movies['Rating'].astype(str)
movies['Votes'] = movies['Votes'].astype(str)

# Split features for Word2Vec training without stopwords
stop_words = set(stopwords.words('english'))
features = []
for index, row in movies.iterrows():
    directors = [word for word in row['Directors'].lower().split() if word not in stop_words]
    stars = [word for word in row['Stars'].lower().split() if word not in stop_words]
    plot = [word for word in row['Plot'].lower().split() if word not in stop_words]
    genre = [word for word in row['Genre'].lower().split() if word not in stop_words]
    moviename = [word for word in row['moviename'].lower().split() if word not in stop_words]
    features.append(directors + stars + plot + genre + moviename)

# Train Word2Vec model
model = Word2Vec(features, vector_size=100, window=5, min_count=1, workers=3)

# Function to get movie vector without stopwords
def get_movie_vector(movie_title):
    movie_title = movie_title.lower()
    tokens = [token for token in movie_title.split() if token not in stop_words]
    if len(tokens) == 0:
        return None
    movie_vec = np.zeros(model.vector_size)
    for token in tokens:
        if token in model.wv:
            movie_vec += model.wv[token]
    movie_vec /= len(tokens)  # Average word vectors
    return movie_vec

app = Flask(__name__)

# Configure CORS for all origins (adjust as needed)
cors = CORS(app, resources={r"/prediction": {"origins": "*"}})

@app.route("/test")
def test_route():
    return jsonify({"message": "Recommendations endpoint works!"})

@app.route("/prediction", methods=['POST'])
def recommend_page():
    data = request.get_json()
    movie_name = data.get('movieTitle')

    if not movie_name:
        return jsonify({"error": "Missing movie title in request"}), 400

    movie_vec = get_movie_vector(movie_name)
    if movie_vec is None:
        return jsonify({"error": "Movie not found or invalid title"}), 404

    # Calculate cosine similarity with all movies (excluding stopwords)
    movie_vectors = np.array([get_movie_vector(mov) for mov in movies['moviename'] if get_movie_vector(mov) is not None])
    similarity_scores = cosine_similarity(movie_vec.reshape(1, -1), movie_vectors)[0]

    # Sort movies by similarity
    sorted_similar_movies = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)

    suggestions = []
    i = 1
    for movie_index, similarity in sorted_similar_movies:
        title = movies.iloc[movie_index]['moviename']
        similarity_percentage = round(similarity * 100, 2)
        if title.lower() == movie_name.lower():
            continue  # Skip the query movie itself
        if i <= 50:
            suggestions.append({
                'rank': i,
                'title': title,
                'similarity': similarity_percentage
            })
        i += 1

    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True, port=1000)


