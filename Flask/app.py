from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from flask_cors import CORS


# Load and preprocess your movie data here (replace with your data loading logic)
movies = pd.read_csv('Top_10000_Movies_IMDb.csv')
movies = movies.drop(['Metascore'], axis=1)
movies = movies.rename(columns={'Movie Name': 'moviename'})
movies['Rating'] = movies['Rating'].astype(str)
movies['Votes'] = movies['Votes'].astype(str)
movies['Directors'] = movies['Directors'].str.replace(' ', '').tolist()
movies['Stars'] = movies['Stars'].str.replace(' ', '').tolist()
combined_features = movies['Directors'] + " " + movies['Stars']*2 + " " + movies['Plot']*2 + " " + movies['Genre']
vectorizer = TfidfVectorizer()
feature_movies = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_movies)

app = Flask(__name__)

# Configure CORS for all origins (adjust as needed)
cors = CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route("/test")
def test_route():
  return jsonify({"message": "Recommendations endpoint works!"})

@app.route("/predict", methods=['POST'])
def recommend_page():
  data = request.get_json()
  movie_name = data.get('movieTitle')
  movie_str = str(movie_name).lower()

  if not movie_name:
    return jsonify({"error": "Missing movie title in request"}), 400

  list_of_all_titles = movies['moviename'].tolist()
  find_close_match = difflib.get_close_matches(movie_str, list_of_all_titles)
  if not find_close_match:
    return jsonify({"error": "Movie not found"}), 404

  close_match = find_close_match[0]
  index_of_the_movie = movies[movies.moviename == close_match].index[0]
  similarity_score = list(enumerate(similarity[index_of_the_movie]))
  sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

  suggestions = []
  i = 1
  for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies.loc[index, 'moviename']
    similarity_percentage = round(movie[1] * 100, 2)
    if i < 16:
      suggestions.append({
          'rank': i,
          'title': title_from_index,
          'similarity': similarity_percentage
      })
    i += 1

  return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)


# Run the Flask development server (optional)
if __name__ == "__main__":
    app.run(debug=True)
