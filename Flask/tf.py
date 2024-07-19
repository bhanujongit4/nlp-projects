from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from flask_cors import CORS
import torch
import torch.nn as nn
import pickle

# Load and preprocess your movie data
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

# Sentiment Analysis Model Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, num_classes):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, embedding_dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.permute(1, 2, 0)  # (batch, embedding_dim, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

# Load the vocabulary
# Load the vocabulary
with open('word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

# Print vocabulary information
print(f"Vocabulary size: {len(word_to_idx)}")
print(f"Maximum token value: {max(word_to_idx.values())}")

# Use the original vocab_size
vocab_size = 10000  # This should match the size used when training the model
embed_dim = 64
num_heads = 8
ff_dim = 512
num_layers = 4
max_len = 100
num_classes = 2

# Load the sentiment model
model = SentimentModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, num_classes).to(device)
model.load_state_dict(torch.load('sentiment_model.pth', map_location=device))
model.eval()

def tokenize(text, word_to_idx, max_len=100):
    # Use vocab_size - 1 as the 'unknown' token
    unk_token = vocab_size - 1
    tokens = [word_to_idx.get(word, unk_token) for word in text.lower().split()[:max_len]]
    if len(tokens) < max_len:
        tokens += [unk_token] * (max_len - len(tokens))
    return tokens

def predict_sentiment(text):
    text = ' '.join(text.split()[:100])  # Limit to first 100 words
    tokens = tokenize(text, word_to_idx)
    tokens_tensor = torch.LongTensor([tokens]).to(device)
    
   
    with torch.no_grad():
        output = model(tokens_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    return prediction.item(), confidence.item()

# Calculate sentiment for all movies
movies['sentiment'], movies['confidence'] = zip(*combined_features.apply(predict_sentiment))

app = Flask(__name__)
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
    
    # Combine similarity with sentiment and confidence
    combined_score = [(i, s * (1 + movies.loc[i, 'sentiment'] * movies.loc[i, 'confidence']))
                      for i, s in similarity_score]
    
    sorted_similar_movies = sorted(combined_score, key=lambda x: x[1], reverse=True)
    
    suggestions = []
    for i, (index, score) in enumerate(sorted_similar_movies[:50], 1):
        title = movies.loc[index, 'moviename']
        sentiment = "Positive" if movies.loc[index, 'sentiment'] == 1 else "Negative"
        suggestions.append({
            'rank': i,
            'title': title,
            'similarity': round(score * 100, 2),
            'sentiment': sentiment,
            'confidence': round(movies.loc[index, 'confidence'] * 100, 2)
        })
    
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)