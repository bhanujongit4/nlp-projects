from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from flask_cors import CORS

app = Flask(__name__)

# Custom JSON Encoder to handle numpy.int64
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)  # Convert np.int64 to Python int
        return json.JSONEncoder.default(self, obj)

app.json_encoder = CustomJSONEncoder

# Load your movie data
movies = pd.read_csv('Top_10000_Movies_IMDb.csv')

# Drop unnecessary columns and handle data types
movies = movies.drop(['Metascore'], axis=1)
movies['Rating'] = movies['Rating'].astype(str)
movies['Votes'] = movies['Votes'].astype(str)

# Preprocess text fields
movies['Directors'] = movies['Directors'].str.replace(' ', '').tolist()
movies['Stars'] = movies['Stars'].str.replace(' ', '').tolist()

# Use label encoding for 'moviename'
label_encoder = LabelEncoder()
movies['moviename_encoded'] = label_encoder.fit_transform(movies['Movie Name'])

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(movies))

cors = CORS(app, resources={r"/pred": {"origins": "*"}})
@app.route('/pred', methods=['POST'])
def predict():
    data = request.get_json()
    movie_name = str(data.get('movieTitle')).lower()

    if not movie_name:
        return jsonify({"error": "Missing movie title in request"}), 400

    inputs = tokenizer(movie_name, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Assuming model output is logits for each class (label)
    scores = outputs[0].cpu().numpy().flatten()

    # Find top 10 similar movies
    top_indices = np.argsort(scores)[-100:][::-1]  # Get top 10 indices
    recommendations = []
    
    for idx in top_indices:
        title = movies.loc[idx, 'Movie Name']
        similarity_percentage = float(scores[idx])  # Convert to Python float
        
        recommendations.append({
            'rank': int(idx),  # Ensure idx is converted to Python int
            'title': title,
            'similarity': similarity_percentage
        })

    print(recommendations)  # For debugging purposes
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
