from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
from flask_cors import CORS

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


vocab_size = 10000 # Including <PAD> and <UNK>
embed_dim = 64
num_heads = 8
ff_dim = 512
num_layers = 4
max_len = 100
num_classes = 2

model = SentimentModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, num_classes).to(device)
model.load_state_dict(torch.load('sentiment_model.pth', map_location=device))
model.eval()

# Load the vocabulary
with open('word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

# Tokenization function
def tokenize(text, word_to_idx, max_len=100):
    tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text.lower().split()[:max_len]]
    if len(tokens) < max_len:
        tokens += [word_to_idx['<PAD>']] * (max_len - len(tokens))
    return tokens

# Prediction function
def predict_sentiment(text):
    tokens = torch.LongTensor([tokenize(text, word_to_idx)]).to(device)
    with torch.no_grad():
        output = model(tokens)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    return prediction.item(), confidence.item()

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    sentiment, confidence = predict_sentiment(text)
    
    return jsonify({
        'text': text,
        'sentiment': 'Positive' if sentiment == 1 else 'Negative',
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True,port=1000)