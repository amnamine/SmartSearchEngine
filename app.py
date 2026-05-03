import os
import re
import json
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & LOGIC
# ==========================================
MODEL_FILE = "smart_search_engine_model.pkl"
JSON_FILE = "scraping5.json"

# Synonyms Dictionary (Same as your training)
SYNONYMS = {
    "telephone": "smartphone", "mobile": "smartphone", "portable": "smartphone",
    "jawl": "smartphone", "hètf": "smartphone", "tel": "smartphone",
    "cellulaire": "smartphone", "kitman": "ecouteurs", "ecouteur": "ecouteurs",
    "casque": "ecouteurs", "airpods": "ecouteurs", "earbuds": "ecouteurs",
    "chargeur": "accessoire", "cable": "accessoire", "fil": "accessoire",
    "usb": "accessoire", "powerbank": "accessoire", "wifi": "modem",
    "routeur": "modem", "box": "modem", "4g": "modem", "tab": "tablette",
    "ipad": "tablette"
}

def preprocess_query(query):
    text = str(query).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.append(SYNONYMS[w])
    return " ".join(expanded)

# ==========================================
# 2. IMAGE LOADER (Restores Images from JSON)
# ==========================================
image_map = {}

def load_images():
    """Loads scraping JSON to map product names to images."""
    global image_map
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # Create a normalized key to match CSV products
                    # We combine Title + Description to match 'clean_name' logic
                    title = item.get('title', '').strip()
                    desc = item.get('description', '').strip()
                    
                    # Try matching by Description (Model) which is usually unique
                    key_desc = desc.lower().replace(" ", "")
                    image_map[key_desc] = item.get('image')
                    
                    # Also map Full Name just in case
                    full_name = f"{title} {desc}".lower().replace(" ", "")
                    image_map[full_name] = item.get('image')
            print(f"Loaded {len(image_map)} images from JSON.")
        except Exception as e:
            print(f"Error loading JSON images: {e}")

# ==========================================
# 3. AI ENGINE CLASS
# ==========================================
class SmartSearchEngine:
    def __init__(self):
        self.product_db = None
        self.pipeline = None
        self.load_model(MODEL_FILE)

    def load_model(self, filename):
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model_package = pickle.load(f)
                self.pipeline = model_package['pipeline']
                self.product_db = model_package['database']
                print("AI Model Loaded Successfully.")
            else:
                print("Model file not found. Please train first.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def search(self, user_query, top_k=20):
        if self.product_db is None: return []
        
        clean_query = preprocess_query(user_query)
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']
        
        try:
            probs = self.pipeline.predict_proba(candidate_features)[:, 1]
            candidates['ai_score'] = probs
            
            # Filter low confidence results
            results = candidates[candidates['ai_score'] > 0.35].sort_values(by='ai_score', ascending=False).head(top_k)
            
            output = []
            for _, row in results.iterrows():
                # Try to find the image
                clean_name_key = row['product_name'].lower().replace(" ", "")
                img_url = image_map.get(clean_name_key, "https://via.placeholder.com/150?text=No+Image")
                
                output.append({
                    "name": row['product_name'],
                    "price": row['price'],
                    "category": row['category'],
                    "description": row['description'],
                    "score": round(row['ai_score'] * 100),
                    "image": img_url
                })
            return output
        except Exception as e:
            print(f"Search error: {e}")
            return []

# Initialize System
load_images()
ai_engine = SmartSearchEngine()

# ==========================================
# 4. FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    results = ai_engine.search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)