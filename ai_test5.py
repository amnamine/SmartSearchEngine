import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# --- CONFIGURATION ---
DATASET_FILE = "dataset_train5.csv"
MODEL_FILE_STORE = "smart_search_engine_model_store.pkl"
MODEL_FILE_CATALOGUE = "smart_search_engine_model_catalogue.pkl"
# Legacy default for single-model scripts
MODEL_FILE = MODEL_FILE_STORE

# --- 1. THE BRAIN: SYNONYM MAPPING (STRICTLY HARDWARE) ---
# Removed: legend, storm, flexy, puce, net (User requirement: No internet offers)
SYNONYMS = {
    # Smartphones
    "telephone": "smartphone",
    "mobile": "smartphone",
    "portable": "smartphone",
    "jawl": "smartphone",
    "hètf": "smartphone",
    "tel": "smartphone",
    "cellulaire": "smartphone",
    
    # Accessories (Audio/Charge)
    "kitman": "ecouteurs",     # Common slang for earphones
    "ecouteur": "ecouteurs",
    "casque": "ecouteurs",
    "airpods": "ecouteurs",
    "earbuds": "ecouteurs",
    "chargeur": "accessoire",
    "cable": "accessoire",
    "fil": "accessoire",
    "usb": "accessoire",
    "powerbank": "accessoire",
    
    # Modems/Routers
    "wifi": "modem",           # Users say "wifi" when looking for a modem
    "routeur": "modem",
    "box": "modem",
    "4g": "modem",
    
    # Tablets
    "tab": "tablette",
    "ipad": "tablette"
}

def preprocess_query(query):
    """Cleans text and expands synonyms."""
    if pd.isna(query):
        return ""
    text = str(query).lower().strip()
    text = re.sub(r'[^\w\s]', '', text) # Remove special chars
    
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.append(SYNONYMS[w])
            
    return " ".join(expanded)

# --- 2. THE AI ENGINE CLASS ---
class SmartSearchEngine:
    def __init__(self):
        self.product_db = None
        # The 'Brain' (Pipeline)
        # Using SGDClassifier (Logistic Regression) for fast, efficient text classification
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 3))),
            ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42))
        ])
        
    def train(self, csv_path, source_tag="boutique"):
        print(f"[AI] Loading dataset from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"[ERROR] Dataset '{csv_path}' not found. Make sure it is in the same folder.")
            return

        # Create features: We combine Query + Product Info to learn the match pattern
        # Format: "QUERY | PRODUCT INFO"
        df['features'] = df['user_query'].apply(preprocess_query) + " | " + \
                         df['product_name'].fillna('') + " " + \
                         df['category'].fillna('') + " " + \
                         df['description'].fillna('') + " " + \
                         df['price'].astype(str)
        
        X = df['features']
        y = df['relevance_label']

        print(f"[AI] Training model on {len(df)} examples...")
        self.pipeline.fit(X, y)
        
        base_cols = ['product_id', 'product_name', 'category', 'description', 'price']
        extra = [c for c in ('product_url', 'image_url') if c in df.columns]
        self.product_db = df[base_cols + extra].drop_duplicates(subset=['product_id']).copy()
        if 'product_url' not in self.product_db.columns:
            self.product_db['product_url'] = ''
        else:
            self.product_db['product_url'] = self.product_db['product_url'].fillna('').astype(str)
        if 'image_url' not in self.product_db.columns:
            self.product_db['image_url'] = ''
        else:
            self.product_db['image_url'] = self.product_db['image_url'].fillna('').astype(str)
        self.product_db['source'] = source_tag
        
        # Pre-compute the search text for the inference phase
        self.product_db['search_text'] = self.product_db['product_name'].fillna('') + " " + \
                                         self.product_db['category'].fillna('') + " " + \
                                         self.product_db['description'].fillna('') + " " + \
                                         self.product_db['price'].astype(str)
        
        print("[AI] Training Complete.")

    def save_model(self, filename):
        """Saves the trained pipeline AND the product database to a file."""
        if self.product_db is None:
            print("[ERROR] Cannot save: Model is not trained yet.")
            return
            
        src = ''
        if self.product_db is not None and len(self.product_db) and 'source' in self.product_db.columns:
            src = str(self.product_db['source'].iloc[0])
        model_package = {
            'pipeline': self.pipeline,
            'database': self.product_db,
            'source': src,
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"[SUCCESS] Model saved to '{filename}'")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")

    def search(self, user_query, top_k=5):
        """Test function to verify the model works immediately after training."""
        if self.product_db is None:
            print("[ERROR] Model not ready.")
            return pd.DataFrame()

        clean_query = preprocess_query(user_query)
        
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']
        
        # Predict probability (0 to 1)
        probs = self.pipeline.predict_proba(candidate_features)[:, 1]
        candidates['ai_score'] = probs
        
        final_results = candidates.sort_values(by='ai_score', ascending=False).head(top_k)
        return final_results[['product_name', 'price', 'ai_score', 'description']]

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    engine = SmartSearchEngine()
    
    # Train with your specific file
    engine.train(DATASET_FILE, source_tag="boutique")
    engine.save_model(MODEL_FILE_STORE)
    
    # --- DEMO ---
    test_queries = [
        "tablette",
        "wifi d-link", 
        "telephone zte",
        "kitman hoco", # Should find earphones
        "modem 4g"
    ]
    
    print("\n" + "="*50)
    print("   SMART SEARCH ENGINE — DEMO   ")
    print("="*50)

    for q in test_queries:
        print(f"\n>> User Search: '{q}'")
        results = engine.search(q)
        
        if not results.empty:
            for i, row in results.iterrows():
                if row['ai_score'] > 0.3: # Only show relevant hits
                    # [MATCH] tag used for safety against encoding errors
                    print(f"   [MATCH] ({row['ai_score']:.2f}) -> {row['product_name']} [{row['price']}]")
        else:
            print("   (No results)")