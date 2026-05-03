import os
import re
import json
import pickle
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from flask import Flask, render_template, request, jsonify

from gsmarena_resolve import pictures_url_to_spec_url
from gsmarena_crawl import (
    load_index,
    resolve_url_for_product,
    guess_marque_from_product_name,
)

app = Flask(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_GSM_SLUG_INDEX, _GSM_BY_BRAND, _GSM_BRAND_PAGES = load_index(_BASE_DIR / "gsmarena_devices.json")
print(
    f"GSMArena index: {len(_GSM_SLUG_INDEX)} slugs, {len(_GSM_BY_BRAND)} brands"
    f" (from gsmarena_devices.json)"
)


def _catalogue_gsmarena_url(row, stored_link: str, image_url: str) -> str:
    """Resolve a direct GSMArena spec URL using crawl index + picture URLs + Bing fallback."""
    img = (image_url or "").strip()
    p = pictures_url_to_spec_url(img)
    if p:
        return p.split("?")[0]

    name = str(row.get("product_name", "") or "")
    marque = guess_marque_from_product_name(name)
    resolved = resolve_url_for_product(
        name,
        marque,
        img,
        _GSM_SLUG_INDEX,
        _GSM_BY_BRAND,
        _GSM_BRAND_PAGES,
    )
    if resolved:
        return resolved.split("?")[0]

    sl = (stored_link or "").strip()
    if sl and re.search(r"gsmarena\.com/[a-z0-9_]+-\d+\.php", sl, re.I) and "res.php3" not in sl.lower():
        return sl.split("?")[0]

    return f"https://www.bing.com/search?q={quote_plus('site:gsmarena.com ' + name)}"

MODEL_FILE_STORE = "smart_search_engine_model_store.pkl"
MODEL_FILE_CATALOGUE = "smart_search_engine_model_catalogue.pkl"
JSON_FILE = "scraping5.json"

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


image_map = {}


def load_images():
    global image_map
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    title = item.get('title', '').strip()
                    desc = item.get('description', '').strip()
                    key_desc = desc.lower().replace(" ", "")
                    image_map[key_desc] = item.get('image')
                    full_name = f"{title} {desc}".lower().replace(" ", "")
                    image_map[full_name] = item.get('image')
            print(f"Loaded {len(image_map)} images from JSON.")
        except Exception as e:
            print(f"Error loading JSON images: {e}")


class SmartSearchEngine:
    def __init__(self, model_path):
        self.product_db = None
        self.pipeline = None
        self.model_path = model_path
        self.default_source = "boutique"
        self.load_model(model_path)

    def load_model(self, filename):
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model_package = pickle.load(f)
                self.pipeline = model_package['pipeline']
                self.product_db = model_package['database']
                pkg_src = model_package.get('source', '')
                if not pkg_src and 'source' in self.product_db.columns:
                    pkg_src = str(self.product_db['source'].iloc[0])
                self.default_source = pkg_src or "boutique"
                print(f"AI model loaded: {filename} ({self.default_source})")
            else:
                print(f"Model file not found: {filename}")
        except Exception as e:
            print(f"Error loading model {filename}: {e}")

    def search(self, user_query, top_k=20, score_threshold=0.35):
        if self.product_db is None or self.pipeline is None:
            return []

        clean_query = preprocess_query(user_query)
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']

        try:
            probs = self.pipeline.predict_proba(candidate_features)[:, 1]
            candidates['ai_score'] = probs
            results = candidates[candidates['ai_score'] > score_threshold].sort_values(
                by='ai_score', ascending=False
            ).head(top_k)

            output = []
            for _, row in results.iterrows():
                clean_name_key = row['product_name'].lower().replace(" ", "")
                src = str(row['source']) if 'source' in row.index and pd.notna(row['source']) else self.default_source

                img_url = ""
                if 'image_url' in row.index:
                    iu = row['image_url']
                    if pd.notna(iu) and str(iu).strip().startswith('http'):
                        img_url = str(iu).strip()
                if not img_url:
                    img_url = image_map.get(
                        clean_name_key,
                        "https://via.placeholder.com/150?text=No+Image"
                    )

                link = ""
                if 'product_url' in row.index:
                    pu = row['product_url']
                    if pd.notna(pu):
                        link = str(pu).strip()

                if src == "catalogue":
                    link = _catalogue_gsmarena_url(row, link, img_url)

                label = "Catalogue" if src == "catalogue" else "Boutique"

                pid = ""
                if "product_id" in row.index and pd.notna(row.get("product_id")):
                    pid = str(row["product_id"]).strip()

                output.append({
                    "product_id": pid,
                    "name": row['product_name'],
                    "price": row['price'],
                    "category": row['category'],
                    "description": row['description'] if pd.notna(row.get('description')) else "",
                    "score": round(row['ai_score'] * 100),
                    "image": img_url,
                    "url": link,
                    "source": src,
                    "source_label": label,
                })
            return output
        except Exception as e:
            print(f"Search error: {e}")
            return []


load_images()
engine_store = SmartSearchEngine(MODEL_FILE_STORE)
engine_catalogue = SmartSearchEngine(MODEL_FILE_CATALOGUE)


def combined_search(query, top_k_each=22, max_total=45):
    out = []
    if engine_store.product_db is not None:
        out.extend(engine_store.search(query, top_k=top_k_each))
    if engine_catalogue.product_db is not None:
        out.extend(engine_catalogue.search(query, top_k=top_k_each))
    out.sort(key=lambda x: -x['score'])
    return out[:max_total]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    results = combined_search(query)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
