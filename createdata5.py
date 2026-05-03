import json
import csv
import random
import re
import uuid

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'scraping5.json'
OUTPUT_FILE = 'dataset_train5.csv'  # Saving as the new version

TARGET_DATASET_SIZE = 3500 

# Keep the boosting logic (It was good for balancing)
# HIGH (15x): Tablets/Modems need to be visible
HIGH_BOOST = ["Tablette", "Routeur_Modem"]
# MEDIUM (3x): Smartphones
MEDIUM_BOOST = ["Smartphone"] 

# Categories to extract core keywords from
CATEGORY_KEYWORDS = {
    "Smartphone": ["zte", "tecno", "oppo", "samsung", "realme", "blade", "nubia", "pova", "spark", "infinix", "galaxy", "redmi", "xiaomi", "v60", "a75", "a35"],
    "Routeur_Modem": ["d-link", "tcl", "modem", "box", "dwr", "wifi", "4g", "routeur", "mw40"],
    "Tablette": ["tablette", "tab", "d-tech", "ipad", "pad"],
    "Accessoire_Audio": ["earbuds", "ecouteur", "airpods", "casque", "kit", "bluetooth", "hoco", "audio"],
    "Accessoire_Charge": ["cable", "chargeur", "powerbank", "usb", "type-c", "lightning", "batterie"],
    "Accessoire_Auto": ["support", "car", "voiture", "fm"]
}

# ==========================================
# 1. THE TYPO ENGINE (Simulating "Lazy User")
# ==========================================
def mess_up_text(text):
    """
    Takes a clean word (e.g., 'tablette') and breaks it 
    like a human typing fast on a phone.
    """
    if len(text) < 3: return text # Don't mess up 2-letter words
    
    # 30% chance to return perfect text (Clean search)
    if random.random() < 0.3:
        return text

    chars = list(text)
    action = random.choice(['delete', 'swap', 'duplicate', 'nothing'])

    try:
        if action == 'delete':
            # Remove one random character (e.g. 'tablette' -> 'tabltte')
            idx = random.randint(0, len(chars) - 1)
            del chars[idx]
        
        elif action == 'swap':
            # Swap two neighbor chars (e.g. 'wifi' -> 'wfii')
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            
        elif action == 'duplicate':
            # Double a char (e.g. 'samsung' -> 'sammsung')
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
            
    except IndexError:
        pass # Safety pass

    return "".join(chars)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'[^\w\s]', ' ', text) 
    return text.strip()

def clean_price(price_str):
    if not price_str: return "0 DA"
    clean = str(price_str).replace('&nbsp;', ' ').replace('&nbsp', ' ').replace('\xa0', ' ')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def get_category(text):
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "Accessoire_General"

def extract_core_keywords(prod):
    """
    Returns a list of 1-2 word distinct keywords for this product.
    Example: ['samsung', 'galaxy', 'a55', 'samsung galaxy']
    """
    keywords = set()
    
    # Add Category (e.g., "Smartphone")
    cat_clean = prod['category'].replace("_", " ").lower()
    keywords.add(cat_clean)
    
    # Add Brand (e.g., "Samsung")
    brand = prod['brand'].lower()
    keywords.add(brand)
    
    # Add Model parts (e.g., "Galaxy", "A55")
    model_parts = prod['model'].lower().split()
    for part in model_parts:
        if len(part) > 1: # Ignore 'a', 'x' etc.
            keywords.add(part)
    
    # Add 2-word combinations (User limit: Max 2 words)
    keywords.add(f"{brand} {model_parts[0] if model_parts else ''}".strip())
    
    # Add Synonyms hardcoded for specific categories
    if "modem" in cat_clean or "routeur" in cat_clean:
        keywords.add("wifi")
        keywords.add("4g")
    
    if "tablette" in cat_clean:
        keywords.add("tab")
        
    return list(keywords)

# ==========================================
# 3. MAIN GENERATOR
# ==========================================
def create_dataset_v5():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] {INPUT_FILE} not found.")
        return

    # --- Step 1: Deduplicate & Clean ---
    products_map = {}
    for item in raw_data:
        brand = clean_text(item.get("title", ""))       
        model = clean_text(item.get("description", "")) 
        
        if model.lower().startswith(brand.lower()):
            clean_name = model
        else:
            clean_name = f"{brand} {model}"
            
        unique_key = clean_name.lower().replace(" ", "")
        
        if unique_key not in products_map and unique_key != "":
            products_map[unique_key] = {
                "id": str(uuid.uuid4())[:8],
                "brand": brand,
                "model": model,
                "name": clean_name,
                "category": get_category(clean_name),
                "price": clean_price(item.get("price", "0 DA"))
            }

    unique_products = list(products_map.values())
    print(f"[Init] Processed {len(unique_products)} unique products.")

    # --- Step 2: Generate Short & Messy Data ---
    dataset_rows = []
    
    # Calculate base rows needed
    base_rows = max(10, TARGET_DATASET_SIZE // len(unique_products))

    for prod in unique_products:
        # Determine number of rows (Keeping your boosting logic)
        if prod['category'] in HIGH_BOOST:
            n_rows = base_rows * 15
        elif prod['category'] in MEDIUM_BOOST:
            n_rows = base_rows * 3
        else:
            n_rows = max(5, int(base_rows * 0.3)) 

        # Get core words: ["samsung", "galaxy", "a55"]
        core_words = extract_core_keywords(prod)
        
        # --- POSITIVES (Matches) ---
        n_pos = int(n_rows * 0.5)
        for _ in range(n_pos):
            # Pick a base word
            base = random.choice(core_words)
            # Mess it up (Typo)
            query = mess_up_text(base)
            
            dataset_rows.append({
                "product_id": prod['id'],
                "product_name": prod['name'],
                "category": prod['category'],
                "description": prod['model'],
                "price": prod['price'],
                "user_query": query,
                "relevance_label": 1 # MATCH
            })

        # --- NEGATIVES (Non-Matches) ---
        n_neg = n_rows - n_pos
        for _ in range(n_neg):
            other = random.choice(unique_products)
            while other['id'] == prod['id']:
                other = random.choice(unique_products)

            # Smart Negative:
            # If I am selling a Tablet, I want to learn that "Samsung" (phone) is NOT me.
            other_words = extract_core_keywords(other)
            base = random.choice(other_words)
            query = mess_up_text(base)

            dataset_rows.append({
                "product_id": prod['id'],
                "product_name": prod['name'],
                "category": prod['category'],
                "description": prod['model'],
                "price": prod['price'],
                "user_query": query,
                "relevance_label": 0 # NO MATCH
            })

    # --- Step 3: Save ---
    random.shuffle(dataset_rows)
    
    headers = ["product_id", "product_name", "category", "description", "price", "user_query", "relevance_label"]
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"[Done] Generated {len(dataset_rows)} rows in '{OUTPUT_FILE}'.")
    print(f"[Info] Queries are now short (1-2 words) and include realistic typos.")

if __name__ == "__main__":
    create_dataset_v5()