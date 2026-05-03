"""
Build dataset_catalogue_train.csv from Clean_Catalogue.csv for SmartSearchEngine training.
"""
import csv
import random
import re
import urllib.parse
import uuid

import pandas as pd

INPUT_FILE = "Clean_Catalogue.csv"
OUTPUT_FILE = "dataset_catalogue_train.csv"
TARGET_DATASET_SIZE = 14000

HIGH_BOOST = ["Smartphone", "Tablette", "Tablet"]
MEDIUM_BOOST = ["Feature Phone", "Basic Phone", "BasicPhone", "Feature Phone"]

CATEGORY_KEYWORDS = {
    "Smartphone": [
        "samsung", "galaxy", "iphone", "xiaomi", "redmi", "oppo", "vivo", "realme",
        "tecno", "infinix", "nokia", "huawei", "honor", "motorola", "oneplus", "zte",
        "blade", "nubia", "pova", "spark", "a10", "a20", "m20", "j4", "j6",
    ],
    "Tablet": ["tablette", "tab", "ipad", "pad", "mediapad"],
    "Basic Phone": ["nokia", "basic", "dual sim", "2g"],
    "Feature Phone": ["feature", "clamshell"],
    "Smartwatch": ["watch", "montre", "band"],
    "Accessoire": ["chargeur", "cable", "case", "écouteur", "casque"],
}


def mess_up_text(text):
    if not text or len(text) < 3:
        return text
    if random.random() < 0.3:
        return text
    chars = list(text)
    action = random.choice(["delete", "swap", "duplicate"])
    try:
        if action == "delete":
            idx = random.randint(0, len(chars) - 1)
            del chars[idx]
        elif action == "swap":
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif action == "duplicate":
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
    except IndexError:
        pass
    return "".join(chars)


def clean_text(val):
    if pd.isna(val):
        return ""
    text = str(val)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def clean_price(val):
    if pd.isna(val):
        return "0 DZD"
    s = str(val).replace("\xa0", " ").strip()
    if not s.endswith("DZD") and not s.endswith("DA"):
        s = f"{s} DZD"
    return s


def norm_category(device_cat, device_type):
    a = clean_text(device_cat) or clean_text(device_type)
    if not a:
        return "Autre"
    return a.replace("_", " ")


def infer_category_keywords_row(name_lower, cat_lower, raw_cat):
    rc = (raw_cat or "").lower()
    if "basic" in rc:
        return "Basic Phone"
    if "feature" in rc:
        return "Feature Phone"
    if "tablet" in rc or "tablette" in rc:
        return "Tablet"
    if "smart" in rc:
        return "Smartphone"
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in name_lower for k in keys) or any(k in cat_lower for k in keys):
            return cat
    return "Autre"


def build_description(row, colmap):
    parts = []
    for key in ("OS", "ram", "Internal_memory", "Chipset", "Technologie cellulaire"):
        c = colmap.get(key)
        if c and c in row.index and pd.notna(row[c]):
            parts.append(clean_text(row[c])[:120])
    return " ".join(p for p in parts if p)[:400]


def gsmarena_search_url(full_name):
    q = clean_text(full_name)
    if not q:
        return ""
    return "https://www.gsmarena.com/res.php3?sSearch=" + urllib.parse.quote_plus(q)


def pick_image_url(row, lien_col, photos_col):
    for c in (lien_col, photos_col):
        if not c or c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.startswith("http"):
            return s
    return ""


def resolve_columns(df):
    cols = list(df.columns)
    if cols and cols[0].startswith("\ufeff"):
        cols[0] = cols[0].lstrip("\ufeff")
    df.columns = cols

    def find(pred):
        for c in df.columns:
            if pred(c):
                return c
        return None

    marque = find(lambda c: c == "Marque")
    nom = find(lambda c: "Nom de" in c and "mod" in c.lower())
    dev_type = find(lambda c: c == "Device Type")
    dev_cat = find(lambda c: "Cat" in c and "gorie" in c)
    price = find(lambda c: c == "Price_DZD")
    pid = find(lambda c: c == "Product_ID")
    lien = find(lambda c: c.startswith("Lien ") and "rence" in c)
    photos = find(lambda c: "Liens photos" in c)

    os_c = find(lambda c: c == "OS")
    ram_c = find(lambda c: c == "ram")
    im_c = find(lambda c: c == "Internal_memory")
    chip_c = find(lambda c: c == "Chipset")
    cell_c = find(lambda c: c == "Technologie cellulaire")

    return {
        "pid": pid,
        "marque": marque,
        "nom": nom,
        "dev_type": dev_type,
        "dev_cat": dev_cat,
        "price": price,
        "lien": lien,
        "photos": photos,
        "OS": os_c,
        "ram": ram_c,
        "Internal_memory": im_c,
        "Chipset": chip_c,
        "Technologie cellulaire": cell_c,
    }


def extract_core_keywords(prod):
    keywords = set()
    cat = prod["category"].lower().replace("_", " ")
    keywords.add(cat)
    brand = prod["brand"].lower()
    if brand:
        keywords.add(brand)
    name = prod["name"].lower()
    for part in name.split():
        if len(part) > 1:
            keywords.add(part)
    if brand:
        first = name.split()[0] if name.split() else ""
        keywords.add(f"{brand} {first}".strip())
    if "smartphone" in cat or "phone" in cat:
        keywords.add("telephone")
        keywords.add("mobile")
    if "tablet" in cat or "tablette" in cat:
        keywords.add("tab")
    return [k for k in keywords if k]


def create_catalogue_dataset():
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
    )
    cm = resolve_columns(df)
    if not cm["pid"] or not cm["nom"]:
        print("[ERROR] Could not resolve Product_ID or model name column.")
        return

    products_map = {}
    for _, row in df.iterrows():
        pid = row.get(cm["pid"])
        if pd.isna(pid):
            continue
        pid = str(pid).strip()
        marque = clean_text(row[cm["marque"]]) if cm["marque"] else ""
        nom = clean_text(row[cm["nom"]]) if cm["nom"] else ""
        if not nom:
            continue
        if marque and nom.lower().startswith(marque.lower()):
            full_name = nom.strip()
        elif marque:
            full_name = f"{marque} {nom}".strip()
        else:
            full_name = nom.strip()
        raw_dc = row[cm["dev_cat"]] if cm["dev_cat"] else ""
        raw_dt = row[cm["dev_type"]] if cm["dev_type"] else ""
        cat = norm_category(raw_dc, raw_dt)
        name_lower = full_name.lower()
        cat_lower = cat.lower()
        train_cat = infer_category_keywords_row(name_lower, cat_lower, str(raw_dc) + " " + str(raw_dt))

        desc = build_description(row, cm)
        price = clean_price(row[cm["price"]]) if cm["price"] else "0 DZD"
        img = pick_image_url(row, cm["lien"], cm["photos"])
        page_url = gsmarena_search_url(full_name)

        key = pid
        if key not in products_map:
            products_map[key] = {
                "id": pid,
                "brand": marque,
                "name": full_name,
                "category": cat,
                "train_cat": train_cat,
                "description": desc or nom[:200],
                "price": price,
                "product_url": page_url,
                "image_url": img,
            }

    unique_products = list(products_map.values())
    print(f"[Init] Catalogue: {len(unique_products)} unique products.")

    base_rows = max(4, TARGET_DATASET_SIZE // max(len(unique_products), 1))
    dataset_rows = []

    for prod in unique_products:
        tc = prod.get("train_cat", prod["category"])
        if tc in HIGH_BOOST:
            n_rows = base_rows * 4
        elif tc in MEDIUM_BOOST or any(
            x in prod["category"].lower() for x in ("phone", "basic")
        ):
            n_rows = base_rows * 2
        else:
            n_rows = max(3, int(base_rows * 0.5))

        core_words = extract_core_keywords(prod)
        if not core_words:
            core_words = [prod["name"].lower()[:12]]

        n_pos = int(n_rows * 0.55)
        for _ in range(n_pos):
            base = random.choice(core_words)
            query = mess_up_text(base)
            dataset_rows.append({
                "product_id": prod["id"],
                "product_name": prod["name"],
                "category": prod["category"],
                "description": prod["description"],
                "price": prod["price"],
                "user_query": query,
                "relevance_label": 1,
                "product_url": prod["product_url"],
                "image_url": prod["image_url"],
            })

        n_neg = n_rows - n_pos
        for _ in range(n_neg):
            other = random.choice(unique_products)
            while other["id"] == prod["id"]:
                other = random.choice(unique_products)
            owords = extract_core_keywords(other)
            if not owords:
                owords = [other["name"][:10]]
            base = random.choice(owords)
            query = mess_up_text(base)
            dataset_rows.append({
                "product_id": prod["id"],
                "product_name": prod["name"],
                "category": prod["category"],
                "description": prod["description"],
                "price": prod["price"],
                "user_query": query,
                "relevance_label": 0,
                "product_url": prod["product_url"],
                "image_url": prod["image_url"],
            })

    random.shuffle(dataset_rows)
    headers = [
        "product_id", "product_name", "category", "description", "price",
        "user_query", "relevance_label", "product_url", "image_url",
    ]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(dataset_rows)

    print(f"[Done] {len(dataset_rows)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    create_catalogue_dataset()
