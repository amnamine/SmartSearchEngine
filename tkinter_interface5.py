import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# ==========================================
# 1. THE AI BACKEND (Synced with Training)
# ==========================================
# STRICTLY Hardware Synonyms (No Offers/Plans)
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
    "kitman": "ecouteurs",
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
    "wifi": "modem",
    "routeur": "modem",
    "box": "modem",
    "4g": "modem",
    
    # Tablets
    "tab": "tablette",
    "ipad": "tablette"
}

def preprocess_query(query):
    """Must match exactly the training preprocessing."""
    text = str(query).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.append(SYNONYMS[w])
            
    return " ".join(expanded)

class SmartSearchEngine:
    def __init__(self):
        self.product_db = None
        self.pipeline = None

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                model_package = pickle.load(f)
            self.pipeline = model_package['pipeline']
            self.product_db = model_package['database']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def search(self, user_query, top_k=15):
        if self.product_db is None: return pd.DataFrame()
        
        clean_query = preprocess_query(user_query)
        
        # Create candidates matching training feature format
        candidates = self.product_db.copy()
        candidate_features = clean_query + " | " + candidates['search_text']
        
        try:
            # Get AI Probability
            probs = self.pipeline.predict_proba(candidate_features)[:, 1]
            candidates['ai_score'] = probs
            
            # Return top results
            return candidates.sort_values(by='ai_score', ascending=False).head(top_k)
        except Exception as e:
            print(f"Search error: {e}")
            return pd.DataFrame()

# ==========================================
# 2. THE MODERN UI
# ==========================================
class SmartSearchEngineApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Smart Search Engine — Product Search")
        self.geometry("600x850")
        self.configure(bg="#F0F2F5") 
        
        self.COLORS = {
            "primary": "#2563eb",
            "dark": "#1e293b",
            "bg": "#F0F2F5",        # Light Grey Background
            "card": "#FFFFFF",      # White Card
            "accent": "#00B894",    # Green (High match)
            "medium": "#FDCB6E",    # Orange (Medium match)
        }
        
        self.FONTS = {
            "header": ("Segoe UI", 20, "bold"),
            "sub": ("Segoe UI", 10),
            "title": ("Segoe UI", 11, "bold"),
            "body": ("Segoe UI", 10),
            "price": ("Segoe UI", 12, "bold")
        }

        # --- Load AI ---
        self.engine = SmartSearchEngine()
        self.model_loaded = False
        
        model_filename = "smart_search_engine_model_store.pkl"
        
        if os.path.exists(model_filename):
            if self.engine.load_model(model_filename):
                self.model_loaded = True
                print(f"Loaded {model_filename} successfully.")
            else:
                messagebox.showerror("Error", f"Failed to load '{model_filename}'.")
        else:
            messagebox.showwarning("Warning", f"File '{model_filename}' not found! Please run the training script first.")

        # --- Build Layout ---
        self.create_header()
        self.create_search_area()
        self.create_suggestions()
        self.create_results_area()
        self.create_footer()

        # Press Enter to search
        self.bind('<Return>', lambda event: self.run_search())

    def create_header(self):
        header = tk.Frame(self, bg=self.COLORS["primary"], height=100)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        # Logo / Title
        lbl_title = tk.Label(header, text="Smart Search Engine", font=self.FONTS["header"], 
                             bg=self.COLORS["primary"], fg="white")
        lbl_title.pack(pady=(20, 0))
        
        lbl_sub = tk.Label(header, text="Hardware Search (Phones, Modems, Accessories)", 
                           font=self.FONTS["sub"], bg=self.COLORS["primary"], fg="#bfdbfe")
        lbl_sub.pack()

    def create_search_area(self):
        frame = tk.Frame(self, bg=self.COLORS["bg"], pady=20, padx=20)
        frame.pack(fill="x")

        self.search_var = tk.StringVar()
        
        # Search Bar Wrapper
        entry_frame = tk.Frame(frame, bg="white", bd=1, relief="solid")
        entry_frame.pack(fill="x", pady=5)
        
        self.entry = tk.Entry(entry_frame, textvariable=self.search_var, font=self.FONTS["body"], 
                              bd=0, bg="white")
        self.entry.pack(fill="x", padx=15, pady=12)
        self.entry.focus()

        # Buttons
        btn_frame = tk.Frame(frame, bg=self.COLORS["bg"])
        btn_frame.pack(fill="x", pady=(15,0))

        btn_search = tk.Button(btn_frame, text="🔍 FIND PRODUCT", command=self.run_search,
                               bg=self.COLORS["dark"], fg="white", font=("Segoe UI", 10, "bold"),
                               relief="flat", cursor="hand2", width=15, pady=5)
        btn_search.pack(side="left", padx=(0, 10))

        btn_reset = tk.Button(btn_frame, text="RESET", command=self.reset_app,
                              bg="#B2BEC3", fg="white", font=("Segoe UI", 10, "bold"),
                              relief="flat", cursor="hand2", width=10, pady=5)
        btn_reset.pack(side="right")

    def create_suggestions(self):
        # "Quick Keywords" Section
        frame = tk.Frame(self, bg=self.COLORS["bg"], padx=20)
        frame.pack(fill="x", pady=(0, 10))

        lbl = tk.Label(frame, text="Quick Access (Hardware):", font=("Segoe UI", 9, "bold"), 
                       bg=self.COLORS["bg"], fg="#636E72")
        lbl.pack(anchor="w", pady=(0, 8))

        # Chip Container
        chips_frame = tk.Frame(frame, bg=self.COLORS["bg"])
        chips_frame.pack(anchor="w")

        # UPDATED: Pure Hardware Suggestions (Removed Samsung Galaxy, Added relevant ones)
        suggestions = ["Modem Wifi", "Routeur D-Link", "Tablette", "Kitman Hoco", "ZTE Blade", "Cable Type-C"]
        
        for kw in suggestions:
            btn = tk.Button(chips_frame, text=kw, 
                            command=lambda k=kw: self.fill_search(k),
                            bg="white", fg=self.COLORS["dark"], 
                            bd=0, relief="groove", font=("Segoe UI", 9),
                            cursor="hand2", padx=12, pady=4)
            btn.pack(side="left", padx=(0, 8))

    def create_results_area(self):
        container = tk.Frame(self, bg=self.COLORS["bg"])
        container.pack(fill="both", expand=True, padx=15, pady=5)

        self.canvas = tk.Canvas(container, bg=self.COLORS["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.COLORS["bg"])

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=550)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def create_footer(self):
        footer = tk.Frame(self, bg="#DFE6E9", height=35)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        
        self.status_lbl = tk.Label(footer, text="System Ready", font=("Segoe UI", 9), 
                                   bg="#DFE6E9", fg="#636E72")
        self.status_lbl.pack(pady=8)

    # --- Logic Functions ---

    def fill_search(self, text):
        self.search_var.set(text)
        self.run_search()

    def reset_app(self):
        self.search_var.set("")
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.status_lbl.config(text="System Ready")
        self.entry.focus()

    def run_search(self):
        if not self.model_loaded: 
            messagebox.showerror("Error", "AI Model not loaded!")
            return
            
        query = self.search_var.get()
        if not query.strip(): return

        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Perform AI Search
        results = self.engine.search(query, top_k=20)
        
        # Filter by relevance
        # > 0.35 is the "Golden Threshold" we found in training
        relevant = results[results['ai_score'] > 0.35]

        if relevant.empty:
            lbl = tk.Label(self.scrollable_frame, text=f"No hardware found for '{query}'", 
                           bg=self.COLORS["bg"], fg="#b2bec3", font=("Segoe UI", 11), justify="center")
            lbl.pack(pady=50)
            self.status_lbl.config(text="0 results found.")
        else:
            count = len(relevant)
            self.status_lbl.config(text=f"Found {count} products.")
            for _, row in relevant.iterrows():
                self.draw_card(row)

    def draw_card(self, row):
        card = tk.Frame(self.scrollable_frame, bg="white", padx=15, pady=12)
        card.pack(fill="x", pady=6)
        
        # 1. Header: Name + Price
        header = tk.Frame(card, bg="white")
        header.pack(fill="x")
        
        tk.Label(header, text=row['product_name'], font=self.FONTS["title"], 
                 bg="white", fg=self.COLORS["dark"]).pack(side="left")
        
        tk.Label(header, text=f"{row['price']}", font=self.FONTS["price"], 
                 bg="white", fg=self.COLORS["primary"]).pack(side="right")
        
        # 2. Category Tag
        cat_text = row.get('category', 'Product')
        tk.Label(card, text=f"[{cat_text}]", font=("Segoe UI", 8, "bold"), 
                 bg="white", fg="#0984e3", anchor="w").pack(fill="x", pady=(2,0))

        # 3. Description
        desc = str(row['description'])
        if len(desc) > 80: desc = desc[:80] + "..." 
        tk.Label(card, text=desc, font=("Segoe UI", 9), bg="white", fg="#636E72", anchor="w").pack(fill="x", pady=(2, 8))

        # 4. AI Confidence Bar
        score = int(row['ai_score'] * 100)
        bar_color = self.COLORS["accent"] if score > 75 else self.COLORS["medium"]
        
        bar_frame = tk.Frame(card, bg="white")
        bar_frame.pack(fill="x")
        
        tk.Label(bar_frame, text="Match:", font=("Segoe UI", 7, "bold"), bg="white", fg="#B2BEC3").pack(side="left")
        
        progress_bg = tk.Frame(bar_frame, bg="#F0F2F5", height=5, width=150)
        progress_bg.pack(side="left", padx=10)
        progress_bg.pack_propagate(False)
        
        fill_width = int((score / 100) * 150)
        progress_fill = tk.Frame(progress_bg, bg=bar_color, height=5, width=fill_width)
        progress_fill.pack(side="left")
        
        tk.Label(bar_frame, text=f"{score}%", font=("Segoe UI", 8, "bold"), bg="white", fg=bar_color).pack(side="right")

if __name__ == "__main__":
    app = SmartSearchEngineApp()
    app.mainloop()