import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# 1. Load the dataset
# Ensure the file is in the current directory
df = pd.read_csv('dataset_train5.csv')

# 2. Preprocessing
# Combine product name and description to create the text features
# We use these text fields to predict the 'category'
df['text_features'] = df['product_name'].astype(str) + " " + df['description'].astype(str)

X = df['text_features']
y = df['category']

# 3. Split the data into Training and Testing sets
# We use stratify=y to ensure the class distribution is maintained in the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create a Classification Pipeline
# - TfidfVectorizer converts text to numerical vectors
# - LinearSVC is the classifier (Support Vector Machine)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Make Predictions
y_pred = pipeline.predict(X_test)

# 7. Print Accuracy and Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Generate and Plot Confusion Matrix using Seaborn
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()