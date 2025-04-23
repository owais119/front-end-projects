import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("sms spamer/SMSSpamCollection.csv", sep='\t', names=['label', 'message'])

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(cleaned)  # Return string for vectorizer

# Apply preprocessing
df['cleaned_text'] = df['message'].apply(preprocess_text)

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])  # ham=0, spam=1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label_encoded'], test_size=0.2, random_state=42)

# Vectorize text using Bag of Words
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict
y_pred = model.predict(X_test_vect)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Make predictions
y_pred = model.predict(X_test_vect)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1-Score
f1 = f1_score(y_test, y_pred)

# Output results
print(f"🔍 Evaluation Metrics:\n")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")


# Optional: Full classification report
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))



#predict


# New message
new_message = "Win a free iPhone now!"

# Preprocess it
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(cleaned)

# Preprocess and vectorize
preprocessed = preprocess_text(new_message)
vectorized = vectorizer.transform([preprocessed])

# Predict
prediction = model.predict(vectorized)
predicted_label = le.inverse_transform(prediction)[0]

print(f"📨 Message: {new_message}")
print(f"🔎 Predicted Class: {predicted_label}")


#plot


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict on test set
y_pred = model.predict(X_test_vect)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = le.classes_  # ['ham', 'spam']

# Plot using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
