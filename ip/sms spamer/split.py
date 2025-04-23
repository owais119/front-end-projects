import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("sms spamer\SMSSpamCollection.csv", sep='\t', names=['label', 'message'])

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return cleaned_tokens

# Apply preprocessing
df['tokens'] = df['message'].apply(preprocess_text)

# Train-test split (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Print sizes to verify
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# (Optional) Save splits
train_df.to_csv("sms_train.csv", index=False)
test_df.to_csv("sms_test.csv", index=False)
