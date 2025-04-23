import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
# Download the 'punkt_tab' resource
nltk.download('punkt_tab') # This line is added to download the missing 'punkt_tab' resource

# Load dataset
df = pd.read_csv("sms spamer\SMSSpamCollection.csv", sep='\t', names=['label', 'message'])

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return cleaned_tokens

# Apply preprocessing
df['tokens'] = df['message'].apply(preprocess_text)

# Display the first few rows
print(df.head())
