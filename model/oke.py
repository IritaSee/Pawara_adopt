import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Create sample dog data
dog_data = [
    {"id": 1, "name": "Golden Retriever", "description": "Friendly, intelligent, and devoted. Golden Retrievers are excellent family companions with a gentle temperament and love for play."},
    {"id": 2, "name": "German Shepherd", "description": "Confident, courageous and smart. German Shepherds are working dogs excelling in roles from police to service work."},
    {"id": 3, "name": "Bulldog", "description": "Docile, willful, and friendly. Bulldogs are known for their loose-jointed, shuffling gait and massive, short-faced head."},
    {"id": 4, "name": "Beagle", "description": "Curious, friendly, and merry. Beagles are scent hounds with great tracking instincts and energy."},
    {"id": 5, "name": "Poodle", "description": "Intelligent, active, and elegant. Poodles come in three sizes and excel in obedience training."},
    {"id": 6, "name": "Labrador Retriever", "description": "Outgoing, gentle, and eager to please. Labradors are athletic and excellent with families."},
    {"id": 7, "name": "Siberian Husky", "description": "Mischievous, loyal, and outgoing. Huskies are built for cold weather and have high energy levels."},
    {"id": 8, "name": "Pug", "description": "Charming, loving, and mischievous. Pugs are companion dogs with distinctive wrinkled faces."},
    {"id": 9, "name": "Border Collie", "description": "Intelligent, energetic, and alert. Border Collies are herding dogs known for their problem-solving abilities."},
    {"id": 10, "name": "Shih Tzu", "description": "Affectionate, playful, and outgoing. Shih Tzus are companion dogs with long, flowing coats."}
]

# Create a DataFrame for easier handling
df = pd.DataFrame(dog_data)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing to the descriptions
df['processed_description'] = df['description'].apply(preprocess_text)

# Create a combined field with name and description for better search results
df['combined_text'] = df['name'] + ' ' + df['processed_description']

# Tokenize the text
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df['combined_text'])
sequences = tokenizer.texts_to_sequences(df['combined_text'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Build the embedding model
def build_embedding_model(word_index, embedding_dim, max_seq_length):
    input_layer = Input(shape=(max_seq_length,))
    
    # Embedding layer
    x = Embedding(len(word_index) + 1,
                  embedding_dim,
                  input_length=max_seq_length)(input_layer)
    
    # Global average pooling to convert sequences to fixed-length vectors
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers for learning
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # Output layer
    output_layer = Dense(embedding_dim, activation='linear')(x)
    
    # Build and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='cosine_similarity', optimizer='adam')
    
    return model

# Create and train the model
embedding_model = build_embedding_model(word_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

# Since we don't have actual training data, we'll just build the model but not train it
# In a real scenario, you would train with paired data (queries and relevant descriptions)
# For this example, we'll use the model as a feature extractor

# Generate embeddings for all dogs in the database
dog_embeddings = embedding_model.predict(padded_sequences)

# Create a synonym dictionary to expand search terms
synonyms = {
    'friendly': ['sociable', 'amiable', 'kind', 'nice', 'warm'],
    'intelligent': ['smart', 'clever', 'bright', 'wise', 'brilliant'],
    'energetic': ['active', 'lively', 'spirited', 'vigorous', 'dynamic'],
    'loyal': ['faithful', 'devoted', 'dedicated', 'committed', 'true'],
    'playful': ['fun', 'jolly', 'cheerful', 'merry', 'joyful'],
    'gentle': ['mild', 'tender', 'kind', 'soft', 'calm'],
    'protective': ['defensive', 'guarding', 'watchful', 'vigilant', 'careful'],
    'small': ['tiny', 'little', 'compact', 'miniature', 'petite'],
    'large': ['big', 'huge', 'gigantic', 'enormous', 'sizable'],
    'family': ['household', 'home', 'children', 'kids', 'domestic']
}

# Function to expand search terms with synonyms
def expand_with_synonyms(query):
    words = query.lower().split()
    expanded_words = []
    
    for word in words:
        expanded_words.append(word)
        if word in synonyms:
            expanded_words.extend(synonyms[word])
    
    return ' '.join(expanded_words)

# Function to search for dogs using the embedding model
def search_dogs(query, top_n=3):
    # Expand query with synonyms
    expanded_query = expand_with_synonyms(query)
    
    # Preprocess the query
    processed_query = preprocess_text(expanded_query)
    
    # Convert query to sequence
    query_sequence = tokenizer.texts_to_sequences([processed_query])
    padded_query = pad_sequences(query_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Get embedding for the query
    query_embedding = embedding_model.predict(padded_query)
    
    # Calculate cosine similarity between query and all dog embeddings
    similarities = cosine_similarity(query_embedding, dog_embeddings)[0]
    
    # Get top N matches
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Create result list
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only include results with positive similarity
            results.append({
                'id': df.iloc[idx]['id'],
                'name': df.iloc[idx]['name'],
                'description': df.iloc[idx]['description'],
                'similarity_score': similarities[idx]
            })
    
    return results if results else [{"message": "Maaf tidak ada hasil yang cocok."}]

# Example searches to test the system
def test_search_examples():
    test_queries = [
        "friendly family dog",
        "intelligent working dog",
        "small companion dog",
        "energetic outdoor dog",
        "dog good with kids",
        "shepherd dog",
        "hunting dog",
        "nonsense query that won't match anything"
    ]
    
    for query in test_queries:
        print(f"\nSearch Query: '{query}'")
        results = search_dogs(query)
        
        if "message" in results[0]:
            print(results[0]["message"])
        else:
            for i, result in enumerate(results):
                print(f"{i+1}. {result['name']} (Score: {result['similarity_score']:.4f})")
                print(f"   {result['description']}")

# Run the test searches
test_search_examples()

# Interactive search function
def interactive_search():
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        results = search_dogs(query)
        
        if "message" in results[0]:
            print(results[0]["message"])
        else:
            print(f"\nResults for '{query}':")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['name']} (Score: {result['similarity_score']:.4f})")
                print(f"   {result['description']}")

interactive_search()