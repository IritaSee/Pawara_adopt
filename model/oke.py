import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout, Concatenate

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Create sample dog data with more detailed descriptions
dog_data = [
    {"id": 1, "name": "Golden Retriever", "description": "Friendly, intelligent, and devoted. Golden Retrievers are excellent family companions with a gentle temperament and love for play. Great with children and other pets, they need regular exercise and grooming for their thick coats."},
    {"id": 2, "name": "German Shepherd", "description": "Confident, courageous and smart. German Shepherds are working dogs excelling in roles from police to service work. They are loyal protectors with strong territorial instincts and need consistent training and mental stimulation."},
    {"id": 3, "name": "Bulldog", "description": "Docile, willful, and friendly. Bulldogs are known for their loose-jointed, shuffling gait and massive, short-faced head. They're calm companions requiring minimal exercise but need special care for skin folds and sensitivity to heat."},
    {"id": 4, "name": "Beagle", "description": "Curious, friendly, and merry. Beagles are scent hounds with great tracking instincts and energy. Perfect for active families, they love to explore but can be stubborn during training and have a tendency to howl."},
    {"id": 5, "name": "Poodle", "description": "Intelligent, active, and elegant. Poodles come in three sizes (standard, miniature, toy) and excel in obedience training. They have hypoallergenic coats requiring professional grooming and are excellent for people with allergies."},
    {"id": 6, "name": "Labrador Retriever", "description": "Outgoing, gentle, and eager to please. Labradors are athletic and excellent with families. They're versatile working dogs, great swimmers, and make perfect companions for outdoor activities, but need plenty of exercise to manage their high energy."},
    {"id": 7, "name": "Siberian Husky", "description": "Mischievous, loyal, and outgoing. Huskies are built for cold weather and have high energy levels. Natural pack dogs who need vigorous daily exercise, they're known for their stunning ice-blue eyes and tendency to escape when bored."},
    {"id": 8, "name": "Pug", "description": "Charming, loving, and mischievous. Pugs are companion dogs with distinctive wrinkled faces. They're apartment-friendly with moderate exercise needs, but prone to breathing issues due to their flat faces and can struggle in hot weather."},
    {"id": 9, "name": "Border Collie", "description": "Intelligent, energetic, and alert. Border Collies are herding dogs known for their problem-solving abilities. Considered the smartest dog breed, they require intense mental and physical stimulation and excel in agility competitions."},
    {"id": 10, "name": "Shih Tzu", "description": "Affectionate, playful, and outgoing. Shih Tzus are companion dogs with long, flowing coats. Bred as palace companions in China, they're excellent lap dogs who enjoy short walks but require daily grooming to maintain their luxurious coats."},
    {"id": 11, "name": "Dachshund", "description": "Clever, stubborn, and curious. Dachshunds have elongated bodies and short legs designed for hunting badgers. Available in smooth, wirehaired, and longhaired varieties, they're bold but can develop back problems without proper care."},
    {"id": 12, "name": "Chihuahua", "description": "Charming, graceful, and sassy. Chihuahuas are the smallest dog breed but have enormous personalities. They form strong bonds with one person, are highly portable, and can live up to 20 years despite being somewhat fragile physically."},
    {"id": 13, "name": "Boxer", "description": "Fun-loving, bright, and active. Boxers are playful and protective medium-sized dogs. Patient with children but energetic enough for vigorous play, they maintain puppy-like behavior well into adulthood and have distinctive expressive faces."},
    {"id": 14, "name": "Australian Shepherd", "description": "Intelligent, work-oriented, and exuberant. Australian Shepherds excel in herding and have stunning multicolored coats. They require extensive exercise and mental challenges, often displaying strong herding instincts even with family members."},
    {"id": 15, "name": "Great Dane", "description": "Friendly, patient, and dependable. Great Danes are gentle giants with elegant appearance. Despite their massive size, they're good apartment dogs due to their calm indoor nature, though they need room to stretch and regular moderate exercise."}
]

# Create a DataFrame for easier handling
df = pd.DataFrame(dog_data)

# More comprehensive text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords - but keep some important descriptive words
    stop_words = set(stopwords.words('english'))
    # Remove these words from stopwords as they might be important for dog descriptions
    important_words = {'small', 'large', 'big', 'tiny', 'active', 'calm', 'quiet', 'loud', 'not', 'no', 'good', 'bad'}
    for word in important_words:
        if word in stop_words:
            stop_words.remove(word)
    
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing to the descriptions
df['processed_description'] = df['description'].apply(preprocess_text)

# Create a combined field with name and description for better search results
# Give more weight to the breed name by repeating it
df['combined_text'] = df['name'] + ' ' + df['name'] + ' ' + df['processed_description']

# Create TF-IDF vectorizer for more accurate text matching
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

# Tokenize the text for deep learning model
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 150  # Increased for longer descriptions
EMBEDDING_DIM = 128

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df['combined_text'])
sequences = tokenizer.texts_to_sequences(df['combined_text'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Build an improved embedding model with attention mechanism
def build_improved_embedding_model(word_index, embedding_dim, max_seq_length):
    # Text input
    input_layer = Input(shape=(max_seq_length,))
    
    # Embedding layer
    embedding_layer = Embedding(len(word_index) + 1,
                  embedding_dim,
                  input_length=max_seq_length)(input_layer)
    
    # Global average pooling
    avg_pool = GlobalAveragePooling1D()(embedding_layer)
    
    # Multi-layer perceptron
    x = Dense(128, activation='relu')(avg_pool)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # Output layer
    output_layer = Dense(embedding_dim, activation='linear')(x)
    
    # Build and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mse', optimizer='adam')
    
    return model

# Create the model
embedding_model = build_improved_embedding_model(word_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

# Generate embeddings for all dogs in the database
dog_embeddings = embedding_model.predict(padded_sequences)

# Create an expanded synonym dictionary to improve matching
synonyms = {
    'friendly': ['sociable', 'amiable', 'kind', 'nice', 'warm', 'approachable', 'welcoming', 'cordial'],
    'intelligent': ['smart', 'clever', 'bright', 'wise', 'brilliant', 'sharp', 'brainy', 'astute'],
    'energetic': ['active', 'lively', 'spirited', 'vigorous', 'dynamic', 'peppy', 'enthusiastic', 'hyperactive'],
    'loyal': ['faithful', 'devoted', 'dedicated', 'committed', 'true', 'steadfast', 'reliable', 'trustworthy'],
    'playful': ['fun', 'jolly', 'cheerful', 'merry', 'joyful', 'frolicsome', 'exuberant', 'carefree'],
    'gentle': ['mild', 'tender', 'kind', 'soft', 'calm', 'docile', 'peaceful', 'delicate'],
    'protective': ['defensive', 'guarding', 'watchful', 'vigilant', 'careful', 'guardian', 'territorial', 'shielding'],
    'small': ['tiny', 'little', 'compact', 'miniature', 'petite', 'diminutive', 'teacup', 'pocket-sized'],
    'large': ['big', 'huge', 'gigantic', 'enormous', 'sizable', 'substantial', 'massive', 'giant'],
    'family': ['household', 'home', 'children', 'kids', 'domestic', 'relatives', 'parents', 'family-friendly'],
    'apartment': ['flat', 'condo', 'condominium', 'small space', 'living space', 'indoor', 'urban'],
    'outdoor': ['outside', 'active', 'adventurous', 'nature', 'hiking', 'exercise', 'walking', 'running'],
    'quiet': ['calm', 'silent', 'peaceful', 'tranquil', 'serene', 'composed', 'gentle', 'relaxed'],
    'vocal': ['loud', 'noisy', 'talkative', 'barking', 'howling', 'yappy', 'chatty', 'communicative'],
    'grooming': ['brushing', 'haircut', 'maintenance', 'coat care', 'shedding', 'fur', 'hair', 'clipping'],
    'training': ['obedience', 'discipline', 'teaching', 'learning', 'commands', 'instruction', 'education', 'skilled'],
    'hunting': ['prey', 'tracking', 'scenting', 'chasing', 'sporting', 'retrieving', 'catching', 'field work'],
    'herding': ['gathering', 'controlling', 'rounding up', 'shepherding', 'working', 'sheep', 'cattle', 'livestock'],
    'guard': ['protect', 'watch', 'security', 'defender', 'patrol', 'vigilant', 'alert', 'sentry'],
    'companion': ['friend', 'buddy', 'pal', 'pet', 'partner', 'lap dog', 'emotional support', 'therapy'],
    'child': ['kid', 'toddler', 'baby', 'children', 'young', 'family', 'gentle with', 'tolerant'],
    'athletic': ['sporty', 'agile', 'fit', 'active', 'nimble', 'energetic', 'physical', 'coordinated'],
    'healthy': ['robust', 'strong', 'hearty', 'well', 'fit', 'sound', 'sturdy', 'disease-resistant'],
    'lazy': ['relaxed', 'laid-back', 'casual', 'easy-going', 'chill', 'calm', 'low-energy', 'sedentary'],
    'hypoallergenic': ['allergy-friendly', 'non-shedding', 'low-dander', 'allergy', 'sensitive', 'sneeze-free']
}

# Function to expand search terms with synonyms - improved to handle multi-word terms
def expand_with_synonyms(query):
    words = query.lower().split()
    expanded_query = []
    
    for word in words:
        expanded_query.append(word)
        if word in synonyms:
            expanded_query.extend(synonyms[word])
    
    # Also check for multi-word synonyms
    for key in synonyms:
        if key in query.lower() and key not in words:  # Check if multi-word key exists in query
            expanded_query.extend(synonyms[key])
    
    return ' '.join(expanded_query)

# Improved search function using both TF-IDF and embedding similarity
def search_dogs(query, top_n=3):
    # Expand query with synonyms
    expanded_query = expand_with_synonyms(query)
    
    # Preprocess the query
    processed_query = preprocess_text(expanded_query)
    
    # Get TF-IDF vector for the query
    query_tfidf = tfidf_vectorizer.transform([processed_query])
    
    # Calculate TF-IDF similarity
    tfidf_similarities = cosine_similarity(query_tfidf, tfidf_matrix)[0]
    
    # Convert query to sequence for deep learning model
    query_sequence = tokenizer.texts_to_sequences([processed_query])
    padded_query = pad_sequences(query_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Get embedding for the query
    query_embedding = embedding_model.predict(padded_query)
    
    # Calculate embedding similarity
    embedding_similarities = cosine_similarity(query_embedding, dog_embeddings)[0]
    
    # Combine similarities (weighted average)
    # Give more weight to TF-IDF for specific term matching
    combined_similarities = 0.7 * tfidf_similarities + 0.3 * embedding_similarities
    
    # Get top N matches
    top_indices = combined_similarities.argsort()[-top_n:][::-1]
    
    # Create result list with explanation of match
    results = []
    for idx in top_indices:
        if combined_similarities[idx] > 0.1:  # Only include somewhat relevant results
            # Extract matching keywords for explanation
            query_terms = set(processed_query.split())
            dog_terms = set(df.iloc[idx]['processed_description'].split())
            matching_terms = query_terms.intersection(dog_terms)
            
            # Create explanation based on matching terms
            if matching_terms:
                match_explanation = f"Matched on: {', '.join(matching_terms)}"
            else:
                match_explanation = "Matched based on related traits"
            
            results.append({
                'id': df.iloc[idx]['id'],
                'name': df.iloc[idx]['name'],
                'description': df.iloc[idx]['description'],
                'similarity_score': combined_similarities[idx],
                'match_explanation': match_explanation
            })
    
    return results if results else [{"message": "Maaf, tidak ada hasil yang cocok dengan kriteria pencarian Anda."}]

# Example searches to test the system
def test_search_examples():
    test_queries = [
        "anjing keluarga yang ramah",
        "anjing cerdas untuk bekerja",
        "anjing kecil untuk apartemen",
        "anjing energik untuk aktivitas luar ruangan",
        "cocok untuk anak-anak",
        "anjing dengan perawatan bulu minimal",
        "anjing pemburu",
        "anjing yang tidak banyak menggonggong",
        "anjing besar tapi tenang"
    ]
    
    for query in test_queries:
        print(f"\nPencarian: '{query}'")
        results = search_dogs(query)
        
        if "message" in results[0]:
            print(results[0]["message"])
        else:
            for i, result in enumerate(results):
                print(f"{i+1}. {result['name']} (Skor: {result['similarity_score']:.4f})")
                print(f"   {result['description']}")
                print(f"   {result['match_explanation']}")

# Run the test searches
test_search_examples()

# Interactive search function with improved UI
def interactive_search():
    print("\n===== PENCARIAN ANJING YANG SESUAI =====")
    print("Masukkan kriteria anjing yang Anda inginkan.")
    print("Contoh: 'anjing keluarga yang tenang' atau 'anjing kecil untuk apartemen'")
    print("Ketik 'keluar' untuk mengakhiri program.\n")
    
    while True:
        query = input("\nMasukkan kriteria pencarian anjing: ")
        if query.lower() in ['keluar', 'exit', 'quit']:
            print("Terima kasih telah menggunakan sistem pencarian anjing!")
            break
            
        results = search_dogs(query)
        
        if "message" in results[0]:
            print(results[0]["message"])
            print("Coba gunakan kata kunci yang berbeda atau lebih umum.")
        else:
            print(f"\nHasil pencarian untuk '{query}':")
            print("-" * 50)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['name']} (Skor: {result['similarity_score']:.4f})")
                print(f"   {result['description']}")
                print(f"   {result['match_explanation']}")
                print("-" * 50)

# Start the interactive search
interactive_search()