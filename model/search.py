import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Parameter untuk model
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10

# Data contoh anjing (dalam keadaan nyata, Anda perlu dataset yang lebih besar)
dog_data = [
    {"id": 1, "breed": "Golden Retriever", "description": "Anjing ramah dan cerdas dengan bulu keemasan yang indah. Sangat cocok untuk keluarga dan suka berenang."},
    {"id": 2, "breed": "German Shepherd", "description": "Anjing setia dan pemberani. Sering digunakan sebagai anjing polisi atau penjaga karena kecerdasannya."},
    {"id": 3, "breed": "Poodle", "description": "Anjing yang elegan dengan bulu keriting. Dikenal karena kecerdasannya dan mudah dilatih."},
    {"id": 4, "breed": "Bulldog", "description": "Anjing kompak dengan wajah keriput. Tenang dan ramah tetapi bisa keras kepala."},
    {"id": 5, "breed": "Siberian Husky", "description": "Anjing energik dengan mata biru mencolok. Suka berlari dan cocok untuk iklim dingin."},
    {"id": 6, "breed": "Labrador Retriever", "description": "Anjing yang ramah dan penuh energi. Sering digunakan sebagai anjing penuntun atau pencari."},
    {"id": 7, "breed": "Chihuahua", "description": "Anjing terkecil di dunia. Berani dan sangat setia pada pemiliknya meskipun ukurannya kecil."},
    {"id": 8, "breed": "Shiba Inu", "description": "Anjing dari Jepang dengan perilaku mirip kucing. Mandiri dan bersih."},
    {"id": 9, "breed": "Border Collie", "description": "Anjing paling cerdas. Suka bekerja dan membutuhkan banyak aktivitas fisik dan mental."},
    {"id": 10, "breed": "Beagle", "description": "Anjing pemburu dengan hidung yang sangat sensitif. Ramah dan suka berkelompok."}
]

# Menyiapkan data
texts = [dog["description"] for dog in dog_data]
breeds = [dog["breed"] for dog in dog_data]

# Tokenisasi teks
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Membuat model embedding
def create_embedding_model():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_layer)
    global_avg_pooling = GlobalAveragePooling1D()(embedding_layer)
    dense1 = Dense(64, activation='relu')(global_avg_pooling)
    dense2 = Dense(32, activation='relu')(dense1)
    model = Model(inputs=input_layer, outputs=dense2)
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Inisialisasi dan latih model (simulasi pelatihan)
model = create_embedding_model()
print("Model summary:")
model.summary()

# Dalam aplikasi nyata, Anda akan melatih model dengan data pelatihan yang sesuai
# Misalnya: model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Fungsi untuk mendapatkan embedding dari deskripsi
def get_embedding(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return model.predict(padded)[0]

# Menghasilkan embedding untuk semua anjing dalam dataset
dog_embeddings = []
print("Generating embeddings for all dogs...")
for dog in tqdm(dog_data):
    embedding = get_embedding(dog["description"])
    dog_embeddings.append(embedding)
dog_embeddings = np.array(dog_embeddings)

# Fungsi pencarian
def search_dogs(query, top_n=3):
    # Mendapatkan embedding untuk query
    query_embedding = get_embedding(query)
    
    # Menghitung cosine similarity
    similarities = cosine_similarity([query_embedding], dog_embeddings)[0]
    
    # Mendapatkan indeks teratas
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            "breed": dog_data[idx]["breed"],
            "description": dog_data[idx]["description"],
            "similarity": similarities[idx]
        })
    
    return results

# Contoh penggunaan
def demo_search():
    # Beberapa contoh query pencarian
    search_queries = [
        "anjing untuk keluarga yang ramah",
        "anjing kecil yang setia",
        "anjing cerdas untuk dilatih",
        "anjing untuk daerah dingin"
    ]
    
    print("\nDEMO PENCARIAN ANJING")
    print("=====================")
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        results = search_dogs(query, top_n=3)
        
        print("Hasil Pencarian:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['breed']} (Kemiripan: {result['similarity']:.4f})")
            print(f"   {result['description']}")

# Jalankan demo
if __name__ == "__main__":
    demo_search()

# Kelas untuk aplikasi pencarian anjing
class DogSearchEngine:
    def __init__(self, model_path=None):
        self.dog_data = dog_data
        self.tokenizer = tokenizer
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = model
            
        # Pre-compute embeddings
        self.dog_embeddings = []
        for dog in dog_data:
            embedding = self.get_embedding(dog["description"])
            self.dog_embeddings.append(embedding)
        self.dog_embeddings = np.array(self.dog_embeddings)
    
    def get_embedding(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        return self.model.predict(padded, verbose=0)[0]
    
    def search(self, query, top_n=3):
        # Mendapatkan embedding untuk query
        query_embedding = self.get_embedding(query)
        
        # Menghitung cosine similarity
        similarities = cosine_similarity([query_embedding], self.dog_embeddings)[0]
        
        # Mendapatkan indeks teratas
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append({
                "breed": self.dog_data[idx]["breed"],
                "description": self.dog_data[idx]["description"],
                "similarity": similarities[idx]
            })
        
        return results
    
    def save_model(self, path):
        self.model.save(path)
        print(f"Model berhasil disimpan di {path}")

# Contoh penggunaan search engine
# search_engine = DogSearchEngine()
# results = search_engine.search("anjing yang baik untuk keluarga dengan anak-anak")
# for i, result in enumerate(results, 1):
#     print(f"{i}. {result['breed']} - Similarity: {result['similarity']:.4f}")