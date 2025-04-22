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
    {"id": 1, "breed": "Golden Retriever", "description": "Golden Retriever ini, yang kini dipanggil Sunny, adalah anjing ramah dan cerdas dengan bulu keemasan yang indah. Ia ditemukan di tepi sungai dalam kondisi basah kuyup dan gemetar, diduga telah ditinggalkan selama beberapa hari. Seorang pemancing menyelamatkannya dan membawanya ke shelter. Setelah dirawat, Sunny menunjukkan sifat lembut dan penuh kasih sayang. Ia sudah tinggal di shelter selama 3 bulan dan kini dalam kondisi sehat, senang bermain air dan menyambut siapa saja dengan ekor bergoyang."},
    {"id": 2, "breed": "German Shepherd", "description": "Anjing German Shepherd ini bernama Rex, terkenal karena keberanian dan kesetiaannya. Ia diselamatkan dari lingkungan penuh kekerasan dan trauma berat. Saat pertama kali datang ke shelter 5 bulan lalu, Rex sangat defensif. Tapi dengan kesabaran, ia kini sudah mulai percaya pada manusia lagi. Rex sudah mulai mengikuti pelatihan dasar dan dalam kondisi fisik yang prima, siap untuk rumah yang penuh kasih."},
    {"id": 3, "breed": "Poodle", "description": "Poodle elegan ini diberi nama Coco. Ia ditemukan berkeliaran di taman kota, dengan bulu kusut dan mata sedih. Tidak ada kalung, tidak ada chip, hanya seekor anjing kecil yang tersesat. Setelah 2 bulan di shelter, Coco sudah tampil cantik setelah grooming dan kini sangat ceria. Ia sangat suka duduk di pangkuan dan menjadi pusat perhatian di ruang adopsi."},
    {"id": 4, "breed": "Bulldog", "description": "Bulldog ini, yang dinamai Bruno, diselamatkan dari tempat penampungan ilegal dalam kondisi yang menyedihkan. Ia kurus dan penuh luka kecil saat pertama kali ditemukan. Kini, setelah 4 bulan di shelter, Bruno telah pulih sepenuhnya dan suka tidur siang di bawah sinar matahari. Meskipun keras kepala, ia sangat manja pada orang-orang yang ia percaya."},
    {"id": 5, "breed": "Siberian Husky", "description": "Husky bermata biru ini kini dikenal dengan nama Frost. Ia ditemukan saat badai salju di pinggiran kota, berlindung di bawah mobil. Setelah 6 minggu di shelter, bulunya kembali mengkilap dan semangatnya membara. Frost sangat energik, membutuhkan ruang luas, dan senang berlarian di area terbuka bersama anjing lain."},
    {"id": 6, "breed": "Labrador Retriever", "description": "Labrador ceria ini dipanggil Buddy. Ia ditinggalkan oleh pemilik lamanya saat pindah rumah. Seorang tetangga menemukannya kelaparan dan sendirian, lalu membawanya ke shelter. Sudah 2 bulan Buddy tinggal di sini dan menjadi favorit relawan karena sifatnya yang ramah. Ia sehat, suka berjalan pagi, dan sangat cocok untuk keluarga dengan anak-anak."},
    {"id": 7, "breed": "Chihuahua", "description": "Chihuahua kecil pemberani ini bernama Lilo. Ia ditemukan di dalam kardus di pinggir pasar saat hujan. Lilo awalnya sangat ketakutan, tapi setelah 1 bulan di shelter, ia menjadi anjing yang manja dan protektif terhadap orang yang ia kenal. Meski kecil, Lilo sangat vokal dan suka digendong."},
    {"id": 8, "breed": "Shiba Inu", "description": "Shiba Inu ini diberi nama Yuki. Ia nyaris tertabrak mobil di jalan raya, tapi berhasil diselamatkan oleh pengemudi yang sigap. Kini setelah 3 bulan di shelter, Yuki menjadi anjing yang tenang dan penuh gaya. Ia sangat menjaga kebersihannya dan hanya makan dari mangkuk yang bersih. Cocok untuk pemilik yang sabar dan memahami sifat Shiba."},
    {"id": 9, "breed": "Border Collie", "description": "Border Collie ini bernama Scout. Ia ditemukan terkunci di gudang kosong tanpa makanan. Berkat naluri bertahan hidupnya dan bantuan warga, Scout selamat. Setelah 2,5 bulan di shelter, ia berkembang menjadi anjing aktif dan cerdas yang suka teka-teki dan permainan interaktif. Scout dalam kondisi prima dan siap untuk rumah yang aktif secara fisik dan mental."},
    {"id": 10, "breed": "Beagle", "description": "Beagle lucu ini bernama Toby. Ia mengikuti aroma makanan ke restoran dan langsung menarik perhatian pemilik restoran. Setelah dicek, chip-nya tidak aktif selama bertahun-tahun. Di shelter selama 1,5 bulan terakhir, Toby menunjukkan sifat ceria, suka bermain, dan sangat senang jalan-jalan pagi. Ia sehat dan suka ditemani anjing lain."}
]

dog_data += [
    {"id": 11, "breed": "Dalmatian", "description": "Diberi nama Spotty karena bintik-bintiknya yang unik, Dalmatian ini ditemukan bersembunyi di kolong truk makanan keliling saat badai. Ia tampak bingung dan kelelahan. Seorang sopir yang baik hati memberikan makanan dan menghubungi shelter. Kini, setelah 2 bulan, Spotty berubah jadi anjing ceria dan enerjik yang suka bermain bola dan berlari mengelilingi halaman."},
    {"id": 12, "breed": "Cocker Spaniel", "description": "Cocker Spaniel bernama Bella ini ditemukan di taman bermain, duduk di dekat ayunan seakan sedang menunggu seseorang yang tak kunjung datang. Anak-anak memberi tahu orang tuanya yang langsung menghubungi shelter. Sudah 3 bulan Bella tinggal di sini, dan kini menjadi anjing manis yang suka dielus dan selalu ingin dekat dengan manusia."},
    {"id": 13, "breed": "Great Dane", "description": "Namanya Titan, dan ukurannya memang sesuai dengan namanya. Great Dane ini ditemukan berjalan perlahan di jalan tol, dehidrasi dan kelaparan. Polisi lalu lintas bekerja sama dengan tim penyelamat untuk mengamankannya. Kini Titan sudah pulih, tenang, dan sangat lembut meskipun tubuhnya besar. Ia butuh rumah dengan ruang yang luas dan hati yang besar."},
    {"id": 14, "breed": "Pomeranian", "description": "Fluffy, si Pomeranian mungil dengan bulu seperti kapas, ditemukan di keranjang belanja supermarket. Tidak ada yang tahu bagaimana ia bisa di sana. Setelah 1 bulan di shelter, Fluffy menunjukkan sifat ceria, suka bermain, dan sangat pintar meniru ekspresi manusia. Cocok untuk apartemen dan pemilik yang suka dimanja balik oleh peliharaannya."},
    {"id": 15, "breed": "Rottweiler", "description": "Bruno adalah Rottweiler kuat yang dulunya dijadikan penjaga gudang, tetapi ditinggalkan begitu saja saat bangunan itu ditutup. Ia diselamatkan dalam kondisi kurus dan penuh luka kecil. Kini, setelah 4 bulan dirawat, Bruno menunjukkan sisi lembutnya, sangat setia dan patuh. Ia butuh pemilik yang berpengalaman dan siap memberi perhatian penuh."},
    {"id": 16, "breed": "Yorkshire Terrier", "description": "Mini, si Yorkie mungil, ditemukan di dalam tas bekas yang ditinggalkan di halte bus. Beratnya hanya 2,5 kg saat ditemukan. Kini, setelah 6 minggu, ia sehat dan aktif, sangat suka digendong dan berjalan di atas karpet. Mini sangat cocok untuk orang tua atau keluarga dengan anak kecil yang lembut."},
    {"id": 17, "breed": "Akita Inu", "description": "Hachiko, dinamai dari legenda Jepang, ditemukan duduk di depan rumah kosong selama berhari-hari, seolah menunggu seseorang. Tetangga setempat yang prihatin melaporkannya ke shelter. Akita ini tenang dan penuh wibawa, sangat setia dan hanya dekat dengan orang yang benar-benar ia percayai. Setelah 3 bulan, ia siap untuk rumah yang sabar dan stabil."},
    {"id": 18, "breed": "Basset Hound", "description": "Anjing bertelinga panjang ini dipanggil Droopy. Ia ditemukan berjalan pelan di tengah hujan dengan tatapan sedih, penuh kutu dan lumpur. Kini setelah 2 bulan di shelter, Droopy bersih, sehat, dan suka tidur di pojok hangat. Ia tenang dan cocok untuk rumah yang santai."},
    {"id": 19, "breed": "Australian Shepherd", "description": "Skye si Australian Shepherd ditemukan berlari tanpa arah di perbukitan, mungkin kehilangan jejak saat mengikuti suara petasan. Sangat energik dan cerdas, ia cepat belajar trik baru dan suka tantangan. Sudah 1 bulan di shelter dan cocok untuk pemilik aktif yang suka hiking atau olahraga luar ruangan."},
    {"id": 20, "breed": "Maltese", "description": "Snowy, Maltese putih bersih ini ditemukan di jalan setelah badai besar, bulunya penuh lumpur dan terikat simpul. Kini, setelah 5 minggu perawatan intensif, Snowy kembali bersinar dan suka duduk di pangkuan siapa pun yang memberinya perhatian. Sangat manja, lembut, dan cocok untuk lingkungan dalam rumah yang tenang."}
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