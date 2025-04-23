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
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import load_model
import os

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Create dog data from the provided descriptions
dog_data = [
    {"id": 1, "name": "Hitam", "traits": "kiul", "age": 4, "gender": "cewek", 
     "description": "Hai, aku HITAM. Aku anjing yang suka tidur. Sebenarnya, aku sudah memiliki tuan yang merupakan bule Australia. Namun, saat ini ia sedang pulang kampung ke daerahnya. Sehingga aku dititipkan disini, karena ia tidak bisa membawa ku ikut bersamanya. Aku merasa bosan & kesepian disini. Tidak ada teman teman yang mengajak ku untuk bermain. Biasanya, aku selalu ditemani oleh tuanku dan selalu diajak bermain olehnya. Aku sangat merindukan tuanku. Apakah kamu mau untuk mengganti peran tuanku sementara waktu?"},
    
    {"id": 2, "name": "Opi", "traits": "ketek", "age": 4 , "gender": "cewe", 
     "description": "Namaku Opi, si kecil lincah yang lahir langsung di animal shelter ini. Ibuku diselamatkan saat sedang hamil besar—badannya kurus, tapi matanya penuh harapan. Beberapa hari setelah ia datang, aku dan saudara-saudaraku lahir di tempat yang aman dan hangat ini. Dari pertama kali membuka mata, aku langsung penasaran dengan dunia. Kata para penjaga, sejak kecil aku sudah paling aktif di antara yang lain. Aku suka menggigit ujung selimut, mengejar bayangan, dan tak pernah bisa diam. “Si ketek satu ini pasti calon pelari maraton,” kata salah satu kakak penjaga sambil tertawa. Aku memang begitu—selalu semangat, selalu ingin tahu, dan selalu bikin suasana jadi ramai. Aku lahir di shelter ini, tapi aku yakin dunia di luar sana luas dan seru. Mungkin, kamu adalah bagian dari dunia itu."},
    
    {"id": 3, "name": "Elu", "traits": "ketek", "age": 3.5 , "gender": "cowok", 
     "description": "Aku dulu punya rumah. Ada tangan hangat yang mengelusku, suara lembut yang memanggil namaku. Tapi suatu hari, kami naik mobil, lalu mereka pergi tanpa kembali. Aku menunggu. Lama. Lapar. Dingin. Tak ada lagi panggilan itu. Aku tidur di jalanan, hujan turun. Tubuhku gemetar. Banyak orang yang melihatku, namun tak ada seorangpun yang memperhatikanku. Seolah olah, aku sama sekali tidak ada di dunia ini. Lalu seseorang datang, wanita dengan payung merah dan tatapan sedih. Dia membungkusku, membawaku pergi. Aku tak punya tenaga untuk takut. Di tempat baru, aku diperhatikan, dan diberi makan. Aku masih ingat rasa ditinggalkan. Tapi sekarang, aku juga tahu rasa diselamatkan. Dan aku sedang belajar percaya lagi."},
    
    {"id": 4, "name": "Lara", "traits": "ketek", "age": 3.5 , "gender": "cewe", 
     "description": "Aku awalnya dirawat oleh sebuah keluarga. Namun, entah kenapa tiba tiba mereka membuangku begitu saja. Aku ditinggalkan di sekitar jalan Pura Batur menuju Pura Besakih. Aku kira aku hanya ditinggalkan sebentar saja, jadi aku tetap menunggu dan berharap mereka akan kembali menjemputku secepatnya. Namun… aku salah. Mereka benar benar meninggalkan ku, sendirian, sepi, dan gelap. Beberapa hari berlalu, kondisi ku sangat menyedihkan. Badan penuh kotoran, bulu yang mulai rontok, dan tulang yang terlihat mencetak badan saking kurusnya aku. Akhirnya aku memutuskan untuk menyerah, aku sudah sangat menanti nanti ajal untuk menjemputku. Tepat sebelum ajal mendatangiku, aku ditolong oleh sebuah keluarga yang akan beribadah ke Pura Besakih. Tatapannya begitu mengasihiku ketika melihat kondisiku yang begitu mengenaskan. Aku pun di rawat oleh keluarga itu, namun mereka tidak bisa merawatku secara terus menerus karena kucing yang mereka pelihara selalu saja mencari masalah pada ku. Akhirnya aku pun dibawa menuju tempat penampungan. Apakah kalian tidak tertarik untuk merawatku?"},
    
    {"id": 5, "name": "Tara", "traits": "paksa", "age": 4 , "gender": "cewe", 
     "description": "Aku diselamatkan dari tempat yang sangat keras. Sejak kecil, aku sudah harus belajar untuk menjaga diriku sendiri. Aku kuat, aku tangguh. Aku tidak takut akan suara keras, bahkan suara petir pun tidak bisa membuatku gemetar. Saat ditemukan, aku sedang menggonggong keras, menjaga seekor anak anjing yang lebih kecil dari gangguan jalanan. Mereka menyebutku pemberani. Kini aku di sini, menunggu seseorang yang bisa melihat keberanianku bukan sebagai ancaman, tapi sebagai kekuatan."},
    
    {"id": 6, "name": "Putih", "traits": "paksa", "age": 3 , "gender": "cowo", 
     "description": "Hai, namaku Putih. Aku diberi nama Putih ketika aku sampai di tempat ini. Dulu, aku dirawat oleh seorang anak kecil yang juga menjadi sahabatku. Ia selalu mengajakku bermain, memandikan ku, dan selalu memberikan ku makanan. Namun, tiba tiba kami kedatangan seekor kucing kecil di rumah kami. Awalnya, tidak ada perubahan yang terjadi pada kehidupan kami. Tetapi seiring berjalan nya waktu, sahabatku tidak lagi pernah bermain denganku. Dia selalu bermain dengan kucing yang baru ia pelihara. Sedih, hatiku terasa sakit ketika sahabatku tersenyum lebih riang saat ia bermain dengan kucing itu daripada aku. Perlahan, aku mulai tersingkirkan dari dunia mereka berdua. Aku tidak pernah diajak bermain lagi. Buluku mulai terlihat kotor, karena aku tidak pernah dimandikan lagi. Pada akhirnya, mereka membuang ku karena aku dianggap tidak berguna, membawa penyakit dan hanya menghabiskan makanan. Aku pun dibawa ke tempat penampungan. Disini, aku dirawat dengan baik. Aku diberi makan & dimandikan hingga buluku terlihat putih bersih kembali."},
    
    {"id": 7, "name": "Coklat", "traits": "kiull", "age": 3 , "gender": "cewe", 
     "description": "Aku…dibuang oleh keluarga yang merawatku. Mereka sering berpergian keluar kota, jadi mereka membutuhkan anjing yang dapat menjaga rumah mereka selama mereka berpergian. Aku dibeli oleh mereka beberapa hari setelah aku lahir. Aku dirawat dengan sangat baik, dipanggil \"Pemberani\", dan sering diajak bermain. Beberapa bulan kemudian, mereka menemukan fakta bahwa aku adalah anjing betina. Semenjak saat itu, perilaku mereka berubah 180°. Mereka sama sekali tidak memperdulikan keberadaanku. Mereka tidak pernah memanggilku dan tidak pernah mengajak ku bermain lagi. Mereka mengira anjing betina tidak bisa menjaga rumah. Beberapa hari kemudian, mereka membawa anjing baru ke rumah. Hati ku terasa sakit, mengetahui posisiku sudah digantikan hanya karena jenis kelaminku. Aku pun dibuang begitu saja ke pinggir jalan. Beruntungnya beberapa jam aku berjalan dari tempatku dibuang, aku ditemukan oleh seorang pemilik penampungan. Dan begitulah kisahku hingga bisa berada di tempat ini"},
    
    {"id": 8, "name": "Mika", "traits": "paksa", "age": 2.5 , "gender": "cewe", 
     "description": "Aku diselamatkan dari jalanan yang penuh bahaya. Saat ditemukan, tubuhku dipenuhi luka dan kotoran. Aku sempat menggigit orang yang mencoba mendekat karena takut. Tapi mereka tidak menyerah. Mereka sabar, memberiku makan, dan akhirnya aku percaya. Aku bukan galak, aku hanya terbiasa bertahan. Tapi kalau kamu sabar, kamu akan melihat sisi lembutku yang tersembunyi."},
    
    {"id": 9, "name": "Jojo", "traits": "kiul", "age": 7 , "gender": "cowo", 
     "description": "Tuanku…meninggalkanku… Ups, bercanda. Tapi tuanku benar benar meninggalkanku untuk sementara waktu. Ia merupakan bule Australia yang dulu berlibur di Bali, namun sekarang dia sudah kembali ke negara asalnya. Tuanku tidak tega meninggalkanku begitu saja, jadi ia menitipkan ku di tempat ini. Aku sangat merindukan tuanku. Biasanya tuanku mengajakku untuk berkeliling"},
    
    {"id": 10, "name": "Molly", "traits": "jaya", "age": 8 , "gender": "cewe", 
     "description": "Dulu, aku kecil dan lucu. Mereka memelukku setiap hari, memanggilku \"anak pintar\", memberiku bantal empuk, mainan, dan pelukan hangat. Aku sangat mencintai mereka. Aku selalu mengibas ekor setiap kali mereka pulang. Tapi tubuhku tumbuh. Aku tak bisa lagi duduk di pangkuan. Mereka mulai marah saat aku merusak barang atau menggonggong terlalu keras. Aku mencoba menjadi anak baik, sungguh. Hari itu, mereka mengajakku naik mobil. Aku senang, kupikir kami akan bermain. Tapi mobil berhenti di tempat asing. Mereka menyuruhku turun, lalu pergi. Aku berlari mengejar, menggonggong sekuatku, tapi mereka tak kembali. Sekarang aku duduk sendiri, menunggu. Aku lapar. Dingin. Tapi yang paling menyakitkan… aku masih menunggu mereka, karena aku masih mencintai mereka. Aku menunggu di tempat itu. Sendiri, lapar, basah. Tapi aku tetap berharap mereka kembali. Sampai suatu hari, seorang wanita menemukanku. Tangannya lembut, suaranya hangat. Aku ikut saja saat dia mengangkatku. Kini aku di penampungan. Ada makanan, tempat tidur, dan kadang belaian hangat. Rasanya... sedikit seperti dicintai lagi. Bukan rumahku yang dulu. Tapi mungkin, ini awal dari rumah yang baru. Aku masih menunggu. Tapi kali ini, dengan sedikit harapan. Aku sama sekali tidak mengingat namaku. Apakah kamu ingin memberi nama baru padaku?"},
    
    {"id": 11, "name": "Naya", "traits": "kiul", "age": 3.5 , "gender": "cewe", 
     "description": "Aku diselamatkan di tengah hujan di sebuah gang sempit. Aku diam saja saat mereka menggendongku, karena aku terlalu lelah untuk melawan. Tapi sekarang aku mulai tersenyum lagi. Aku senang tidur di tempat hangat dan suka mendengarkan suara orang-orang di sekitarku. Aku kalem dan tidak rewel, cocok buat kamu yang suka ketenangan."},
    
    {"id": 12, "name": "Rika", "traits": "jaya", "age": 4 , "gender": "cewe", 
     "description": "Aku ditemukan di jalan, kurus dan kotor, tapi dengan mata yang masih penuh semangat. Mereka bilang aku seperti pejuang kecil yang tidak mau menyerah. Aku punya semangat untuk hidup, dan aku belajar cepat. Sekarang aku sehat, lincah, dan siap untuk jadi bintang di rumah barumu. Aku adalah juara yang sedang menunggu panggungku sendiri."}
]


# Create a DataFrame for easier handling
df = pd.DataFrame(dog_data)

# Add combined fields for sifat (traits), umur (age), and gender
df['combined_description'] = df.apply(lambda row: f"{row['name']} {row['traits']} {row['age']} {row['gender']} {row['description']}", axis=1)

# More comprehensive text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords - but keep some important descriptive words
    stop_words = set(stopwords.words('english'))
    # Remove these words from stopwords as they might be important for dog descriptions
    important_words = {'suka', 'bermain', 'tidur', 'sakit', 'bosan', 'kesepian', 'dibuang', 'lucu', 'kecil'}
    # for word in important_words:
    #     if word in stop_words:
    #         stop_words.remove(word)
    
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing to the descriptions
df['processed_description'] = df['combined_description'].apply(preprocess_text)

# Create TF-IDF vectorizer for more accurate text matching
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_description'])

# Tokenize the text for deep learning model
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200  # Increased for longer descriptions
EMBEDDING_DIM = 128

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df['processed_description'])
sequences = tokenizer.texts_to_sequences(df['processed_description'])
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
    model.save('main_model.h5', save_format='h5')
    print("Model saved as main_model.h5")
    
    return model

# Create or load the model
if os.path.exists('main_model.h5'):
    embedding_model = load_model('main_model.h5')
else:
    embedding_model = build_improved_embedding_model(word_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)


# Generate embeddings for all dogs in the database
dog_embeddings = embedding_model.predict(padded_sequences)

# Create an expanded synonym dictionary to improve matching - adjusted for Indonesian
synonyms = {
    'bermain': ['main', 'senang', 'gembira', 'ceria', 'aktif'],
    'cerdas': ['pintar', 'pandai', 'cakap', 'cepat belajar', 'terlatih'],
    'energik': ['aktif', 'lincah', 'semangat', 'bertenaga', 'gesit'],
    'setia': ['loyal', 'patuh', 'taat', 'tunduk', 'nurut'],
    'ceria': ['senang', 'gembira', 'riang', 'bahagia', 'antusias'],
    'lembut': ['halus', 'penyayang', 'tenang', 'damai', 'kalem'],
    'pelindung': ['penjaga', 'pengawas', 'waspada', 'protektif', 'teritorial'],
    'kecil': ['mungil', 'minis', 'kecil', 'kecil', 'imut'],
    'besar': ['bongsor', 'tinggi', 'raksasa', 'gagah', 'besar'],
    'keluarga': ['rumah', 'anak', 'orangtua', 'domestik', 'keluarga'],
    'apartemen': ['flat', 'kondominium', 'rumah kecil', 'ruangan', 'indoor'],
    'outdoor': ['luar', 'aktif', 'petualang', 'alam', 'jalan-jalan'],
    'tenang': ['kalem', 'diam', 'damai', 'santai', 'tenang'],
    'berisik': ['ribut', 'bising', 'menggonggong', 'menggonggong', 'cerewet'],
    'perawatan': ['sikat', 'potong rambut', 'mandi', 'bulu', 'rambut'],
    'latihan': ['patuh', 'disiplin', 'belajar', 'perintah', 'instruksi'],
    'berburu': ['mangsa', 'melacak', 'memburu', 'mengejar', 'menangkap'],
    'penggembala': ['mengumpulkan', 'mengontrol', 'menggiring', 'bekerja', 'ternak'],
    'penjaga': ['melindungi', 'mengawasi', 'keamanan', 'waspada', 'patroli'],
    'teman': ['sahabat', 'kawan', 'teman', 'partner', 'pendamping'],
    'anak': ['kecil', 'anak', 'balita', 'keluarga', 'lembut'],
    'atletis': ['sportif', 'aktif', 'bugar', 'lincah', 'fisik'],
    'sehat': ['kuat', 'bugar', 'sehat', 'prima', 'tangguh'],
    'malas': ['santai', 'rileks', 'tenang', 'kalem', 'santai'],
    'dibuang': ['ditinggalkan', 'ditelantarkan', 'diusir', 'terlantar', 'sendirian'],
    'rescue': ['diselamatkan', 'ditolong', 'bantuan', 'penyelamatan', 'terselamatkan'],
    'sedih': ['sedih', 'galau', 'kesepian', 'menyedihkan', 'merana'],
    'kiul': ['kalem', 'tenang', 'pendiam', 'pasif', 'santai'],
    'ketek': ['lincah', 'aktif', 'energik', 'semangat', 'ceria'],
    'paksa': ['keras', 'tegas', 'kuat', 'dominan', 'tangguh'],
    'jaya': ['pemenang', 'sukses', 'hebat', 'unggul', 'terbaik'],
    'cewe': ['betina', 'perempuan', 'wanita'],
    'cowo': ['jantan', 'pria', 'laki']
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
                match_explanation = f"Cocok dengan: {', '.join(matching_terms)}"
            else:
                match_explanation = "Cocok berdasarkan sifat yang berhubungan"
            
            results.append({
                'id': df.iloc[idx]['id'],
                'name': df.iloc[idx]['name'],
                'traits': df.iloc[idx]['traits'],
                'age': df.iloc[idx]['age'],
                'gender': df.iloc[idx]['gender'],
                'description': df.iloc[idx]['description'],
                'similarity_score': combined_similarities[idx],
                'match_explanation': match_explanation
            })
    
    return results if results else [{"message": "Maaf, tidak ada hasil yang cocok dengan kriteria pencarian Anda."}]

# Example searches to test the system
def test_search_examples():
    test_queries = [
        "anjing yang suka tidur",
        "anjing yang pernah dibuang",
        "anjing betina muda",
        "anjing yang tenang dan kalem",
        "anjing yang cocok untuk keluarga",
        "anjing dengan sifat kiul",
        "anjing dengan cerita sedih",
        "anjing yang perlu diselamatkan",
        "anjing yang ditinggal pemiliknya"
    ]
    
    for query in test_queries:
        print(f"\nPencarian: '{query}'")
        results = search_dogs(query)
        
        if "message" in results[0]:
            print(results[0]["message"])
        else:
            for i, result in enumerate(results):
                print(f"{i+1}. {result['name']} ({result['traits']}, {result['age']}, {result['gender']}) - Skor: {result['similarity_score']:.4f}")
                print(f"   {result['description'][:100]}...")
                print(f"   {result['match_explanation']}")

# Run the test searches
test_search_examples()

# Interactive search function with improved UI
def interactive_search():
    print("\n===== PENCARIAN ANJING YANG SESUAI =====")
    print("Masukkan kriteria anjing yang Anda inginkan.")
    print("Contoh: 'anjing yang tenang dan suka tidur' atau 'anjing muda yang butuh keluarga baru'")
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
                print(f"{i+1}. {result['name']} - {result['traits']} - {result['age']} bulan - {result['gender']}")
                print(f"   Skor: {result['similarity_score']:.4f}")
                print(f"   Deskripsi: {result['description'][:150]}...")
                print(f"   {result['match_explanation']}")
                print("-" * 50)

# Start the interactive search
interactive_search()