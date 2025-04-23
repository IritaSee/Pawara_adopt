from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import os
from oke import preprocess_text, tokenizer, MAX_SEQUENCE_LENGTH

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@localhost/pawara_adopt'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define Dog model for database
class Dog(db.Model):
    __tablename__ = 'dogs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    traits = db.Column(db.String(100))
    age = db.Column(db.Float)
    gender = db.Column(db.String(10))
    description = db.Column(db.Text, nullable=False)
    story_embedding = db.Column(db.JSON)  # Store story embeddings as JSON -> need to change our current database to JSON type and add a new column to the database

# Load the trained model
try:
    model = load_model('main_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_embedding(text):
    """Generate embedding for a given text using the loaded model"""
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return model.predict(padded)[0].tolist()

def search_similar_dogs(query, top_n=3):
    """Search for dogs with similar stories based on embedding similarity"""
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Get all dogs from database
    all_dogs = Dog.query.all()
    
    # Calculate similarities
    similarities = []
    for dog in all_dogs:
        if dog.story_embedding:
            similarity = cosine_similarity(
                [query_embedding], 
                [dog.story_embedding]
            )[0][0]
            similarities.append((dog, similarity))
    
    # Sort by similarity and get top N results
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_n]
    
    # Format results
    results = []
    for dog, similarity in top_results:
        if similarity > 0.1:  # Only include somewhat relevant results
            results.append({
                'id': dog.id,
                'name': dog.name,
                'traits': dog.traits,
                'age': dog.age,
                'gender': dog.gender,
                'description': dog.description,
                'similarity_score': float(similarity)
            })
    
    return results

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'No search query provided'
            }), 400
        
        # Get search results
        results = search_similar_dogs(query)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dogs', methods=['GET'])
def get_all_dogs():
    """Get all dogs from database"""
    try:
        dogs = Dog.query.all()
        results = [{
            'id': dog.id,
            'name': dog.name,
            'traits': dog.traits,
            'age': dog.age,
            'gender': dog.gender,
            'description': dog.description
        } for dog in dogs]
        
        return jsonify({
            'success': True,
            'dogs': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/update-embeddings', methods=['POST'])
def update_embeddings():
    """Update story embeddings for all dogs in database"""
    try:
        dogs = Dog.query.all()
        updated_count = 0
        
        for dog in dogs:
            if dog.description:
                # Generate and store embedding
                embedding = get_embedding(dog.description)
                dog.story_embedding = embedding
                updated_count += 1
        
        # Commit changes to database
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Updated embeddings for {updated_count} dogs'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Script to initialize database and create tables
def init_db():
    """Initialize database and create tables"""
    with app.app_context():
        db.create_all()
        print("Database initialized and tables created")

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Start the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)