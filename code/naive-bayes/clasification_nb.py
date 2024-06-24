import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Función para normalizar una reseña de película en inglés
def normalize_review(review):
    # Convertir el texto a minúsculas
    review = review.lower()
    
    # Tokenizar el texto en palabras
    tokens = word_tokenize(review)
    
    # Eliminar signos de puntuación
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Eliminar palabras vacías (stop words)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lematización de palabras
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Unir las palabras normalizadas en una cadena de texto nuevamente
    normalized_review = ' '.join(tokens)
    
    return normalized_review

# Cargar el vectorizador TF-IDF y el modelo Naive Bayes
tfidf_vectorizer = joblib.load('../gui/tfidf_vectorizer.pkl')
nb_classifier = joblib.load('naive_bayes_model.pkl')

# Definir las nuevas reseñas en inglés
new_reviews = [
    "This movie was excellent, I loved every moment.",
    "I would not recommend this film to anyone, it is of poor quality.",
    "The acting was terrible, I will never watch this actor again.",
    "What a great experience! The cinematography was stunning and the plot excellent."
]

# Normalizar las nuevas reseñas
normalized_new_reviews = [normalize_review(review) for review in new_reviews]

# Convertir las nuevas reseñas a representación TF-IDF
X_new_tfidf = tfidf_vectorizer.transform(normalized_new_reviews)

# Predecir el sentimiento de las nuevas reseñas
y_pred_new = nb_classifier.predict(X_new_tfidf)

# Asignar etiquetas predictivas a las reseñas originales
for i, review in enumerate(new_reviews):
    if y_pred_new[i] == 'positive':
        print(f'Review: "{review}" - Prediction: Positive')
    else:
        print(f'Review: "{review}" - Prediction: Negative')
