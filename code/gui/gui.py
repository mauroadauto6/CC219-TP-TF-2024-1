import tkinter as tk
from tkinter import messagebox
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
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

# Función para clasificar una reseña usando SVM
def classify_review_svm():
    # Obtener la reseña ingresada por el usuario desde la interfaz gráfica
    user_review = entry_review.get()
    
    # Normalizar la reseña ingresada por el usuario
    normalized_review = normalize_review(user_review)
    
    # Convertir la reseña normalizada a representación TF-IDF
    X_review_tfidf = tfidf_vectorizer.transform([normalized_review])
    
    # Predecir el sentimiento de la reseña con SVM
    y_pred_review = svm_classifier.predict(X_review_tfidf)
    
    # Mostrar el resultado de la clasificación con SVM
    if y_pred_review[0] == 'positive':
        messagebox.showinfo("Resultado de la Clasificación (SVM)", "La reseña es POSITIVA según SVM.")
    else:
        messagebox.showinfo("Resultado de la Clasificación (SVM)", "La reseña es NEGATIVA según SVM.")

# Función para clasificar una reseña usando Naive Bayes
def classify_review_nb():
    # Obtener la reseña ingresada por el usuario desde la interfaz gráfica
    user_review = entry_review.get()
    
    # Normalizar la reseña ingresada por el usuario
    normalized_review = normalize_review(user_review)
    
    # Convertir la reseña normalizada a representación TF-IDF
    X_review_tfidf = tfidf_vectorizer.transform([normalized_review])
    
    # Predecir el sentimiento de la reseña con Naive Bayes
    y_pred_review = nb_classifier.predict(X_review_tfidf)
    
    # Mostrar el resultado de la clasificación con Naive Bayes
    if y_pred_review[0] == 'positive':
        messagebox.showinfo("Resultado de la Clasificación (Naive Bayes)", "La reseña es POSITIVA según Naive Bayes.")
    else:
        messagebox.showinfo("Resultado de la Clasificación (Naive Bayes)", "La reseña es NEGATIVA según Naive Bayes.")

# Cargar el vectorizador TF-IDF y los modelos SVM y Naive Bayes
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_classifier = joblib.load('../svm/svm_model.pkl')
nb_classifier = joblib.load('../naive-bayes/naive_bayes_model.pkl')

# Crear la interfaz gráfica usando Tkinter
root = tk.Tk()
root.title("Clasificador de Reseñas de Películas")

# Etiqueta y entrada para ingresar la reseña
label_review = tk.Label(root, text="Ingrese una reseña de película:")
label_review.pack(pady=10)
entry_review = tk.Entry(root, width=50)
entry_review.pack(pady=10)

# Botón y función para clasificar la reseña con SVM
button_classify_svm = tk.Button(root, text="Clasificar con SVM", command=classify_review_svm)
button_classify_svm.pack(pady=5)

# Botón y función para clasificar la reseña con Naive Bayes
button_classify_nb = tk.Button(root, text="Clasificar con Naive Bayes", command=classify_review_nb)
button_classify_nb.pack(pady=5)

# Ejecutar el bucle principal de la interfaz gráfica
root.mainloop()
