import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Ruta donde se guardó el modelo y el tokenizador
MODEL_PATH = "./fine_tuned_sentiment_model"

@st.cache_resource
def load_model():
    """Carga el tokenizador y el modelo pre-entrenado."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

st.title("Análisis de Sentimiento de Reseñas de Películas")
st.write("Introduce una reseña de película para predecir si es positiva o negativa.")

# Cargar el modelo y el tokenizador (se cachean para no recargar en cada interacción)
tokenizer, model = load_model()

# Área de texto para la entrada del usuario
user_input = st.text_area("Escribe tu reseña aquí:", "Me encantó esta película, fue increíble y muy emocionante.")

if st.button("Predecir Sentimiento"):
    if user_input:
        # Tokenizar la entrada
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Realizar la predicción
        with torch.no_grad():
            logits = model(**inputs).logits

        # Obtener la clase predicha (0 para negativo, 1 para positivo)
        predicted_class_id = logits.argmax().item()

        # Mapear el ID a la etiqueta de sentimiento
        sentiment_labels = {0: "Negativo", 1: "Positivo"}
        predicted_sentiment = sentiment_labels[predicted_class_id]

        st.write("\n--- Resultado ---")
        st.write(f"Sentimiento Predicho: **{predicted_sentiment}**")

        # Opcional: Mostrar las probabilidades (logits)
        # st.write(f"Logits: {logits.tolist()}")
    else:
        st.warning("Por favor, escribe una reseña para predecir.")