

import streamlit as st
import os
import json
import requests
from transformers import pipeline

# --- 1. Configuración de la Ruta del Modelo Local ---
# Construye la ruta absoluta al directorio del modelo basándose en la ubicación del script.
# Esto hace que la aplicación sea más robusta y no dependa del directorio de trabajo actual.
try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Si __file__ no está definido (por ejemplo, en un entorno interactivo como un notebook)
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "fine_tuned_sentiment_model_full_data")

# --- 2. Carga del Modelo Local (SLM) ---
@st.cache_resource
def load_local_pipeline():
    """Carga el pipeline de análisis de sentimiento del modelo local."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"El directorio del modelo '{MODEL_PATH}' no se encuentra.")
        return None, None
    
    try:
        # Carga el pipeline de análisis de sentimiento
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
        
        # Carga el mapeo de etiquetas desde el archivo de configuración
        config_path = os.path.join(MODEL_PATH, 'config.json')
        with open(config_path) as f:
            config = json.load(f)
        
        id2label = config.get('id2label')
        if id2label is None:
            # Si id2label no está en el config, lo creamos manualmente
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        else:
            id2label = {int(k): v for k, v in id2label.items()}

        return sentiment_pipeline, id2label
    except Exception as e:
        st.error(f"Error al cargar el modelo local: {e}")
        return None, None

slm_pipeline, id2label_map = load_local_pipeline()

def query_local_model(review):
    """Realiza una predicción de sentimiento usando el SLM local."""
    if slm_pipeline is None or id2label_map is None:
        return "Error: El modelo local no está disponible."

    try:
        # Realiza la predicción
        result = slm_pipeline(review)[0]
        label_id = int(result['label'].split('_')[1])
        label = id2label_map.get(label_id, "Desconocido")
        score = result['score']
        
        return f"Resultado: {label} ({score:.2%})"
    except Exception as e:
        return f"Error durante la predicción: {e}"

# --- 3. Lógica para el LLM (Hugging Face Inference API) ---
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except (KeyError, FileNotFoundError):
    HF_TOKEN = None

def query_llm_api(review):
    """Consulta un LLM a través de la API de Inferencia de Hugging Face."""
    if not HF_TOKEN:
        return "Error: No se encontró el token de Hugging Face. Configúralo en .streamlit/secrets.toml"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Prompt Engineering: Le damos al LLM una instrucción clara y concisa.
    prompt = f"""
    Analyze the sentiment of the following movie review. Respond only with the word 'POSITIVE' or 'NEGATIVE'.
    Review: "{review}"
    """
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.1,
            "return_full_text": False,
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Lanza una excepción para códigos de estado 4xx/5xx
        
        result = response.json()
        answer = result[0]['generated_text'].strip()
        return f"Resultado: {answer}"

    except requests.exceptions.RequestException as e:
        return f"Error de conexión con la API: {e}"
    except Exception as e:
        return f"Error al procesar la respuesta de la API: {e}"

# --- 4. Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("Comparador de Modelos de Sentimiento: SLM vs LLM")

st.info("""
Introduce una reseña de película en inglés para comparar el rendimiento de un modelo pequeño y eficiente (DistilBERT, local) 
contra un modelo de lenguaje grande (Llama 3 8B, en la nube).
""")

user_input = st.text_area(
    "Introduce la reseña aquí:",
    "This movie was absolutely fantastic, a true masterpiece of cinema!",
    height=150
)

if st.button("Analizar Sentimiento", type="primary"):
    if not user_input:
        st.warning("Por favor, introduce una reseña.")
    elif slm_pipeline is None:
        st.error("La aplicación no puede iniciarse porque el modelo local no se pudo cargar.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tu Modelo (DistilBERT - Local)")
            with st.spinner("Procesando con el modelo local..."):
                slm_result = query_local_model(user_input)
                st.markdown(f"**Análisis:**\n{slm_result}")

        with col2:
            st.subheader("LLM (Llama 3 8B - Hugging Face API)")
            with st.spinner("Consultando al LLM en la nube..."):
                llm_result = query_llm_api(user_input)
                st.markdown(f"**Análisis:**\n{llm_result}")

