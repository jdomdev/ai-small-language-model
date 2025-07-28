import streamlit as st
import requests
from transformers import pipeline
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Sentiment Analysis Comparator",
    page_icon="🎭",
    layout="wide"
)

st.title("Sentiment Analysis: SLM vs. LLM Comparator 🚀")
st.markdown("""
Introduce una reseña de una película en inglés y compara las predicciones de sentimiento
de diferentes modelos de Hugging Face.
""")

# --- Modelos a Comparar ---
# Reemplaza 'tu-usuario/tu-modelo-slm' con el ID de tu modelo en el Hugging Face Hub
YOUR_SLM_MODEL = "jdomdev/imdb-slm-vs-llm-pill"
# Otro modelo SLM popular para comparar
DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Un modelo LLM para la API de Inferencia
INFERENCE_API_LLM = "meta-llama/Meta-Llama-3-8B-Instruct"


# --- Carga de Modelos con Pipeline (en caché para eficiencia) ---
# Streamlit guardará en caché los modelos para no tener que descargarlos cada vez.
@st.cache_resource
def load_pipeline(model_name):
    """Carga un pipeline de Hugging Face."""
    st.info(f"Cargando el modelo: {model_name}...")
    return pipeline("sentiment-analysis", model=model_name)

# --- Función para la API de Inferencia ---
def query_inference_api(payload):
    """Hace una petición a la API de Inferencia de Hugging Face."""
    # Obtenemos el token de los secrets de Streamlit/Hugging Face
    # hf_token = os.environ.get("HF_TOKEN")
        # Obtenemos el token de los secrets de Streamlit/Hugging Face
    hf_token = st.secrets.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("Hugging Face API token no encontrado. Por favor, configúralo en los 'Secrets' de tu Space.")
        return None

    api_url = f"https://api-inference.huggingface.co/models/{INFERENCE_API_LLM}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Usamos un prompt específico para guiar al LLM
    prompt = f"""Analyze the sentiment of the following movie review and classify it as either 'POSITIVE' or 
 'NEGATIVE'. Respond with only one word.
 Review: "{payload['inputs']}"
 Sentiment:"""

    response = requests.post(api_url, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 3}})

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error en la llamada a la API de Inferencia ({INFERENCE_API_LLM}): {response.status_code} - {response.text}")
        return None

# --- Interfaz de Usuario ---
user_input = st.text_area("Escribe aquí la reseña de la película:", "I loved this movie, the acting was superb and the plot was gripping!", height=150)

if st.button("Analizar Sentimiento"):
    if user_input:
        st.subheader("Resultados del Análisis:")

        # Creamos columnas para una mejor visualización
        col1, col2, col3 = st.columns(3)

        # 1. Tu modelo SLM afinado
        with col1:
            st.info(f"Tu SLM: `{YOUR_SLM_MODEL}`")
            with st.spinner("Analizando..."):
                your_model_pipeline = load_pipeline(YOUR_SLM_MODEL)
                result = your_model_pipeline(user_input)
                st.json(result)

        # 2. Modelo DistilBERT estándar
        with col2:
            st.info(f"Otro SLM: `{DISTILBERT_MODEL}`")
            with st.spinner("Analizando..."):
                distilbert_pipeline = load_pipeline(DISTILBERT_MODEL)
                result = distilbert_pipeline(user_input)
                st.json(result)

        # 3. LLM vía API de Inferencia
        with col3:
            st.info(f"LLM (API): `{INFERENCE_API_LLM}`")
            with st.spinner("Consultando al LLM..."):
                result = query_inference_api({"inputs": user_input})
                if result:
                    # El resultado de la API es más complejo, lo parseamos
                    generated_text = result[0].get('generated_text', '').strip()
                    # Extraemos la última palabra, que debería ser POSITIVE o NEGATIVE
                    final_sentiment = generated_text.split()[-1]
                    st.json({"label": final_sentiment.upper()})

    else:
        st.warning("Por favor, introduce una reseña para analizar.")