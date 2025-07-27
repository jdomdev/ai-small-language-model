import streamlit as st
import os
import json
from dotenv import load_dotenv
from transformers import pipeline
from huggingface_hub import InferenceClient

# Cargar variables de entorno
load_dotenv()

# --- 1. Configuración de la Ruta del Modelo Local ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "fine_tuned_sentiment_model_full_data")

# --- 2. Carga del Modelo Local (SLM) ---
@st.cache_resource
def load_local_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"El directorio del modelo '{MODEL_PATH}' no se encuentra.")
        return None, None
    
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
        config_path = os.path.join(MODEL_PATH, 'config.json')
        with open(config_path) as f:
            config = json.load(f)
        id2label = config.get('id2label')
        if id2label is None:
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        else:
            id2label = {int(k): v for k, v in id2label.items()}
        return sentiment_pipeline, id2label
    except Exception as e:
        st.error(f"Error al cargar el modelo local: {e}")
        return None, None

slm_pipeline, id2label_map = load_local_pipeline()

def query_local_model(review):
    if slm_pipeline is None or id2label_map is None:
        return "Error: El modelo local no está disponible."

    try:
        result = slm_pipeline(review)[0]
        label_id = int(result['label'].split('_')[1])
        label = id2label_map.get(label_id, "Desconocido")
        score = result['score']
        return f"Resultado: {label} ({score:.2%})"
    except Exception as e:
        return f"Error durante la predicción: {e}"

# --- 3. LLM vía InferenceClient ---
# Modelos recomendados gratuitos (puedes probar con InferenceClient):
# ----------------------------------------------------------------------------
# Modelo	                                Tamaño	        Destacado por
# ----------------------------------------------------------------------------
# HuggingFaceH4/zephyr-7b-beta	            7B	            Muy bueno para tareas generales
# meta-llama/Meta-Llama-3-8B-Instruct	    8B	            Preciso y multilingüe
# ----------------------------------------------------------------------------
# mistralai/Mistral-7B-Instruct-v0.2	    7B	            Rápido y versátil
# Qwen/Qwen1.5-7B-Chat	                    7B	            Buen manejo de instrucciones
# NousResearch/Nous-Hermes-2-Mistral-7B-DPO	7B	            Instruct-tuned
# openchat/openchat-3.5-0106	            ChatGPT-like	Responde bien a instrucciones
# ----------------------------------------------------------------------------
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.environ.get("HF_TOKEN")

@st.cache_resource
def get_inference_client():
    if not HF_TOKEN:
        st.error("No se encontró el token de Hugging Face. Asegúrate de tener un archivo .env con HF_TOKEN.")
        return None
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

client = get_inference_client()

def query_llm_api(review):
    if client is None:
        return "Error: No se pudo inicializar el cliente de Hugging Face."

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "Analyze the sentiment. Respond with POSITIVE or NEGATIVE."},
                {"role": "user", "content": review}
            ],
            max_tokens=10,
            temperature=0.1,
        )

        return f"Resultado: {response.choices[0].message['content'].strip()}"
    except Exception as e:
        return f"Error al consultar el LLM: {e}"

# --- 4. Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("Comparador de Modelos de Sentimiento: SLM vs LLM")

st.info("""
Introduce una reseña de película en inglés para comparar el rendimiento de un modelo pequeño y eficiente (DistilBERT, local) 
contra un modelo de lenguaje grande (Zephyr 7B, en la nube).
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
            st.subheader("LLM (Zephyr 7B - Hugging Face API)")
            with st.spinner("Consultando al LLM en la nube..."):
                llm_result = query_llm_api(user_input)
                st.markdown(f"**Análisis:**\n{llm_result}")
        