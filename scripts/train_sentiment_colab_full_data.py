# El modelo actual pesa 257MB, lo que confirma que la división será necesaria para subirlo a GitHub.

# Vamos a crear el notebook de Colab paso a paso. Te proporcionaré el contenido del notebook en bloques de código que podrás copiar
# y pegar en un nuevo archivo .ipynb en Colab.

# Contenido del Notebook de Colab (train_sentiment_colab.ipynb)


# Bloque 1: Configuración Inicial e Importaciones

# Este bloque se encarga de instalar las librerías necesarias (aunque muchas ya están en Colab, es buena práctica incluirlas para
# asegurar la reproducibilidad), montar Google Drive y definir las rutas.

# @title 1. Configuración Inicial e Importaciones

# Instalar librerías necesarias (muchas ya están en Colab, pero es buena práctica)
# !pip install transformers[torch] datasets evaluate scikit-learn accelerate pandas numpy

# Importar librerías
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split
import re
import os

# Montar Google Drive para guardar/cargar archivos grandes
from google.colab import drive
drive.mount('/content/drive')

# Definir la ruta base en Google Drive para guardar los resultados y el modelo
# Asegúrate de que esta carpeta exista en tu Google Drive
DRIVE_BASE_PATH = "/content/drive/MyDrive/SLM_Training_Results"
os.makedirs(DRIVE_BASE_PATH, exist_ok=True)

print("Configuración inicial completada.")


# Bloque 2: Carga y Guardado de Datasets Originales a CSV

# Aquí cargaremos los datasets completos de IMDb y los guardaremos como CSVs separados en tu Google Drive.

# @title 2. Carga y Guardado de Datasets Originales a CSV

print("Cargando el dataset IMDb completo...")
dataset = load_dataset("imdb")
print("Dataset IMDb cargado.")

# Convertir a Pandas DataFrame y guardar como CSV
df_train_original = pd.DataFrame(dataset["train"])
df_test_original = pd.DataFrame(dataset["test"])

train_csv_path = os.path.join(DRIVE_BASE_PATH, "imdb_train_original.csv")
test_csv_path = os.path.join(DRIVE_BASE_PATH, "imdb_test_original.csv")

df_train_original.to_csv(train_csv_path, index=False)
df_test_original.to_csv(test_csv_path, index=False)

print(f"Dataset de entrenamiento original guardado en: {train_csv_path}")
print(f"Dataset de prueba original guardado en: {test_csv_path}")
print(f"Tamaño del dataset de entrenamiento original: {len(df_train_original)} registros")
print(f"Tamaño del dataset de prueba original: {len(df_test_original)} registros")


# Bloque 3: Unificación y Limpieza de Datos

# Este es el bloque crucial para la limpieza y unificación. Incluiré una función de limpieza básica y comprobaciones de
# nulos/duplicados.

# @title 3. Unificación y Limpieza de Datos

print("Unificando datasets de entrenamiento y prueba...")
df_full = pd.concat([df_train_original, df_test_original], ignore_index=True)
print(f"Dataset unificado creado con {len(df_full)} registros.")

print("Realizando comprobaciones de limpieza de datos...")

# Comprobar nulos
print(f"Valores nulos por columna antes de la limpieza:\n{df_full.isnull().sum()}")

# Comprobar duplicados
print(f"Registros duplicados antes de la limpieza: {df_full.duplicated().sum()}")
if df_full.duplicated().sum() > 0:
    df_full.drop_duplicates(inplace=True)
    print(f"Registros duplicados eliminados. Nuevo tamaño: {len(df_full)}")

# Función de limpieza de texto
def clean_text(text):
    text = str(text).lower() # Convertir a string y a minúsculas
    text = re.sub(r'<br />', ' ', text) # Eliminar etiquetas <br />
    text = re.sub(r'[^a-z0-9\s]', '', text) # Eliminar caracteres especiales (mantener letras, números, espacios)
    text = re.sub(r'\s+', ' ', text).strip() # Eliminar espacios extra
    return text

print("Aplicando limpieza de texto a la columna 'text'...")
df_full['text'] = df_full['text'].apply(clean_text)

# Comprobar si hay "nulos" como cadenas vacías o solo espacios después de la limpieza
print(f"Registros con texto vacío después de la limpieza: {(df_full['text'] == '').sum()}")
if (df_full['text'] == '').sum() > 0:
    df_full = df_full[df_full['text'] != '']
    print(f"Registros con texto vacío eliminados. Nuevo tamaño: {len(df_full)}")

# Guardar el dataset unificado y limpio
full_cleaned_csv_path = os.path.join(DRIVE_BASE_PATH, "imdb_full_cleaned.csv")
df_full.to_csv(full_cleaned_csv_path, index=False)
print(f"Dataset unificado y limpio guardado en: {full_cleaned_csv_path}")


# Bloque 4: División 80/20 para Entrenamiento y Prueba

# Aquí dividiremos el dataset limpio en 80% para entrenamiento y 20% para prueba.

# @title 4. División 80/20 para Entrenamiento y Prueba

print("Dividiendo el dataset unificado en 80% entrenamiento y 20% prueba...")
train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['label'])

# Convertir DataFrames de Pandas a objetos Dataset de Hugging Face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Eliminar la columna '__index_level_0__' que se añade automáticamente al convertir de pandas
train_dataset = train_dataset.remove_columns(["__index_level_0__"])
test_dataset = test_dataset.remove_columns(["__index_level_0__"])

print(f"Tamaño del dataset de entrenamiento (80%): {len(train_dataset)} registros")
print(f"Tamaño del dataset de prueba (20%): {len(test_dataset)} registros")


# Bloque 5: Tokenización y Carga del Modelo (Adaptado del Script Original)

# Este bloque es una adaptación directa de tu train_sentiment_model.py.

# @title 5. Tokenización y Carga del Modelo

# Cargar el Tokenizador
print("\nCargando el tokenizador DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Función de Preprocesamiento
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

print("\nPreprocesando el dataset de entrenamiento y prueba...")
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Cargar el Modelo
print("\nCargando el modelo DistilBERT para clasificación de secuencias...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# Bloque 6: Definición de Métricas y Configuración de Entrenamiento

# También adaptado de tu script original.

# @title 6. Definición de Métricas y Configuración de Entrenamiento

# Definir Métricas de Evaluación
print("\nDefiniendo métricas de evaluación...")
metric = load("accuracy")
f1_metric = load("f1")
precision_metric = load("precision")
recall_metric = load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {**accuracy, **f1, **precision, **recall}

# Configurar Argumentos de Entrenamiento
print("\nConfigurando argumentos de entrenamiento...")
training_args = TrainingArguments(
    output_dir=os.path.join(DRIVE_BASE_PATH, "results"), # Directorio para guardar los resultados en Drive
    num_train_epochs=3,                   # Número de épocas de entrenamiento
    per_device_train_batch_size=16,  # Tamaño del batch por dispositivo (GPU/CPU)
    per_device_eval_batch_size=16,   # Tamaño del batch para evaluación
    warmup_steps=500,                   # Número de pasos para el calentamiento del learning rate
    weight_decay=0.01,                  # Regularización L2
    logging_dir=os.path.join(DRIVE_BASE_PATH, "logs"), # Directorio para los logs de TensorBoard en Drive
    logging_steps=100,
    report_to="none",                   # No reportar a ninguna plataforma (ej. wandb)
    save_strategy="epoch",            # Guardar el modelo al final de cada época
    load_best_model_at_end=True,     # Cargar el mejor modelo al final del entrenamiento
    metric_for_best_model="f1",      # Métrica para determinar el mejor modelo
)


# Bloque 7: Entrenamiento del Modelo

# @title 7. Entrenamiento del Modelo

# Crear el Trainer
print("\nCreando el Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Entrenar el Modelo
print("\nIniciando el entrenamiento del modelo...")
trainer.train()
print("\nEntrenamiento completado.")

# Evaluar el Modelo Final
print("\nEvaluando el modelo final en el conjunto de prueba...")
eval_results = trainer.evaluate()
print(f"Resultados de la evaluación final: {eval_results}")


# Bloque 8: Guardado del Modelo y División para GitHub

# Este bloque guardará el modelo en Google Drive y luego implementará la lógica para dividir el archivo pytorch_model.bin si es
# demasiado grande.

# @title 8. Guardado del Modelo y División para GitHub

# 10. Guardar el Modelo en Google Drive
model_save_path_drive = os.path.join(DRIVE_BASE_PATH, "fine_tuned_sentiment_model_full_data")
print(f"\nGuardando el modelo y tokenizador en Google Drive: {model_save_path_drive}")
trainer.save_model(model_save_path_drive)
tokenizer.save_pretrained(model_save_path_drive)
print("Modelo y tokenizador guardados exitosamente en Google Drive.")

# --- Lógica para dividir el modelo para GitHub ---
# GitHub tiene un límite de 100MB por archivo.
# El archivo principal del modelo suele ser 'pytorch_model.bin' o 'model.safetensors'.

model_bin_path = os.path.join(model_save_path_drive, "pytorch_model.bin")
if not os.path.exists(model_bin_path):
    # Si no es pytorch_model.bin, podría ser safetensors
    model_bin_path = os.path.join(model_save_path_drive, "model.safetensors")

if os.path.exists(model_bin_path):
    file_size_mb = os.path.getsize(model_bin_path) / (1024 * 1024)
    print(f"\nTamaño del archivo del modelo ({os.path.basename(model_bin_path)}): {file_size_mb:.2f} MB")

    if file_size_mb > 90: # Usamos 90MB como umbral para estar seguros por debajo de 100MB
        print(f"El archivo del modelo ({os.path.basename(model_bin_path)}) excede los 90MB. Dividiendo...")

        def split_file(filepath, chunk_size_mb=90):
            chunk_size = int(chunk_size_mb * 1024 * 1024)
            base_filename = os.path.basename(filepath)
            output_dir = os.path.dirname(filepath)

            with open(filepath, 'rb') as f:
                part_num = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    part_filename = os.path.join(output_dir, f"{base_filename}.part{part_num:03d}")
                    with open(part_filename, 'wb') as part_f:
                        part_f.write(chunk)
                    print(f"  Creada parte: {os.path.basename(part_filename)}")
                    part_num += 1
            print(f"Archivo '{base_filename}' dividido en {part_num} partes en {output_dir}.")
            print("Puedes subir estas partes a GitHub. Recuerda NO subir el archivo original grande.")
            print("Para reconstruir, usa el comando 'cat' o la función 'join_files' proporcionada.")

        split_file(model_bin_path)
    else:
        print("El archivo del modelo es menor de 90MB. No es necesario dividirlo para GitHub.")
else:
    print(f"Advertencia: No se encontró el archivo principal del modelo ({os.path.basename(model_bin_path)}).")


# Instrucciones para el Usuario

# 1. Crea un nuevo Notebook en Google Colab:
#     * Ve a Google Colab (https://colab.research.google.com/).
#     * Haz clic en Archivo -> Nuevo notebook.
# 2. Copia y Pega los Bloques de Código:
#     * Copia cada bloque de código que te he proporcionado y pégalo en celdas separadas en tu nuevo notebook.
#     * Ejecuta cada celda en orden.
# 3. Asegúrate de que la carpeta `SLM_Training_Results` exista en tu Google Drive antes de ejecutar el notebook, o cámbiala a una
# ruta que prefieras.
# 4. Conectar Colab con GitHub (para commits):
#     * Una vez que el entrenamiento haya terminado y tengas el notebook con los resultados, puedes guardarlo en GitHub.
#     * Ve a Archivo -> Guardar una copia en GitHub....
#     * Sigue las instrucciones para autorizar Colab y seleccionar tu repositorio.
#     * Para los archivos del modelo divididos, tendrás que descargarlos de Google Drive a tu máquina local y luego subirlos
#     manualmente a GitHub (o usar git directamente en Colab si clonas tu repositorio y trabajas dentro de él, lo cual es más
#     avanzado).


# Función para Reconstruir el Modelo (para uso local)

# Si necesitas reconstruir el modelo a partir de las partes descargadas de GitHub en tu máquina local, usa esta función Python:

import os

def join_files(output_filepath, part_prefix):
    """
    Reconstruye un archivo a partir de sus partes.
    output_filepath: Ruta completa del archivo final a reconstruir.
    part_prefix: Prefijo de los archivos de las partes (ej.
    "./fine_tuned_sentiment_model_full_data/pytorch_model.bin.part").
                 Asegúrate de que incluya la ruta completa a las partes.
    """
    print(f"Reconstruyendo archivo en: {output_filepath}")
    with open(output_filepath, 'wb') as outfile:
        part_num = 0
        while True:
            # Formato de nombre de parte: .part000, .part001, etc.
            part_filename = f"{part_prefix}{part_num:03d}"
            if not os.path.exists(part_filename):
                break
            print(f"  Añadiendo parte: {os.path.basename(part_filename)}")
            with open(part_filename, 'rb') as infile:
                outfile.write(infile.read())
            part_num += 1
    if part_num > 0:
        print(f"Archivo reconstruido exitosamente en '{output_filepath}' a partir de {part_num} partes.")
    else:
        print(f"Advertencia: No se encontraron partes con el prefijo '{part_prefix}'.")

# Ejemplo de uso (ajusta las rutas según donde descargues las partes):
# model_dir = "./fine_tuned_sentiment_model_full_data" # Carpeta donde están las partes
# output_file = os.path.join(model_dir, "pytorch_model.bin")
# part_base_name = "pytorch_model.bin.part" # Nombre base del archivo original
# join_files(output_file, os.path.join(model_dir, part_base_name))