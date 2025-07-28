from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

# 1. Cargar el Dataset
# Usaremos el dataset IMDb para análisis de sentimiento de películas.
# Contiene reseñas de películas etiquetadas como 'positive' (1) o 'negative' (0).
print("Cargando el dataset IMDb...")
dataset = load_dataset("imdb")
print("Dataset cargado. Ejemplos:")
print(dataset["train"][0])
print(dataset["test"][0])

# Limitar el tamaño del dataset para entrenamiento y prueba
# Para propósitos de entrenamiento rápido, limitamos el tamaño del dataset.
# En un entorno real, usaríamos todo el dataset.
print("\nLimitando el tamaño del dataset para entrenamiento y prueba...")
train_size = 3000
test_size = 1500
dataset["train"] = dataset["train"].select(range(train_size))
dataset["test"] = dataset["test"].select(range(test_size))
print(f"Dataset limitado: {len(dataset['train'])} ejemplos de entrenamiento, {len(dataset['test'])} ejemplos de prueba.")

# 2. Cargar el Tokenizador
# Usaremos un tokenizador de un modelo pre-entrenado pequeño (DistilBERT)
# que es eficiente y bueno para fine-tuning.
print("\nCargando el tokenizador DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 3. Función de Preprocesamiento
# Esta función tokenizará el texto y lo preparará para el modelo.
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

print("\nPreprocesando el dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Cargar el Modelo
# Cargamos un modelo pre-entrenado para clasificación de secuencias.
# Especificamos el número de etiquetas (2: positivo/negativo).
print("\nCargando el modelo DistilBERT para clasificación de secuencias...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 5. Definir Métricas de Evaluación
# Usaremos Accuracy, F1-score, Precision y Recall.
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

# 6. Configurar Argumentos de Entrenamiento
# Aquí definimos cómo se entrenará el modelo (épocas, tamaño de batch, etc.).
print("\nConfigurando argumentos de entrenamiento...")
training_args = TrainingArguments(
    output_dir="./results",          # Directorio para guardar los resultados
    num_train_epochs=3,              # Número de épocas de entrenamiento
    per_device_train_batch_size=16,  # Tamaño del batch por dispositivo (GPU/CPU)
    per_device_eval_batch_size=16,   # Tamaño del batch para evaluación
    warmup_steps=500,                # Número de pasos para el calentamiento del learning rate
    weight_decay=0.01,               # Regularización L2
    logging_dir="./logs",            # Directorio para los logs de TensorBoard
    logging_steps=100,
    report_to="none",                # No reportar a ninguna plataforma (ej. wandb)
)

# 7. Crear el Trainer
# El Trainer es una clase de Hugging Face que simplifica el entrenamiento.
print("\nCreando el Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Entrenar el Modelo
print("\nIniciando el entrenamiento del modelo...")
trainer.train()
print("\nEntrenamiento completado.")

# 9. Evaluar el Modelo Final
print("\nEvaluando el modelo final en el conjunto de prueba...")
eval_results = trainer.evaluate()
print(f"Resultados de la evaluación final: {eval_results}")

# 10. Guardar el Modelo
# Guardamos el modelo entrenado y el tokenizador para futuras inferencias.
model_save_path = "./fine_tuned_sentiment_model"
print(f"\nGuardando el modelo y tokenizador en: {model_save_path}")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("Modelo y tokenizador guardados exitosamente.")

