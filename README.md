# Detección de Emojis con SLM (DistilBERT)

Este proyecto muestra cómo entrenar y probar un modelo de lenguaje pequeño (SLM) para clasificar frases según su emoji usando Hugging Face Transformers y PyTorch.

## Requisitos

- Python 3.8 o superior
- Acceso a internet para descargar modelos preentrenados

## Instalación de dependencias

Instala todas las dependencias necesarias ejecutando:

```
pip install -r requirements.txt
```

## Ejecución paso a paso

1. **Clona o descarga este repositorio y entra en la carpeta del proyecto.**
2. **(Opcional) Crea y activa un entorno virtual:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ejecuta el script principal:**
   - **Windows:**
     ```powershell
     python ejem.py
     ```
   - **Linux/Mac:**
     ```bash
     python ejem.py
     ```

## Script especial para automatizar todo

### Windows (PowerShell)
Copia y guarda esto como `run_all_win.ps1` y ejecútalo en PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python ejem.py
```

### Linux/Mac (Bash)
Copia y guarda esto como `run_all_unix.sh` y ejecútalo en la terminal:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ejem.py
```

## Notas
- El script `ejem.py` entrena y evalúa el modelo, y permite probar frases interactivamente.
- Si tienes problemas con dependencias, asegúrate de tener la última versión de pip: `pip install --upgrade pip`.
- El entrenamiento es rápido porque el dataset es pequeño y el modelo es ligero.

---

**Autor:** Juan Domingo
**Autor:** Juan Carlos Macías
