# Instrucciones

## Clonar el repositorio

```bash
git clone https://github.com/TrodriTen/Quantum_Algorithms.git
```

## Hacer un ambiente virtual e instalar librerias

### Creacion ambiente

```bash 
python3 -m venv venv
```

### Activacion ambiente

```bash
source ./venv/bin/activate
```

### Instalacion librerias

```bash
pip install -r requirements.txt
```

## Ejecucion Taller 4

Para correr el taller 4 hay varias opciones, sin embargo, la estandar es: 

```bash
python3 ./homeworks/taller_4.py --xlsx ./data/DIVIPOLA_Municipios.xlsx 
```

## Ejecución Taller 6 — Búsqueda por similaridad coseno (Reuters)

El **taller_6.py** implementa un mini motor de búsqueda usando **TF-IDF (sparse)** + **cosine similarity**.
Puedes ejecutarlo de dos formas: con el **corpus Reuters de NLTK** o con una **carpeta local de `.txt`**.

### 0. Descargar recursos NLTK

```bash
python3 -c "import nltk; nltk.download('reuters'); nltk.download('punkt')"
```

### 1. Opción A - Usar NLTK Reuters

```bash
python3 ./homeworks/taller_6.py --use_nltk --query "oil price opec" --top_k 10
```

**Salida esperada (ejemplo abreviado):**

```
Documentos cargados: 10,788
Matriz TF-IDF (CSR) creada.
Docs: 10,788 | Vocab: 17,416 | NNZ: 571,344 | Densidad: 0.003041 (~0.3041%)

Consulta: "oil price opec"

Top resultados:
 1. training/144                                      score=0.625690  cats=[crude]
 ...
```

> Nota: `cats=[crude]` son las categorías de Reuters; aparecen solo en modo NLTK.

### 2. Opción B - Usar carpeta local de `.txt`

Coloca tus documentos en una carpeta (cada archivo `.txt` = 1 documento), por ejemplo `./data/reuters_txt/`, y ejecuta:

```bash
python3 ./homeworks/taller_6.py --data_dir ./data/reuters_txt --query "british jaguar sales" --top_k 5
```


