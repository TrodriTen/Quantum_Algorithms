
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

import nltk
from nltk.corpus import reuters as nltk_reuters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_txt_directory(data_dir: str) -> Tuple[List[str], List[str]]:
    p = Path(data_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")

    doc_ids, docs = [], []
    for name in sorted(os.listdir(p)):
        if name.lower().endswith(".txt"):
            doc_ids.append(name)
            with open(p / name, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
    if not docs:
        raise RuntimeError(f"No se encontraron .txt en {data_dir}")
    return doc_ids, docs


def read_nltk_reuters() -> Tuple[List[str], List[str]]:
    if nltk is None or nltk_reuters is None:
        raise RuntimeError("NLTK/Reuters no está disponible. Instale nltk y ejecute nltk.download('reuters', 'punkt').")
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    fileids = nltk_reuters.fileids()
    texts = [nltk_reuters.raw(fid) for fid in fileids]
    return fileids, texts


def build_vectorizer(ngram_max: int = 1) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, ngram_max),
        stop_words="english",
        max_df=0.9,      
        min_df=2,       
        dtype=np.float32,
        norm="l2",       
    )
    return vec


def sparsity_report(X) -> str:
    nnz = X.nnz
    total = X.shape[0] * X.shape[1]
    density = nnz / total if total > 0 else 0.0
    return f"Docs: {X.shape[0]:,} | Vocab: {X.shape[1]:,} | NNZ: {nnz:,} | Densidad: {density:.6f} (~{density*100:.4f}%)"


def search(
    query: str,
    vec: TfidfVectorizer,
    X,
    doc_ids: List[str],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    qv = vec.transform([query]) 
    #sims = (X @ qv.T).toarray().ravel() 
    sims = cosine_similarity(X, qv, dense_output=False).toarray().ravel()
    top_idx = np.argsort(-sims)[:top_k]
    return [(doc_ids[i], float(sims[i])) for i in top_idx if sims[i] > 0]


def load_corpus(use_nltk: bool, data_dir: str) -> Tuple[List[str], List[str]]:
    if use_nltk:
        print("Leyendo corpus Reuters desde NLTK...")
        return read_nltk_reuters()
    else:
        print(f"Leyendo .txt desde: {data_dir}")
        return read_txt_directory(data_dir)


def main():
    parser = argparse.ArgumentParser(description="Búsqueda por similaridad coseno sobre un corpus estilo Reuters.")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--use_nltk", action="store_true", help="Usar el corpus Reuters de NLTK")
    parser.add_argument("--data_dir", type=str, default="./corpus", help="Directorio con .txt (si no se usa NLTK)")
    parser.add_argument("--query", type=str, required=True, help="Consulta de búsqueda, p.ej. 'british jaguar sales'")
    parser.add_argument("--top_k", type=int, default=10, help="Número de documentos a retornar")
    parser.add_argument("--ngram_max", type=int, default=1, help="Usar unigramas (1) o bigramas (2)")
    args = parser.parse_args()

    doc_ids, docs = load_corpus(args.use_nltk, args.data_dir)
    print(f"Documentos cargados: {len(docs):,}")

    vectorizer = build_vectorizer(args.ngram_max)
    X = vectorizer.fit_transform(docs)
    print("Matriz TF-IDF (CSR) creada.")
    print(sparsity_report(X))

    print(f"\nConsulta: \"{args.query}\"")
    results = search(args.query, vectorizer, X, doc_ids, top_k=args.top_k)

    if not results:
        print("No se encontraron documentos relevantes (score > 0).")
        sys.exit(0)

    print("\nTop resultados:")
    for rank, (doc_id, score) in enumerate(results, 1):
        cats = ", ".join(nltk_reuters.categories(doc_id)) if args.use_nltk else ""
        cat_str = f"  cats=[{cats}]" if cats else ""
        print(f"{rank:>2}. {doc_id:<50}  score={score:.6f}{cat_str}")

    try:
        id2text = {i: t for i, t in zip(doc_ids, docs)}
        query_terms = [w for w in args.query.lower().split() if len(w) > 1]
        print("\nSnippets:")
        for rank, (doc_id, score) in enumerate(results, 1):
            text = id2text[doc_id]
            low = text.lower()
            pos = min((low.find(term) for term in query_terms if term in low), default=-1)
            if pos >= 0:
                start = max(0, pos - 50)
                end = min(len(text), pos + 100)
                snippet = text[start:end].replace("\n", " ")
            else:
                snippet = text[:120].replace("\n", " ")
            print(f"{rank:>2}. {doc_id:50s} ... {snippet} ...")
    except Exception:
        pass


if __name__ == "__main__":
    main()
