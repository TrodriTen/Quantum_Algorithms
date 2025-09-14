import os
import time
import math
import heapq
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

def detect_header_row(xlsx_path: str, max_search: int = 30) -> int:
    raw = pd.read_excel(xlsx_path, header=None)
    for i in range(min(max_search, len(raw))):
        row = raw.iloc[i].astype(str).str.upper().tolist()
        if any('LATITUD' == x for x in row) and any('LONGITUD' == x for x in row):
            return i
    return 0

def load_points_flexible(xlsx_path: str) -> pd.DataFrame:
    hdr = detect_header_row(xlsx_path)
    df = pd.read_excel(xlsx_path, header=hdr)
    df.columns = [str(c).strip().upper() for c in df.columns]

    lat_col_candidates = [c for c in df.columns if 'LAT' in c]
    lon_col_candidates = [c for c in df.columns if 'LON' in c]
    if not lat_col_candidates or not lon_col_candidates:
        raise ValueError("No se encontraron columnas de latitud/longitud.")

    lat_col = lat_col_candidates[0]
    lon_col = lon_col_candidates[0]

    if 'NOMBRE.1' in df.columns:
        name_col = 'NOMBRE.1'
    elif 'MUNICIPIO' in df.columns:
        name_col = 'MUNICIPIO'
    else:
        cands = [c for c in df.columns if 'NOMBRE' in c]
        if not cands:
            tmp = df[[lat_col, lon_col]].dropna().copy()
            tmp['name'] = [f"NODO_{i}" for i in range(len(tmp))]
            clean = tmp.rename(columns={lat_col:'lat', lon_col:'lon'})
            clean['name_norm'] = clean['name']
            return clean[['lat','lon','name','name_norm']]
        name_col = cands[0]

    clean = df[[lat_col, lon_col, name_col]].dropna().rename(
        columns={lat_col:'lat', lon_col:'lon', name_col:'name'}
    ).reset_index(drop=True)

    def strip_accents(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

    clean['name_norm'] = clean['name'].apply(lambda s: strip_accents(s).upper().strip())
    return clean[['lat','lon','name','name_norm']]

def build_datos_dict(clean: pd.DataFrame) -> Dict[str, List]:
    return {
        'poblado': clean['name'].tolist(),
        'latitud': clean['lat'].astype(float).tolist(),
        'longitud': clean['lon'].astype(float).tolist(),
    }

R_EARTH_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(a))

def build_graph(clean: pd.DataFrame, d_max_km: float) -> Tuple[List[List[Tuple[int,float]]], int, float]:
    coords = clean[['lat','lon']].to_numpy(dtype=float)
    n = len(clean)
    adj: List[List[Tuple[int,float]]] = [[] for _ in range(n)]

    edge_count = 0
    t0 = time.time()
    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(i+1, n):
            lat2, lon2 = coords[j]
            d = float(haversine_km(lat1, lon1, lat2, lon2))
            if d <= d_max_km:
                adj[i].append((j, d))
                adj[j].append((i, d))
                edge_count += 1
    t1 = time.time()
    return adj, edge_count, (t1 - t0)

def dijkstra(adj: List[List[Tuple[int,float]]], src: int, tgt: Optional[int] = None):
    n = len(adj)
    dist = np.full(n, np.inf, dtype=float)
    prev = np.full(n, -1, dtype=int)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if tgt is not None and u == tgt:
            break
        for v,w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def reconstruct_path(prev: np.ndarray, src: int, tgt: int) -> List[int]:
    path = []
    u = tgt
    while u != -1:
        path.append(int(u))
        if u == src:
            break
        u = int(prev[u])
    path.reverse()
    if not path or path[0] != src:
        return []
    return path

def find_index_by_name(clean: pd.DataFrame, query: str) -> int:
    def norm(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').upper().strip()
    q = norm(query)
    arr = clean['name_norm'].to_numpy()
    matches = np.where(arr == q)[0]
    if len(matches) > 0:
        return int(matches[0])
    idx = [i for i,x in enumerate(arr) if q in x]
    if idx:
        return idx[0]
    raise ValueError(f"No se encontró '{query}' en la columna de nombres.")


def estimate_all_sources_time(adj, clean: pd.DataFrame, sample_k: int = 60, seed: int = 42) -> Tuple[float,float,float,int]:
    n = len(clean)
    rng = np.random.default_rng(seed)
    sources = rng.choice(n, size=min(sample_k, n), replace=False)

    t0 = time.time()
    total_reach = 0
    for s in sources:
        dist_all, _ = dijkstra(adj, int(s), tgt=None)
        total_reach += int(np.isfinite(dist_all).sum())
    t1 = time.time()

    sample_time = t1 - t0
    avg_time_per_source = sample_time / len(sources)
    est_time_all_sources = avg_time_per_source * n
    avg_reachable = total_reach // len(sources)
    return sample_time, avg_time_per_source, est_time_all_sources, avg_reachable


def plot_route(clean: pd.DataFrame, path_idx: List[int], title: str = "Ruta con saltos ≤ d km", out_png: Optional[str] = None):
    plt.figure(figsize=(6,8))
    plt.scatter(clean['lon'], clean['lat'], s=6, alpha=0.5, label="Municipios")
    if len(path_idx) > 1:
        coords = clean.iloc[path_idx][['lon','lat']].to_numpy()
        plt.plot(coords[:,0], coords[:,1], linewidth=2, label="Ruta")
        plt.scatter(coords[:,0], coords[:,1], s=12)
    plt.title(title)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    XLSX_PATH = "./data/DIVIPOLA_Municipios.xlsx"  
    D_MAX_KM = 60.0
    ORIG_NAME = "SAN GIL"
    DEST_NAME = "IPIALES"
    DO_EXPORT = True
    OUT_DIR = "./homeworks/results" 
    ESTIMATE_ALL_SOURCES = True

    clean = load_points_flexible(XLSX_PATH)
    n = len(clean)
    datos = build_datos_dict(clean)
    print(f"[INFO] Nodos cargados: {n}")

    adj, edge_count, build_time = build_graph(clean, D_MAX_KM)
    avg_deg = (2*edge_count)/n if n > 0 else 0.0
    print(f"[INFO] Aristas: {edge_count}  |  Grado promedio: {avg_deg:.2f}  |  Tiempo construcción: {build_time:.3f}s")

    src = find_index_by_name(clean, ORIG_NAME)
    tgt = find_index_by_name(clean, DEST_NAME)
    t0 = time.time()
    dist, prev = dijkstra(adj, src, tgt)
    t1 = time.time()

    path_idx = reconstruct_path(prev, src, tgt)
    if not path_idx:
        raise RuntimeError("No hay ruta con los saltos máximos especificados.")

    total_km = float(dist[tgt])
    hops = max(0, len(path_idx) - 1)
    one_to_one_time = t1 - t0

    print(f"[RUTA] {clean.iloc[src]['name']} → {clean.iloc[tgt]['name']}")
    print(f"  - Distancia total: {total_km:.3f} km")
    print(f"  - Saltos (aristas): {hops}")
    print(f"  - Tiempo Dijkstra (1 consulta): {one_to_one_time:.4f} s")

    if DO_EXPORT:
        os.makedirs(OUT_DIR, exist_ok=True)
        ruta_df = clean.iloc[path_idx][['name','lat','lon']].reset_index(drop=True)
        csv_path = os.path.join(OUT_DIR, f"ruta_{clean.iloc[src]['name']}_a_{clean.iloc[tgt]['name']}_d{int(D_MAX_KM)}km.csv")
        ruta_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[SAVE] CSV ruta: {csv_path}")

        png_path = os.path.join(OUT_DIR, f"ruta_plot_d{int(D_MAX_KM)}km.png")
        plot_route(clean, path_idx, title=f"Ruta {clean.iloc[src]['name']} → {clean.iloc[tgt]['name']} (≤ {int(D_MAX_KM)} km)", out_png=png_path)
        print(f"[SAVE] PNG ruta: {png_path}")

    if ESTIMATE_ALL_SOURCES:
        sample_time, avg_time_per_src, est_time_all, avg_reach = estimate_all_sources_time(adj, clean, sample_k=60, seed=42)
        print("[ESTIMACIÓN TODOS LOS ORÍGENES]")
        print(f"  - Tiempo en muestra (60 orígenes): {sample_time:.3f} s")
        print(f"  - Promedio por origen: {avg_time_per_src:.5f} s")
        print(f"  - Estimado para {n} orígenes: {est_time_all:.3f} s")
        print(f"  - Nodos alcanzables promedio por origen (con d={int(D_MAX_KM)}): {avg_reach}")
