import argparse
import math
import unicodedata
from typing import Dict, List, Tuple, Iterable
import pandas as pd
import numpy as np
import sys


def detect_header_row(xlsx_path: str, max_search: int = 30) -> int:
    raw = pd.read_excel(xlsx_path, header=None)
    for i in range(min(max_search, len(raw))):
        row = raw.iloc[i].astype(str).str.upper().tolist()
        if any('LATITUD' == x for x in row) and any('LONGITUD' == x for x in row):
            return i
    return 0

def load_points_flexible(xlsx_path: str) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: lat, lon, name, name_norm (si name existe)."""
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
    elif 'NOMBRE' in df.columns:
        name_col = 'NOMBRE'
    elif 'POBLADO' in df.columns:
        name_col = 'POBLADO'
    else:
        name_col = None

    if name_col is None:
        tmp = df[[lat_col, lon_col]].dropna().copy()
        tmp['name'] = [f"NODO_{i}" for i in range(len(tmp))]
        clean = tmp.rename(columns={lat_col:'lat', lon_col:'lon'})
        clean['name_norm'] = clean['name']
        return clean[['lat','lon','name','name_norm']]

    clean = df[[lat_col, lon_col, name_col]].dropna().rename(
        columns={lat_col:'lat', lon_col:'lon', name_col:'name'}
    ).reset_index(drop=True)

    def strip_accents(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

    clean['name_norm'] = clean['name'].apply(lambda s: strip_accents(s).upper().strip())
    return clean[['lat','lon','name','name_norm']]

def load_points_rigida(xlsx_path: str) -> pd.DataFrame:
    """Versión 'rígida' que espera LAT*, LON* y opcionalmente NOMBRE/MUNICIPIO/POBLADO."""
    hdr = detect_header_row(xlsx_path)
    df = pd.read_excel(xlsx_path, header=hdr)
    df.columns = [str(c).strip().upper() for c in df.columns]
    name_col_candidates = [c for c in df.columns if 'NOMBRE' in c or 'MUNICIPIO' in c or 'POBLADO' in c]
    lat_col_candidates = [c for c in df.columns if 'LAT' in c]
    lon_col_candidates = [c for c in df.columns if 'LON' in c]
    if not lat_col_candidates or not lon_col_candidates:
        raise ValueError("No se encontraron columnas de LAT/LON.")
    name_col = name_col_candidates[0] if name_col_candidates else None
    lat_col = lat_col_candidates[0]
    lon_col = lon_col_candidates[0]
    keep = [c for c in [name_col, lat_col, lon_col] if c is not None]
    clean = df[keep].dropna()
    rename_map = {}
    if name_col: rename_map[name_col] = 'name'
    rename_map[lat_col] = 'lat'
    rename_map[lon_col] = 'lon'
    clean = clean.rename(columns=rename_map).reset_index(drop=True)

    if 'name' in clean.columns:
        def strip_accents(s: str) -> str:
            return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
        clean['name_norm'] = clean['name'].apply(lambda s: strip_accents(s).upper().strip())
    else:
        clean['name'] = [f"NODO_{i}" for i in range(len(clean))]
        clean['name_norm'] = clean['name']
    return clean[['lat','lon','name','name_norm']]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlambda/2)**2)
    return 2*R*math.asin(math.sqrt(a))

def distance_matrix(coords: List[Tuple[float,float]]) -> np.ndarray:
    n = len(coords)
    D = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            D[i,j] = D[j,i] = d
    return D

def held_karp(D: np.ndarray) -> Tuple[float, List[int]]:
    n = D.shape[0]
    dp: Dict[Tuple[int,int], Tuple[float,int]] = {(1<<0, 0): (0.0, -1)} 
    for r in range(2, n+1):
        for subset in (mask for mask in range(1<<n) if (mask & 1) and (mask.bit_count()==r)):
            for j in range(1, n):
                if not (subset & (1<<j)): 
                    continue
                best = (float("inf"), -1)
                prevmask = subset ^ (1<<j)
                for k in range(n):
                    if k==j or not (prevmask & (1<<k)):
                        continue
                    if (prevmask, k) in dp:
                        cand = dp[(prevmask,k)][0] + D[k,j]
                        if cand < best[0]:
                            best = (cand, k)
                if best[0] < float("inf"):
                    dp[(subset, j)] = best

    full = (1<<n) - 1
    best = (float("inf"), -1)
    for j in range(1, n):
        if (full, j) in dp:
            cand = dp[(full, j)][0] + D[j, 0]
            if cand < best[0]:
                best = (cand, j)

    tour = [0]
    mask = full
    j = best[1]
    while j != -1:
        tour.append(j)
        mask, j = mask ^ (1<<j), dp[(mask, tour[-1])][1]
    tour.reverse()
    tour = tour[:-1] 
    return best[0], tour

def mst_prim(D: np.ndarray) -> List[Tuple[int,int]]:
    n = len(D)
    in_mst = [False]*n
    in_mst[0] = True
    edges = []
    while len(edges) < n-1:
        best = (float("inf"), -1, -1)
        for u in range(n):
            if not in_mst[u]: 
                continue
            for v in range(n):
                if in_mst[v]: 
                    continue
                if D[u,v] < best[0]:
                    best = (D[u,v], u, v)
        _, u, v = best
        edges.append((u,v))
        in_mst[v] = True
    return edges

def degree_from_edges(n: int, edges: Iterable[Tuple[int,int]]) -> List[int]:
    deg = [0]*n
    for u,v in edges:
        deg[u]+=1; deg[v]+=1
    return deg

def greedy_min_matching(nodes: List[int], D: np.ndarray) -> List[Tuple[int,int]]:
    nodes = list(nodes)
    used = set()
    match = []
    pairs = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            pairs.append((D[u,v], u, v))
    for _,u,v in sorted(pairs):
        if u in used or v in used:
            continue
        used.add(u); used.add(v)
        match.append((u,v))
    return match

def eulerian_tour(n: int, edges: List[Tuple[int,int]]) -> List[int]:
    adj = {i: [] for i in range(n)}
    for idx,(u,v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    used = set()
    stack = [0]
    path = []
    while stack:
        u = stack[-1]
        while adj[u] and adj[u][-1][1] in used:
            adj[u].pop()
        if not adj[u]:
            path.append(stack.pop())
        else:
            v, eid = adj[u].pop()
            if eid in used:
                continue
            used.add(eid)
            stack.append(v)
    return path[::-1]

def shortcut_to_hamiltonian(path: List[int]) -> List[int]:
    seen = set()
    tour = []
    for v in path:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    tour.append(tour[0])
    return tour[:-1]

def two_opt(route: List[int], D: np.ndarray) -> Tuple[float, List[int]]:
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                a, b = route[i-1], route[i]
                c, d = route[j], route[(j+1)%n]
                delta = (D[a,c] + D[b,d]) - (D[a,b] + D[c,d])
                if delta < -1e-9:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
    length = sum(D[route[i], route[(i+1)%n]] for i in range(n))
    return float(length), route

def christofides_2opt(D: np.ndarray) -> Tuple[float, List[int]]:
    n = len(D)
    mst = mst_prim(D)
    deg = degree_from_edges(n, mst)
    odd = [i for i,d in enumerate(deg) if d%2==1]
    matching = greedy_min_matching(odd, D) 
    edges = mst + matching
    path = eulerian_tour(n, edges)
    tour = shortcut_to_hamiltonian(path)
    length = sum(D[tour[i], tour[(i+1)%n]] for i in range(n))
    length, tour = two_opt(tour, D)
    return float(length), tour


def _ru_maxrss_bytes() -> int:
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(ru if sys.platform == "darwin" else ru * 1024)
    except Exception:
        return 0

def measure_time_and_memory(fn, *args, **kwargs):
    import time, tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = {
        "time_s": float(t1 - t0),
        "peak_tracemalloc_bytes": int(peak_traced),
    }
    return result, metrics

def fmt_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def normalize(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn').upper().strip()

def pick_by_names(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    wanted_norm = [normalize(x) for x in names]
    idx_map = {n:i for i, n in enumerate(wanted_norm)}
    sub = df[df["name_norm"].isin(idx_map.keys())].copy()
    sub["order"] = sub["name_norm"].map(idx_map)
    sub = sub.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return sub

def pick_first_n(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def read_names_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def tour_to_table(df: pd.DataFrame, D: np.ndarray, tour: List[int], length_km: float) -> pd.DataFrame:
    rows = []
    n = len(tour)
    for i in range(n):
        u = tour[i]
        v = tour[(i+1)%n]
        rows.append({
            "Paso": i+1,
            "Desde": df.iloc[u]["name"],
            "→": "→",
            "Hasta": df.iloc[v]["name"],
            "Distancia (km)": round(float(D[u,v]), 2)
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="TSP para 11/15/29 ciudades con Excel de municipios.")
    parser.add_argument("--xlsx", required=True, help="Ruta al archivo Excel (p.ej. DIVIPOLA_Municipios.xlsx)")
    parser.add_argument("--rigida", action="store_true", help="Usar loader rígido en lugar del flexible")
    parser.add_argument("--names11", type=str, help="Archivo con 11 nombres (uno por línea)")
    parser.add_argument("--names15", type=str, help="Archivo con 15 nombres")
    parser.add_argument("--names29", type=str, help="Archivo con 29 nombres")
    parser.add_argument("--seed15", type=int, default=15, help="Semilla si falta lista de 15")
    parser.add_argument("--seed29", type=int, default=29, help="Semilla si falta lista de 29")
    args = parser.parse_args()

    loader = load_points_rigida if args.rigida else load_points_flexible
    df = loader(args.xlsx)

    default_11 = ["Medellín","Pereira","Bucaramanga","Cali","Leticia","Manizales",
                  "Barranquilla","Cartagena","Pasto","San Andrés","Cúcuta"]
    names11 = read_names_file(args.names11) if args.names11 else default_11
    df11 = pick_by_names(df, names11)
    if len(df11) != 11:
        print("[Aviso] No se encontraron todas las 11 ciudades; usando muestreo.")
        df11 = pick_first_n(df, 11, seed=11)

    if args.names15:
        names15 = read_names_file(args.names15)
        df15 = pick_by_names(df, names15)
        if len(df15) != 15:
            print("[Aviso] Lista de 15 incompleta; usando muestreo.")
            df15 = pick_first_n(df, 15, seed=args.seed15)
    else:
        df15 = pick_first_n(df, 15, seed=args.seed15)

    if args.names29:
        names29 = read_names_file(args.names29)
        df29 = pick_by_names(df, names29)
        if len(df29) != 29:
            print("[Aviso] Lista de 29 incompleta; usando muestreo.")
            df29 = pick_first_n(df, 29, seed=args.seed29)
    else:
        df29 = pick_first_n(df, 29, seed=args.seed29)

    D11 = distance_matrix(list(zip(df11["lat"], df11["lon"])))
    D15 = distance_matrix(list(zip(df15["lat"], df15["lon"])))
    D29 = distance_matrix(list(zip(df29["lat"], df29["lon"])))

    (len11, tour11), m11 = measure_time_and_memory(held_karp, D11)
    (len15, tour15), m15 = measure_time_and_memory(held_karp, D15)
    (len29, tour29), m29 = measure_time_and_memory(christofides_2opt, D29)

    tbl11 = tour_to_table(df11, D11, tour11, len11)
    tbl15 = tour_to_table(df15, D15, tour15, len15)
    tbl29 = tour_to_table(df29, D29, tour29, len29)

    tbl11.to_csv("./homeworks/results/tour_11_ciudades.csv", index=False)
    tbl15.to_csv("./homeworks/results/tour_15_ciudades.csv", index=False)
    tbl29.to_csv("./homeworks/results/tour_29_ciudades.csv", index=False)

    metrics_rows = [
        {"case":"11", "n":len(D11), "solver":"Held-Karp", "distance_km":round(len11, 6),
         "time_s":m11["time_s"],
         "peak_tracemalloc_bytes":m11["peak_tracemalloc_bytes"]},
        {"case":"15", "n":len(D15), "solver":"Held-Karp", "distance_km":round(len15, 6),
         "time_s":m15["time_s"],
         "peak_tracemalloc_bytes":m15["peak_tracemalloc_bytes"]},
        {"case":"29", "n":len(D29), "solver":"Christofides+2opt", "distance_km":round(len29, 6),
         "time_s":m29["time_s"],
         "peak_tracemalloc_bytes":m29["peak_tracemalloc_bytes"]},
    ]
    pd.DataFrame(metrics_rows).to_csv("./homeworks/results/metrics_tsp.csv", index=False)

    def show(case, length, m):
        print(f"[{case}] Distancia total: {length:.2f} km | "
              f"Tiempo: {m['time_s']:.4f} s | "
              f"Mem. Python (pico): {fmt_bytes(m['peak_tracemalloc_bytes'])}")

    show("11 (Held-Karp)", len11, m11)
    show("15 (Held-Karp)", len15, m15)
    show("29 (Christofides+2opt)", len29, m29)
    print("Tours: tour_11_ciudades.csv, tour_15_ciudades.csv, tour_29_ciudades.csv")
    print("Métricas: metrics_tsp.csv")

if __name__ == "__main__":
    main()