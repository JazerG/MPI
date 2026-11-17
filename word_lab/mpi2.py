"""
MPI version 2 - Broadcast + size-aware (LPT) static distribution.

Version 1 gives every process the same NUMBER of files, but the corpus is
heterogeneous: generator.py makes ~60% small files (2k-5k words), ~30%
medium (10k-20k) and ~10% large (50k-100k). Work is proportional to the
number of BYTES/TOKENS, not to the number of files, so equal file counts
can still produce unequal work.

Version 2 keeps the same pipeline (bcast of the query, scatter of the file
lists, gather of partial counters) but rank 0 balances the WORK instead of
the file count, using the Longest-Processing-Time-first (LPT) greedy rule:

  1. get the size in bytes of every file (os.path.getsize, a cheap stat)
  2. sort files by size, largest first
  3. assign each file to the process with the smallest accumulated bytes
     (min-heap)

LPT is guaranteed to be within 4/3 of the optimal makespan, and with
thousands of files the per-process byte totals end up practically equal.
It adds no extra communication rounds: it is still a single scatter.

Run:
  mpirun --allow-run-as-root -np 4 python mpi2.py
"""

import heapq
import os
import sys
from collections import Counter

from mpi4py import MPI

CONSULTA_NAME = "consulta.txt"
CASE_SENSITIVE = False
TOP_N = 10
OUTPUT_FILE = "mpi2_results.csv"
BASELINE_FILE = "baseline_results.csv"
VERSION = "mpi2"


# --------------------------------------------------------------------------
# Helpers (same semantics as the sequential baseline)
# --------------------------------------------------------------------------
def cargar_consulta(consulta_path, case_sensitive=False):
    with open(consulta_path, "r", encoding="utf-8") as f:
        palabras = [line.strip() for line in f if line.strip()]
    if not case_sensitive:
        palabras = [w.lower() for w in palabras]
    return sorted(set(palabras))


def listar_archivos(dataset_dir):
    archivos = []
    for fname in os.listdir(dataset_dir):
        if fname.startswith("file_") and fname.endswith(".txt"):
            ruta = os.path.join(dataset_dir, fname)
            if os.path.isfile(ruta):
                archivos.append(fname)
    archivos.sort()  # deterministic order: file_0001, file_0002, ...
    return archivos


def contar_en_archivos(dataset_dir, archivos, palabras_objetivo, case_sensitive=False):
    """Counts query-word occurrences in the given files (baseline logic)."""
    freq = Counter()
    tokens = 0
    nbytes = 0
    for fname in archivos:
        ruta = os.path.join(dataset_dir, fname)
        nbytes += os.path.getsize(ruta)
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                palabras = linea.split()
                if not case_sensitive:
                    palabras = [w.lower() for w in palabras]
                tokens += len(palabras)
                for w in palabras:
                    if w in palabras_objetivo:
                        freq[w] += 1
    return freq, tokens, nbytes


def particion_lpt_por_tamano(dataset_dir, archivos, p):
    """Greedy LPT: largest file first, to the least-loaded process (in bytes)."""
    tamanos = [(os.path.getsize(os.path.join(dataset_dir, f)), f) for f in archivos]
    tamanos.sort(reverse=True)
    heap = [(0, r) for r in range(p)]          # (accumulated bytes, rank)
    bloques = [[] for _ in range(p)]
    for size, fname in tamanos:
        carga, r = heapq.heappop(heap)
        bloques[r].append(fname)
        heapq.heappush(heap, (carga + size, r))
    for b in bloques:
        b.sort()
    return bloques


# --------------------------------------------------------------------------
# Reporting (rank 0 only)
# --------------------------------------------------------------------------
def guardar_resultados_csv(path, freq_global):
    with open(path, "w", encoding="utf-8") as f:
        f.write("palabra,conteo\n")
        for palabra in sorted(freq_global):
            f.write(f"{palabra},{freq_global[palabra]}\n")


def comparar_con_baseline(baseline_path, freq_global):
    if not os.path.isfile(baseline_path):
        return "NOT_AVAILABLE (run baseline_secuencial.py first)"
    esperado = {}
    with open(baseline_path, "r", encoding="utf-8") as f:
        next(f)
        for linea in f:
            palabra, conteo = linea.rstrip("\n").rsplit(",", 1)
            esperado[palabra] = int(conteo)
    return "OK" if esperado == dict(freq_global) else "MISMATCH"


def reportar(stats, freq_global, n_archivos, elapsed, dataset_dir, p):
    total_tokens = sum(s["tokens"] for s in stats)
    tiempos = [s["local_time"] for s in stats]
    media = sum(tiempos) / len(tiempos)
    li = max(tiempos) / media if media > 0 else float("nan")
    spread = (max(tiempos) - min(tiempos)) / media if media > 0 else float("nan")

    resultados_path = os.path.join(dataset_dir, OUTPUT_FILE)
    guardar_resultados_csv(resultados_path, freq_global)
    check = comparar_con_baseline(os.path.join(dataset_dir, BASELINE_FILE), freq_global)

    print(f"[{VERSION}] Size-aware LPT distribution, p = {p}")
    print(f"Execution time: {elapsed:.6f} s")
    print(f"Files processed: {n_archivos}")
    print(f"Tokens read: {total_tokens}")
    print(f"Total occurrences found: {sum(freq_global.values())}")
    print(f"Results saved to: {resultados_path}")
    print(f"Check against baseline: {check}\n")

    print("Per-process report:")
    print(f"  {'rank':>4} {'files':>6} {'MB':>8} {'tokens':>10} {'local_time_s':>13}")
    for s in stats:
        print(f"  {s['rank']:>4} {s['n_files']:>6} {s['bytes']/1e6:>8.2f} "
              f"{s['tokens']:>10} {s['local_time']:>13.6f}")
    print(f"\nLocal time  max={max(tiempos):.6f}  mean={media:.6f}  min={min(tiempos):.6f}")
    print(f"Load imbalance LI = max/mean = {li:.4f}")
    print(f"Relative spread (max-min)/mean = {spread:.4f}\n")

    print(f"Top {TOP_N} query words in the corpus:")
    for palabra, cuenta in freq_global.most_common(TOP_N):
        print(f"  {palabra}: {cuenta}")

    # Machine-readable lines (used by run_all.sh and analyze_results.py)
    for s in stats:
        print(f"PROC rank={s['rank']} files={s['n_files']} bytes={s['bytes']} "
              f"tokens={s['tokens']} local_time={s['local_time']:.6f}")
    print(f"LI={li:.6f}")
    print(f"CHECK={check}")
    print(f"EXECUTION_TIME={elapsed:.6f}")
    sys.stdout.flush()


# --------------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    comm.Barrier()                     # all processes start together
    t0 = MPI.Wtime()

    # 1-3. rank 0 reads the query and lists the corpus
    consulta, bloques, error = None, None, None
    if rank == 0:
        try:
            consulta = cargar_consulta(os.path.join(dataset_dir, CONSULTA_NAME), CASE_SENSITIVE)
            archivos = listar_archivos(dataset_dir)
            # 4. size-aware distribution (same number of BYTES per process)
            bloques = particion_lpt_por_tamano(dataset_dir, archivos, p)
        except (FileNotFoundError, NotADirectoryError) as e:
            error = str(e)

    error = comm.bcast(error, root=0)
    if error:
        if rank == 0:
            print("Error:", error, "- generate the dataset with generator.py first")
        return

    # 2. broadcast of the query words
    consulta = comm.bcast(consulta, root=0)
    palabras_objetivo = set(consulta)

    # 4. each process receives its balanced list of files
    mis_archivos = comm.scatter(bloques, root=0)

    # 5. local counting
    t_local0 = MPI.Wtime()
    freq_local, tokens, nbytes = contar_en_archivos(
        dataset_dir, mis_archivos, palabras_objetivo, CASE_SENSITIVE)
    local_time = MPI.Wtime() - t_local0

    stats_local = {"rank": rank, "n_files": len(mis_archivos), "bytes": nbytes,
                   "tokens": tokens, "local_time": local_time}

    # 6. gather partial results in rank 0
    parciales = comm.gather((dict(freq_local), stats_local), root=0)

    # 7. global result
    if rank == 0:
        freq_global = Counter()
        for freq, _ in parciales:
            freq_global.update(freq)
        elapsed = MPI.Wtime() - t0
        stats = [s for _, s in parciales]
        n_archivos = sum(s["n_files"] for s in stats)
        reportar(stats, freq_global, n_archivos, elapsed, dataset_dir, p)


if __name__ == "__main__":
    main()
