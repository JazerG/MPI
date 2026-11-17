# 1. Parallel Word Counting in a Text Corpus with MPI

Lab 2 — Parallel and distributed computing with MPI (`mpi4py`).

---

# 2. Problem Description

`generator.py` builds a synthetic Spanish corpus in `dataset/`:

* `consulta.txt`: 200 query words, one per line.
* `file_0001.txt … file_3000.txt`: 3000 text files, **44,951,458 tokens (≈275 MB)** in total.
  File sizes are deliberately heterogeneous: ~60% small (2k–5k words), ~30% medium (10k–20k) and ~10% large (50k–100k). Words are drawn with Zipf-like weights (`weighted=True`), so a few words are very frequent.

The task is to count how many times each query word appears in the whole corpus and to report the 10 most frequent ones. The work is *embarrassingly parallel*: every file can be processed independently, and the partial counts only need to be added at the end. The interesting questions are therefore (a) how close to the ideal speedup we can get and (b) how the way files are distributed among processes affects the load balance.

---

# 3. Environment and Execution Instructions

## Repository content

| File | Description |
|---|---|
| `generator.py` | Dataset generator (as provided, unchanged) |
| `baseline_secuencial.py` | Sequential reference (as provided, unchanged) |
| `run_all.sh` | Batch experiment script (as provided, unchanged) |
| `mpi1.py` | MPI version 1: broadcast + static block distribution |
| `mpi2.py` | MPI version 2: broadcast + size-aware (LPT) distribution |
| `analyze_results.py` | Helper: reads `results/logs/` and builds the tables (T_p, S_p, E_p, LI) in `results/summary.md` |

## Environment used for the measurements reported below

| Item | Value |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, **2 vCPUs** (virtual machine) |
| RAM | 7 GB |
| OS | Linux 6.18 |
| Python | 3.13 |
| MPI | Open MPI 5.0.11, mpi4py 4.1.2 |
| Processes `p` | **2** (number of processes/cores detected on this machine — see 3.a) |

## How to run

All commands are executed from the `word_lab/` folder. Use `"%cd%:/app"` instead of `"$(pwd)":/app` on the Windows command prompt.

```bash
# 0. Detect how many processes/cores are available (p)
docker run --rm augustosalazar/slim-mpi:2 nproc

# 1. Generate the dataset
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py

# 2. Sequential baseline
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# 3. MPI version 1 and version 2 (replace 4 by the detected p)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root -np 4 python /app/mpi1.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root -np 4 python /app/mpi2.py

# 4. Full experiment (dataset + baseline + RUNS runs of each version with p = NP_LIST)
docker run --rm -v "$(pwd)":/app -e NP_LIST=4 -e RUNS=5 augustosalazar/slim-mpi:2 sh /app/run_all.sh

# 5. Extra baseline runs (run_all.sh runs the baseline only once)
for i in 2 3 4 5; do
  docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh -c \
    "python /app/baseline_secuencial.py > /app/results/logs/baseline_secuencial_run$i.log"
done

# 6. Build the tables
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/analyze_results.py
```

Both MPI programs print a human-readable report plus machine-readable lines (`PROC rank=… local_time=…`, `LI=…`, `CHECK=…`, `EXECUTION_TIME=…`). `run_all.sh` extracts `EXECUTION_TIME` into `results/times_summary.csv`, and `analyze_results.py` uses the `PROC` lines to compute the load-imbalance metrics.

## 3.a Experimental Plan

| Item | Decision |
|---|---|
| Independent variable | Distribution strategy: none (sequential), static blocks (v1), size-aware LPT (v2) |
| Fixed factors | Same dataset (`seed=42`, 3000 files, 200 query words), same machine, same per-file counting code, same `p` |
| Number of processes | `p` = number of processes detected on the machine (`nproc`) = **2**. No oversubscription, so each process owns a core |
| Repetitions | **5 runs** per program (the lab asks for at least 3) |
| Measured variables | Total execution time; local processing time `t_i` of every process; number of files and bytes per process |
| Derived metrics | `T_seq` = mean baseline time; `T_p` = mean MPI total time; `S_p = T_seq / T_p`; `E_p = S_p / p`; `LI = max(t_i) / mean(t_i)` per run (averaged over runs); relative spread `(max − min) / mean` |
| Correctness criterion | The full count table of every MPI run must be **identical** to `baseline_results.csv` (not only the top 10). Each run prints `CHECK=OK/MISMATCH` |
| Warm-up / cache | The dataset is generated right before the runs, so the 275 MB corpus is in the OS page cache for every program (all programs measured under the same warm-cache condition) |
| Timing | Baseline: `time.perf_counter()` around reading the query + counting (CSV writing excluded). MPI: `MPI.Wtime()` from a `Barrier` at start until rank 0 has merged the global counter (CSV writing excluded), so both include the same phases |

## 3.b Sequential baseline (what it does and how)

`baseline_secuencial.py` reads `consulta.txt` into a `set` (lower-cased), iterates over every `file_*.txt` in `dataset/`, splits each line with `str.split()`, lower-cases the tokens, and increments a `collections.Counter` for every token that belongs to the query set. It then writes `baseline_results.csv` (word, count) and prints the elapsed time and the top 10. It is the reference for correctness and for `T_seq`.

## 3.c MPI version 1 (what it does and how)

`mpi1.py` follows exactly the required pipeline:

1. **rank 0** reads `consulta.txt`;
2. **rank 0** broadcasts the query words to every process with `comm.bcast`;
3. **rank 0** lists the `file_*.txt` files (sorted by name);
4. the list is split **statically into `p` contiguous blocks with the same number of files** (±1) and delivered with `comm.scatter`;
5. each process counts the query words in its files, using the *same* per-file logic as the baseline, and measures its local time `t_i` with `MPI.Wtime()`;
6. the partial counters and per-process statistics are collected in rank 0 with `comm.gather`;
7. rank 0 merges the counters, prints the per-process report (files, MB, tokens, local time), `LI`, the relative spread and the top 10, and checks the full table against `baseline_results.csv`.

The data exchanged is tiny (200 words, 3000 file names, `p` counters of at most 200 entries), so communication cost is negligible compared with ~275 MB of text processing.

## 3.d Test procedure

1. Generate the dataset and run the baseline (saves `baseline_results.csv`, the correctness oracle).
2. Run `run_all.sh` with `NP_LIST=<p>` and `RUNS=5`; it regenerates the dataset (same seed, identical files), runs the baseline once and each MPI version 5 times, saving one log per run.
3. Run the baseline 4 more times to obtain a 5-run `T_seq`.
4. Verify that every MPI log reports `CHECK=OK` and the same top 10 as the baseline.
5. Run `analyze_results.py` to compute `T_p`, `S_p`, `E_p`, `LI` and spread from the logs.
6. Because `p = 2` hides most of the structural imbalance, also compute the **bytes assigned to each process** by both partitioning functions for `p = 2 … 16` (deterministic, no timing noise) as supporting evidence.

---

# 4. Experimental Plan Execution

**Correctness:** all 10 MPI runs reported `CHECK=OK` — the complete 200-word count table equals the baseline, and the top 10 is:

```
a: 785774 | para: 392156 | sus: 228913 | otros: 105530 | ante: 99832
unos: 88794 | otra: 83901 | vosotros: 61617 | mios: 58420 | tuya: 56635
```

(identical to the reference output published with the lab; files processed 3000, tokens 44,951,458, occurrences 3,631,778).

## 4.a Sequential baseline timing

| Run | Time (s) |
|---|---|
| 1 | 6.5481 |
| 2 | 6.9279 |
| 3 | 7.0672 |
| 4 | 6.9767 |
| 5 | 6.3711 |
| **Mean (T_seq)** | **6.7782** (std 0.3017) |

Throughput ≈ 275 MB / 6.78 s ≈ **40.6 MB/s** on one core.

## 4.b MPI version 1 timing results (p = 2)

| Run | Total time (s) | LI = max/mean | Spread (max−min)/mean | Check |
|---|---|---|---|---|
| 1 | 3.3871 | 1.0141 | 0.0283 | OK |
| 2 | 3.4800 | 1.0208 | 0.0415 | OK |
| 3 | 3.3431 | 1.0335 | 0.0671 | OK |
| 4 | 3.4654 | 1.0102 | 0.0203 | OK |
| 5 | 3.3684 | 1.0265 | 0.0531 | OK |
| **Mean** | **3.4088** (std 0.0606) | **1.0210** | **0.0421** | |

* **S_p = 6.7782 / 3.4088 = 1.988**
* **E_p = 1.988 / 2 = 0.994**

## 4.c Load imbalance evidence (v1)

Local processing time per process (s):

| Rank | Files | MB | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean |
|---|---|---|---|---|---|---|---|---|
| 0 | 1500 | 140.48 | 3.2849 | 3.4705 | 3.3340 | 3.3888 | 3.3578 | 3.3672 |
| 1 | 1500 | 134.57 | 3.3790 | 3.3293 | 3.1177 | 3.4583 | 3.1842 | 3.2937 |

* Both processes receive **exactly 1500 files**, but rank 0 receives **5.9 MB (4.4%) more text** because block 1 happens to contain more large files. On average rank 0 was 2.2% slower than rank 1, and the mean `LI` was **1.021** (spread 4.2%).
* The imbalance is real but small at `p = 2`, and it is mixed with timing noise: in runs 1 and 4 rank 1 was the slowest despite having less data.
* The imbalance is **structural** (equal *file counts* ≠ equal *work*) and grows with `p`, because each block contains fewer files and the 10% of large files are spread less evenly. Bytes per process produced by the v1 partition for this dataset:

| p | v1 bytes LI | v1 max / min MB per process | v2 bytes LI | v2 max / min MB |
|---|---|---|---|---|
| 2 | 1.0215 | 140.5 / 134.6 | 1.0000 | 137.5 / 137.5 |
| 4 | 1.0818 | 74.4 / 66.1 | 1.0000 | 68.8 / 68.8 |
| 6 | 1.0438 | 47.9 / 43.4 | 1.0000 | 45.8 / 45.8 |
| 8 | 1.1000 | 37.8 / 29.4 | 1.0000 | 34.4 / 34.4 |
| 12 | 1.1791 | 27.0 / 18.2 | 1.0000 | 22.9 / 22.9 |
| 16 | 1.2909 | 22.2 / 13.5 | 1.0004 | 17.2 / 17.2 |

Since the processing cost is linear in the amount of text (≈40–42 MB/s per process in every program), the bytes-LI is a good predictor of the time-LI once timing noise is removed.

## 4.d MPI version 2: how it corrects the imbalance, and its results

**Idea.** The cost of a file is proportional to its size, not to the fact of being one file. v2 keeps the same pipeline as v1 (`bcast` → `scatter` → local count → `gather`) and changes only step 4: rank 0 obtains the size of every file (`os.path.getsize`, a metadata call with negligible cost) and applies the **LPT (Longest Processing Time first)** greedy rule — sort files by size, largest first, and assign each one to the process with the smallest accumulated number of bytes (min-heap). This yields practically identical byte loads (137.53 MB per process at `p = 2`, LI_bytes = 1.0000 up to `p = 12`) without adding any communication round. A dynamic master–worker scheme was not chosen because the cost of each file is known in advance, so a static size-aware assignment achieves the same balance without request/response messages and without dedicating rank 0 as a coordinator.

**Results (p = 2):**

| Run | Total time (s) | LI = max/mean | Spread (max−min)/mean | Check |
|---|---|---|---|---|
| 1 | 3.3814 | 1.0064 | 0.0129 | OK |
| 2 | 3.3124 | 1.0028 | 0.0056 | OK |
| 3 | 3.4349 | 1.0237 | 0.0475 | OK |
| 4 | 3.3977 | 1.0094 | 0.0188 | OK |
| 5 | 3.4341 | 1.0455 | 0.0910 | OK |
| **Mean** | **3.3921** (std 0.0502) | **1.0176** | **0.0352** | |

| Rank | Files | MB | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean |
|---|---|---|---|---|---|---|---|---|
| 0 | 1499 | 137.53 | 3.3175 | 3.2978 | 3.2622 | 3.3148 | 3.1230 | 3.2631 |
| 1 | 1501 | 137.53 | 3.3605 | 3.2795 | 3.4209 | 3.3778 | 3.4208 | 3.3719 |

**Comparison**

| Version | p | T_p mean (s) | Speedup S_p | Efficiency E_p | Mean LI | Worst LI |
|---|---|---|---|---|---|---|
| Sequential | 1 | 6.7782 ± 0.3017 | 1.000 | 1.000 | — | — |
| mpi1 (static blocks) | 2 | 3.4088 ± 0.0606 | 1.988 | 0.994 | 1.0210 | 1.0335 |
| mpi2 (size-aware LPT) | 2 | 3.3921 ± 0.0502 | 1.998 | 0.999 | 1.0176 | 1.0455 |

---

# 5. Analysis

## a. Parallel performance

Yes. Version 1 cut the time from 6.78 s to 3.41 s, a speedup of **1.988** with **99.4% efficiency**, i.e. within 0.6% of the ideal `S_p = 2`. This is expected for an embarrassingly parallel problem in which the only serial parts — listing 3000 files, broadcasting 200 words, scattering 3000 names and merging two 200-entry counters — take about 0.009 s in v1 (the gap between `T_p` and the slowest local time, averaged over the runs; 0.017 s in v2, whose extra ~0.008 s is the 3000 `getsize` calls). The small remaining gap is explained mainly by load imbalance: `T_p` is dictated by the slowest process, and in v1 the slowest process was on average 2.1% above the mean (`LI = 1.021`). An efficiency this close to 1 should also be read with care, because the baseline itself varies by ±0.30 s (4.5%) between runs; the speedup is essentially ideal, but the third decimal is not meaningful.

## b. Load balance

Version 1 showed a real, data-driven imbalance: both ranks processed 1500 files, but rank 0 received 4.4% more bytes and was on average 2.2% slower, giving a mean LI of 1.021 and a spread of 4.2%. Version 2 eliminated the *data* imbalance completely (137.53 MB each) and lowered the mean LI to **1.018** and the spread to **3.5%**, with a slightly better `T_p` (3.392 vs 3.409 s; S_p 1.998 vs 1.988). However, at `p = 2` the improvement is **not** large enough to be called a real performance gain: the 0.017 s difference (0.5%) is smaller than the run-to-run standard deviation (0.05–0.06 s), and a Welch two-sample t-test gives t = 0.47 (p = 0.65), far from significance. Part of v2's gain is also given back by its slightly larger serial overhead (0.017 s vs 0.009 s). The remaining LI in v2 comes from the system, not the data — with identical byte loads, rank 1 was consistently ~3% slower, which points to the virtual machine (CPU contention, scheduling, cache effects). The bytes analysis in 4.c shows where v2 does matter: the structural LI of static blocks grows to 1.08 at p = 4, 1.10 at p = 8 and 1.29 at p = 16, whereas LPT stays at 1.000, so the expected reduction of the critical path is about 8%, 9% and 22% respectively.

## c. Limitations

* **Only two cores (p = 2).** This is the main limitation. With two processes the structural imbalance of v1 is only ~2%, which is below the timing noise, so the experiment cannot demonstrate a statistically significant difference between v1 and v2. The higher-`p` evidence is a byte-level model, not a timed measurement; it is credible because the throughput was linear in bytes (~41 MB/s per process in all programs), but it ignores effects that appear at larger `p` (memory bandwidth, page-cache contention).
* **Shared virtual machine.** The baseline varied by 4.5% and the per-rank times by several percent, even with identical loads. This noise is comparable to the effect being measured.
* **Five runs.** Enough for a mean and a standard deviation, but too few for tight confidence intervals on differences of < 1%.
* **Warm page cache.** The 275 MB corpus fits in RAM and was cached after generation, so the problem is CPU-bound. With a cold cache or a slower disk, I/O contention could reduce the speedup.
* **One dataset, one seed.** A corpus with more extreme size variance (or fewer, larger files) would make v1 worse and v2's benefit clearer.
* **Native environment.** The reported numbers were taken with Open MPI 5 / Python 3.13 on the host instead of the `slim-mpi` container; absolute times will differ in Docker, though the relative comparisons should hold.

Overall, the conclusion that MPI gives near-ideal speedup is strong; the conclusion that v2 is faster than v1 is **not** supported by the timings at `p = 2` and rests on the byte-level analysis for larger `p`.

---

# 6. Conclusions

For this problem I would recommend the size-aware distribution of MPI version 2 (LPT on file bytes). Both MPI versions produced exactly the baseline counts and nearly ideal scaling on 2 processes — v1 reached S_p = 1.988 and E_p = 0.994, v2 reached S_p = 1.998 and E_p = 0.999 — and v2 also reduced the mean load imbalance from LI = 1.021 to 1.018, with a perfect byte split (137.53 MB per process) instead of a 4.4% difference. At p = 2 these differences are within the measurement noise (std ≈ 0.05–0.06 s), so the timings alone show that v2 is *at least as good as* v1, not that it is clearly faster. The recommendation rests on the fact that v2 costs nothing extra (one `stat` per file and the same single `scatter`), while the structural imbalance of equal-count blocks grows with the number of processes on this heterogeneous corpus (bytes-LI 1.08 at p = 4, 1.10 at p = 8, 1.29 at p = 16, against 1.00 for LPT). A dynamic work queue would only be worth its extra messages if the processes ran on nodes with different speeds, where a static assignment cannot adapt.
