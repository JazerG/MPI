"""
Builds the experiment tables (Markdown) from the logs written by run_all.sh.

Reads:
  results/logs/baseline_secuencial*.log   ("Tiempo de ejecución: X segundos")
  results/logs/mpi{1,2}_np{P}_run{R}.log  (PROC ..., LI=..., EXECUTION_TIME=...)

Writes results/summary.md and prints it.

Usage (inside the container, after run_all.sh):
  python /app/analyze_results.py
"""

import glob
import os
import re
import statistics as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "results", "logs")
OUT = os.path.join(APP_DIR, "results", "summary.md")

PROC_RE = re.compile(r"PROC rank=(\d+) files=(\d+) bytes=(\d+) tokens=(\d+) local_time=([\d.]+)")


def fmean(xs):
    return sum(xs) / len(xs)


def fstd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def leer_baseline():
    tiempos = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "baseline_secuencial*.log"))):
        with open(path, encoding="utf-8") as f:
            m = re.search(r"Tiempo de ejecuci\S+n:\s*([\d.]+)", f.read())
        if m:
            tiempos.append(float(m.group(1)))
    return tiempos


def leer_mpi(version):
    runs = {}
    for path in glob.glob(os.path.join(LOG_DIR, f"{version}_np*_run*.log")):
        m = re.search(r"_np(\d+)_run(\d+)\.log$", path)
        npr, run = int(m.group(1)), int(m.group(2))
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        procs = [dict(rank=int(a), files=int(b), bytes=int(c), tokens=int(d), t=float(e))
                 for a, b, c, d, e in PROC_RE.findall(txt)]
        t_total = re.search(r"EXECUTION_TIME=([\d.]+)", txt)
        check = re.search(r"CHECK=(\S+)", txt)
        if not procs or not t_total:
            continue
        procs.sort(key=lambda x: x["rank"])
        tl = [x["t"] for x in procs]
        runs.setdefault(npr, []).append(dict(
            run=run, total=float(t_total.group(1)), procs=procs,
            li=max(tl) / fmean(tl), spread=(max(tl) - min(tl)) / fmean(tl),
            check=check.group(1) if check else "?"))
    for npr in runs:
        runs[npr].sort(key=lambda r: r["run"])
    return runs


def main():
    out = []
    w = out.append

    base = leer_baseline()
    if not base:
        print("No baseline logs found in", LOG_DIR)
        return
    t_seq = fmean(base)
    w("## Sequential baseline\n")
    w("| Run | Time (s) |\n|---|---|")
    for i, t in enumerate(base, 1):
        w(f"| {i} | {t:.4f} |")
    w(f"| **Mean (T_seq)** | **{t_seq:.4f}** (std {fstd(base):.4f}) |\n")

    resumen = {}
    for version in ("mpi1", "mpi2"):
        data = leer_mpi(version)
        for npr, runs in sorted(data.items()):
            w(f"## {version} — p = {npr}\n")
            w("### Total execution time and LI per run\n")
            w("| Run | Total time (s) | LI = max/mean | Spread (max-min)/mean | Check vs baseline |")
            w("|---|---|---|---|---|")
            for r in runs:
                w(f"| {r['run']} | {r['total']:.4f} | {r['li']:.4f} | {r['spread']:.4f} | {r['check']} |")
            totals = [r["total"] for r in runs]
            lis = [r["li"] for r in runs]
            tp = fmean(totals)
            s = t_seq / tp
            e = s / npr
            w(f"| **Mean** | **{tp:.4f}** (std {fstd(totals):.4f}) | **{fmean(lis):.4f}** | "
              f"{fmean([r['spread'] for r in runs]):.4f} | |\n")

            w("### Local processing time per process (s)\n")
            hdr = "| Rank | Files | MB | " + " | ".join(f"Run {r['run']}" for r in runs) + " | Mean |"
            w(hdr)
            w("|" + "---|" * (4 + len(runs)))
            for k in range(npr):
                p0 = runs[0]["procs"][k]
                ts = [r["procs"][k]["t"] for r in runs]
                w(f"| {k} | {p0['files']} | {p0['bytes']/1e6:.2f} | "
                  + " | ".join(f"{t:.4f}" for t in ts) + f" | {fmean(ts):.4f} |")
            w("")
            resumen[(version, npr)] = dict(tp=tp, s=s, e=e, li=fmean(lis),
                                           li_max=max(lis), std=fstd(totals))

    w("## Comparison\n")
    w(f"T_seq = {t_seq:.4f} s\n")
    w("| Version | p | T_p mean (s) | Speedup S_p | Efficiency E_p | Mean LI | Worst LI |")
    w("|---|---|---|---|---|---|---|")
    for (version, npr), d in sorted(resumen.items(), key=lambda x: (x[0][1], x[0][0])):
        w(f"| {version} | {npr} | {d['tp']:.4f} ± {d['std']:.4f} | {d['s']:.3f} | "
          f"{d['e']:.3f} | {d['li']:.4f} | {d['li_max']:.4f} |")

    texto = "\n".join(out)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
