"""
Experiment 1: heslá a hashovanie.

1. Načíta 15 vymyslených hesiel z data/passwords.csv.
2. zxcvbn: skóre 0-4 a odhad počtu pokusov na uhádnutie.
3. Benchmark: hashe/s pre SHA-256, bcrypt (cost 12), Argon2id
   (64 MiB, 3 iterácie, paralelizmus 1); 5 zahrievacích + 30 meraní.
4. Odhad času prelomenia = počet pokusov / hashe za sekundu.

Výstupy (results/):
  table1_passwords.csv, table2_hash_speed.csv, crack_times.csv,
  hash_rates_raw.csv, experiment_meta.json,
  graph1_hash_speed.png, graph2_crack_time.png

Spustenie: python passwords.py
"""
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import bcrypt
import matplotlib
matplotlib.use("Agg")  # bez okna, iba ukladanie PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argon2.low_level import Type, hash_secret_raw
from zxcvbn import zxcvbn

# ----------------------------- pevné parametre -----------------------------
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "passwords.csv"
RESULTS = BASE / "results"

WARMUP_RUNS = 5          # zahrievacie behy (nezapočítavajú sa)
MEASURE_RUNS = 30        # merané behy
BCRYPT_COST = 12
ARGON2_MEMORY_KIB = 64 * 1024   # 64 MiB
ARGON2_TIME_COST = 3            # iterácie
ARGON2_PARALLELISM = 1
BENCH_PASSWORD = b"benchmark-password-Experiment1"
BENCH_SALT = b"0123456789abcdef"  # 16 B, pevná soľ (iba pre benchmark)

# počet hashov v jednom meraní (rýchle algoritmy treba merať v dávke)
HASHES_PER_RUN = {"SHA-256": 200_000, "bcrypt (cost 12)": 2, "Argon2id": 3}

ALGOS = ["SHA-256", "bcrypt (cost 12)", "Argon2id"]
CATEGORY_LABELS = {
    "1_velmi_slabe": "veľmi slabé",
    "2_slabe": "slabé",
    "3_stredne": "stredné",
    "4_silne": "silné",
    "5_velmi_silne": "veľmi silné",
}
# referenčné čiary pre graf 2 (sekundy)
TIME_REFS = [(3600, "1 hodina"), (86400 * 365, "1 rok"), (86400 * 365 * 100, "100 rokov")]


# ------------------------------- hashovanie --------------------------------
def hash_sha256():
    hashlib.sha256(BENCH_PASSWORD).digest()


def hash_bcrypt():
    bcrypt.hashpw(BENCH_PASSWORD, bcrypt.gensalt(rounds=BCRYPT_COST))


def hash_argon2id():
    hash_secret_raw(
        secret=BENCH_PASSWORD, salt=BENCH_SALT,
        time_cost=ARGON2_TIME_COST, memory_cost=ARGON2_MEMORY_KIB,
        parallelism=ARGON2_PARALLELISM, hash_len=32, type=Type.ID,
    )


HASH_FUNCS = {"SHA-256": hash_sha256, "bcrypt (cost 12)": hash_bcrypt, "Argon2id": hash_argon2id}


def measure_rate(func, n):
    """Vráti hashe za sekundu pre dávku n hashov."""
    start = time.perf_counter()
    for _ in range(n):
        func()
    return n / (time.perf_counter() - start)


def run_benchmark():
    rows = []
    for name in ALGOS:
        func, n = HASH_FUNCS[name], HASHES_PER_RUN[name]
        print(f"Benchmark {name}: {WARMUP_RUNS} zahrievacích + {MEASURE_RUNS} meraní ...")
        for _ in range(WARMUP_RUNS):
            measure_rate(func, n)
        for i in range(1, MEASURE_RUNS + 1):
            rows.append({"algorithm": name, "run": i, "hashes_per_second": measure_rate(func, n)})
    return pd.DataFrame(rows)


# ------------------------------- zxcvbn ------------------------------------
def score_passwords():
    df = pd.read_csv(DATA_FILE, dtype=str)
    scores, guesses = [], []
    for pw in df["password"]:
        res = zxcvbn(pw)
        scores.append(res["score"])
        guesses.append(float(res["guesses"]))
    df["zxcvbn_score"] = scores
    df["guesses"] = guesses
    df["guesses_log10"] = np.log10(df["guesses"])
    df["category_label"] = df["category"].map(CATEGORY_LABELS)
    return df


# ------------------------------- zariadenie --------------------------------
def device_info():
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except Exception:
        cpu = platform.processor() or platform.machine()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "warmup_runs": WARMUP_RUNS,
        "measure_runs": MEASURE_RUNS,
        "hashes_per_run": HASHES_PER_RUN,
        "bcrypt_cost": BCRYPT_COST,
        "argon2": {"memory_kib": ARGON2_MEMORY_KIB, "time_cost": ARGON2_TIME_COST,
                   "parallelism": ARGON2_PARALLELISM},
        "note": "Meranie na jednom CPU, jedno vlákno. GPU/ASIC útočník je rádovo rýchlejší.",
    }


# -------------------------------- grafy ------------------------------------
def plot_hash_speed(table2, meta):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ALGOS))
    means = [table2.loc[a, "mean"] for a in ALGOS]
    ax.bar(x, means, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ALGOS)
    ax.set_ylabel("Hashe za sekundu (log. os)")
    ax.set_title("Rýchlosť hashovania na jednom CPU")
    for xi, m in zip(x, means):
        ax.text(xi, m * 1.2, f"{m:,.1f}".replace(",", " "), ha="center", fontsize=9)
    ax.set_ylim(top=max(means) * 10)
    fig.text(0.01, 0.01, f"{meta['measure_runs']} meraní + {meta['warmup_runs']} zahrievacích behov, 1 vlákno",
             fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "graph1_hash_speed.png", dpi=200)
    plt.close(fig)


def plot_crack_time(crack_cat, meta):
    cats = list(CATEGORY_LABELS)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.26
    x = np.arange(len(cats))
    colors = {"SHA-256": "#d62728", "bcrypt (cost 12)": "#ff7f0e", "Argon2id": "#2ca02c"}
    for i, a in enumerate(ALGOS):
        vals = [crack_cat.loc[(c, a), "crack_time_s"] for c in cats]
        ax.bar(x + (i - 1) * width, vals, width, label=a, color=colors[a])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats])
    ax.set_xlabel("Kategória hesla")
    ax.set_ylabel("Odhadovaný čas prelomenia [s] (log. os)")
    ax.set_title("Čas prelomenia podľa kategórie hesla a algoritmu\n(geometrický priemer z 3 hesiel v kategórii)")
    for sec, label in TIME_REFS:
        ax.axhline(sec, color="gray", linestyle="--", linewidth=0.8)
        ax.text(-0.48, sec * 1.4, label, fontsize=8, color="dimgray", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.text(0.01, 0.01, "1 vlákno, čas = pokusy zxcvbn / hashe za s", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "graph2_crack_time.png", dpi=200)
    plt.close(fig)


# --------------------------------- main ------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    meta = device_info()

    # 1-2) heslá + zxcvbn  -> Tabuľka 1
    pw = score_passwords()
    table1 = pw[["category_label", "password", "zxcvbn_score", "guesses", "guesses_log10"]]
    table1.to_csv(RESULTS / "table1_passwords.csv", index=False, encoding="utf-8")

    # 3) benchmark -> Tabuľka 2
    raw = run_benchmark()
    raw.to_csv(RESULTS / "hash_rates_raw.csv", index=False)
    g = raw.groupby("algorithm")["hashes_per_second"]
    table2 = pd.DataFrame({
        "mean": g.mean(), "median": g.median(), "max": g.max(),
        "min": g.min(), "std": g.std(), "n_runs": g.count(),
    }).loc[ALGOS]
    table2.round(2).to_csv(RESULTS / "table2_hash_speed.csv", encoding="utf-8")

    # 4) čas prelomenia = pokusy / (hashe/s)  (použijeme priemernú rýchlosť)
    rows = []
    for _, r in pw.iterrows():
        for a in ALGOS:
            rate = table2.loc[a, "mean"]
            rows.append({
                "category": r["category"], "category_label": r["category_label"],
                "password": r["password"], "zxcvbn_score": r["zxcvbn_score"],
                "guesses": r["guesses"], "algorithm": a,
                "hashes_per_second": rate, "crack_time_s": r["guesses"] / rate,
            })
    crack = pd.DataFrame(rows)
    crack.to_csv(RESULTS / "crack_times.csv", index=False, encoding="utf-8")

    # geometrický priemer za kategóriu (3 heslá); aritmetický priemer by
    # bol na logaritmickej škále zavádzajúci (dominuje mu najsilnejšie heslo)
    crack["log10_time"] = np.log10(crack["crack_time_s"])
    crack_cat = crack.groupby(["category", "algorithm"]).agg(log10_time=("log10_time", "mean"))
    crack_cat["crack_time_s"] = 10 ** crack_cat["log10_time"]
    crack = crack.drop(columns="log10_time")
    crack_cat.to_csv(RESULTS / "crack_times_by_category.csv", encoding="utf-8")

    # grafy
    plot_hash_speed(table2, meta)
    plot_crack_time(crack_cat, meta)

    with open(RESULTS / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nTabuľka 2 (hashe/s):")
    print(table2.round(2).to_string())
    print(f"\nHotovo. Výstupy sú v: {RESULTS}")


if __name__ == "__main__":
    main()
