"""Эксперименты: время работы и число наборов разной длины при разных порогах."""
import os
import time
from collections import Counter

import matplotlib.pyplot as plt

from apriori import apriori, load_baskets

SUPPORTS = [0.01, 0.03, 0.05, 0.10, 0.15]
OUT = "results"


def main():
    os.makedirs(OUT, exist_ok=True)
    baskets = load_baskets("baskets.csv")
    print("Корзин:", len(baskets))

    times, counts = [], []
    for s in SUPPORTS:
        t = time.perf_counter()
        res = apriori(baskets, s)
        times.append(time.perf_counter() - t)
        counts.append(Counter(len(k) for k in res))
        print(f"{s:.0%}: {times[-1]:.3f} c, наборов {len(res)}, по длинам {dict(sorted(counts[-1].items()))}")

    labels = [f"{int(s * 100)}%" for s in SUPPORTS]

    # 1. время
    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color="steelblue")
    for i, v in enumerate(times):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xlabel("Порог поддержки")
    plt.ylabel("Время, с")
    plt.title("Время работы Apriori при разных порогах поддержки")
    plt.tight_layout()
    plt.savefig(f"{OUT}/time.png", dpi=200)
    plt.close()

    # 2. число наборов по длинам (сгруппированные столбцы)
    lengths = sorted({k for c in counts for k in c})
    width = 0.8 / len(lengths)
    plt.figure(figsize=(9, 5))
    for j, L in enumerate(lengths):
        vals = [c.get(L, 0) for c in counts]
        xs = [i + j * width for i in range(len(SUPPORTS))]
        plt.bar(xs, vals, width, label=f"длина {L}")
    plt.xticks([i + 0.4 - width / 2 for i in range(len(SUPPORTS))], labels)
    plt.yscale("symlog")
    plt.xlabel("Порог поддержки")
    plt.ylabel("Число частых наборов (лог. шкала)")
    plt.title("Частые наборы разной длины")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/lengths.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
