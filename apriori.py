"""Поиск частых наборов объектов алгоритмом Apriori.

Запуск:
    python apriori.py baskets.csv 0.05 support
    python apriori.py baskets.csv 0.05 lex
"""
import argparse
import csv
from itertools import combinations


def load_baskets(path):
    print(path)
    """Каждая строка файла = одна корзина (заголовка нет)."""
    baskets = []
    with open(path, encoding="cp1251", newline="") as f:
        for row in csv.reader(f):
            items = {c.strip() for c in row if c.strip()}
            if items:
                baskets.append(items)
    return baskets


def apriori(baskets, min_support):
    """Возвращает словарь {кортеж_товаров: поддержка (доля от 0 до 1)}."""
    n = len(baskets)
    if n == 0:
        return {}
    min_count = min_support * n

    # битовые маски товаров
    masks = {}
    for i, basket in enumerate(baskets):
        for item in basket:
            masks[item] = masks.get(item, 0) | (1 << i)

    # частые наборы длины 1; храним {набор: маска}
    level = {}
    for item, m in masks.items():
        if m.bit_count() >= min_count:
            level[(item,)] = m

    result = {s: m.bit_count() / n for s, m in level.items()}

    while level:
        prev = sorted(level)
        prev_set = set(prev)
        new_level = {}
        for a, b in combinations(prev, 2):
            # объединяем наборы, у которых совпадают все элементы кроме последнего
            if a[:-1] != b[:-1]:
                continue
            cand = a + (b[-1],)
            # свойство Apriori: все подмножества длины k-1 должны быть частыми
            if any(sub not in prev_set for sub in combinations(cand, len(cand) - 1)):
                continue
            m = level[a] & level[b]
            if m.bit_count() >= min_count:
                new_level[cand] = m
        for s, m in new_level.items():
            result[s] = m.bit_count() / n
        level = new_level
    return result


def sort_result(result, order):
    if order == "support":
        return sorted(result.items(), key=lambda kv: (-kv[1], kv[0]))
    if order == "lex":
        return sorted(result.items(), key=lambda kv: kv[0])
    raise ValueError("order должен быть 'support' или 'lex'")


def main():
    p = argparse.ArgumentParser(description="Apriori")
    p.add_argument("dataset")
    p.add_argument("support", type=float, help="порог поддержки, например 0.05")
    p.add_argument("order", choices=["support", "lex"], help="порядок вывода")
    args = p.parse_args()

    baskets = load_baskets(args.dataset)
    result = apriori(baskets, args.support)
    print(f"Корзин: {len(baskets)}, порог: {args.support:.2%}, найдено наборов: {len(result)}")
    for items, sup in sort_result(result, args.order):
        print(f"{{{', '.join(items)}}}  {sup:.4f} ({sup:.2%})")


if __name__ == "__main__":
    main()