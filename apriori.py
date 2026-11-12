"""Поиск частых наборов объектов алгоритмом Apriori.

Запуск:
    python apriori.py baskets.csv 0.05 support
    python apriori.py baskets.csv 0.05 lex
"""
import argparse
import csv
from itertools import combinations


def load_baskets(path, encoding="cp1251"):
    """Каждая строка файла = одна корзина (заголовка нет)."""
    baskets = []
    with open(path, encoding=encoding, newline="") as f:
        for row in csv.reader(f):
            items = {c.strip() for c in row if c.strip()}
            if items:
                baskets.append(items)
    return baskets


def apriori(baskets, min_support):
    """Возвращает словарь {кортеж_товаров: поддержка (доля от 0 до 1)}."""
    if not 0 < min_support <= 1:
        raise ValueError("support must be in (0, 1]")
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


def association_rules(frequent, min_confidence=0.7, max_items=None):
    """Правила (антецедент, консеквент, поддержка, достоверность).

    frequent содержит ВСЕ частые подмножества. Пустые части запрещены.
    Ограничение max_items относится к суммарному числу объектов.
    """
    if not 0 < min_confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    if max_items is not None and max_items < 2:
        raise ValueError("max_items must be >= 2")
    rules = []
    for union, support in frequent.items():
        if max_items is not None and len(union) > max_items:
            continue
        for size in range(1, len(union)):
            for antecedent in combinations(union, size):
                consequent = tuple(x for x in union if x not in antecedent)
                confidence = support / frequent[antecedent]
                if confidence + 1e-12 >= min_confidence:
                    rules.append((antecedent, consequent, support, confidence))
    return rules


def sort_rules(rules, order):
    if order == 'support':
        return sorted(rules, key=lambda r: (-r[2], r[0], r[1]))
    if order == 'lex':
        return sorted(rules, key=lambda r: (r[0], r[1]))
    raise ValueError("order must be support or lex")


def format_rule(rule):
    a, c, support, confidence = rule
    return (f"{{{', '.join(a)}}} → {{{', '.join(c)}}}; "
            f"поддержка {support:.4%}; достоверность {confidence:.4%}")


def main():
    p = argparse.ArgumentParser(description="Apriori и ассоциативные правила")
    p.add_argument('dataset')
    p.add_argument('support', type=float)
    p.add_argument('order', choices=['support', 'lex'])
    p.add_argument('--confidence', type=float, default=0.7)
    p.add_argument('--max-items', type=int, default=None)
    p.add_argument('--encoding', default='cp1251')
    p.add_argument('--sets-only', action='store_true')
    args = p.parse_args()
    try:
        baskets = load_baskets(args.dataset, args.encoding)
        frequent = apriori(baskets, args.support)
        rules = association_rules(frequent, args.confidence, args.max_items)
    except (ValueError, OSError) as e:
        p.error(str(e))
    print(f"Корзин: {len(baskets)}; частых наборов: {len(frequent)}")
    for items, sup in sort_result(frequent, args.order):
        print(f"{{{', '.join(items)}}}; поддержка {sup:.4%}")
    if not args.sets_only:
        print(f"Правил: {len(rules)}")
        for rule in sort_rules(rules, args.order):
            print(format_rule(rule))


if __name__ == '__main__':
    main()
