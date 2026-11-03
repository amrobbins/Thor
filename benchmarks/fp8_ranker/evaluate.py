"""Shortlist regret curves. Standard library only.

An ordering is a list of candidate indices of one cell. regret(N) = best measured time among the first N candidates /
the cell's oracle (the autotuner times the first N and keeps the fastest). Orderings:
  model          sort by score, ties by sweep list position;
  model_leaders  production's generalShortlist rule: each family's leader (lowest local rank, first in list order)
                 first, in list order, then the rest by score;
  frozen         VALUE: the frozen ranker's list order (list_position). ARG: the frozen model's margin on the ARG
                 features, with leaders (the counterfactual "reuse today's model for ARG");
  list           ARG: the generator order (VALUE twins in VALUE order, then the ARG-only passes).
"""

import math


def order_by_scores(cell, scores, leaders=False):
    indices = list(range(len(cell.candidates)))
    by_score = sorted(indices, key=lambda i: (-scores[i], cell.candidates[i].list_position))
    if not leaders:
        return by_score
    lead = {}
    for i in sorted(indices, key=lambda i: (cell.candidates[i].local_rank, cell.candidates[i].list_position)):
        lead.setdefault(cell.candidates[i].family, i)
    first = sorted(lead.values(), key=lambda i: cell.candidates[i].list_position)
    chosen = set(first)
    return first + [i for i in by_score if i not in chosen]


def list_order(cell):
    return sorted(range(len(cell.candidates)), key=lambda i: cell.candidates[i].list_position)


def regret_curve(cell, order, max_n):
    oracle = cell.oracle_ms
    curve = []
    best = math.inf
    for n in range(1, max_n + 1):
        if n <= len(order):
            best = min(best, cell.candidates[order[n - 1]].median_ms)
        curve.append(best / oracle)
    return curve


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))] if values else float("nan")


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def summarize(curves, n):
    """Stats of regret at depth n (1-based) over a list of curves."""
    values = [curve[n - 1] for curve in curves]
    return {"n": n, "cells": len(values), "geomean": geomean(values), "p95": quantile(values, 0.95),
            "p99": quantile(values, 0.99), "max": max(values) if values else float("nan"),
            "exact": sum(v <= 1.0 for v in values) / len(values) if values else float("nan")}


def smallest_depth(curves, max_n, predicate):
    for n in range(1, max_n + 1):
        if predicate(summarize(curves, n)):
            return n
    return None
