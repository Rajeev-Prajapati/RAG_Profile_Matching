
from statistics import mean, median
from math import ceil, isfinite


def validate_k(k):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")


def validate_ranking(retrieved):
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("Ranking contains duplicate candidates")


def precision_at_k(retrieved, relevant, k):
    validate_k(k)
    validate_ranking(retrieved)

    relevant = set(relevant)
    hits = len(set(retrieved[:k]) & relevant)

    return hits / k


def recall_at_k(retrieved, relevant, k):
    validate_k(k)
    validate_ranking(retrieved)

    relevant = set(relevant)

    if not relevant:
        return None

    hits = len(set(retrieved[:k]) & relevant)

    return hits / len(relevant)


def hit_rate_at_k(retrieved, relevant, k):
    validate_k(k)
    validate_ranking(retrieved)

    if not relevant:
        return None

    return int(
        bool(set(retrieved[:k]) & set(relevant))
    )


def reciprocal_rank(retrieved, relevant):
    validate_ranking(retrieved)

    relevant = set(relevant)

    if not relevant:
        return None

    for rank, candidate_id in enumerate(retrieved, start=1):
        if candidate_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rankings, relevant_sets):
    if len(rankings) != len(relevant_sets):
        raise ValueError("Rankings and labels must align")

    scores = [
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_sets)
    ]

    valid = [score for score in scores if score is not None]

    return mean(valid) if valid else None


def percentile(values, percent):
    if not values:
        return None

    if not 0 <= percent <= 100:
        raise ValueError("Percent must be between 0 and 100")

    values = sorted(values)

    # Nearest-rank percentile.
    index = max(0, ceil(percent / 100 * len(values)) - 1)

    return values[index]


def latency_summary(latencies_ms):
    if not latencies_ms:
        return {
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None
        }

    if any(
        not isfinite(value) or value < 0
        for value in latencies_ms
    ):
        raise ValueError("Invalid latency value")

    return {
        "mean_ms": round(mean(latencies_ms), 3),
        "median_ms": round(median(latencies_ms), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3)
    }


def evaluate_ranking(retrieved, relevant, ks=(5, 10)):
    validate_ranking(retrieved)

    metrics = {}

    for k in ks:
        validate_k(k)

        metrics[f"precision_at_{k}"] = precision_at_k(
            retrieved, relevant, k
        )

        metrics[f"recall_at_{k}"] = recall_at_k(
            retrieved, relevant, k
        )

        metrics[f"hit_rate_at_{k}"] = hit_rate_at_k(
            retrieved, relevant, k
        )

    metrics["reciprocal_rank"] = reciprocal_rank(
        retrieved, relevant
    )

    return metrics
