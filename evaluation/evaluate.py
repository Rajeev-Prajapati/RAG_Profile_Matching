
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from job_matcher import JobMatcher
from retrieval.semantic_retriever import SemanticRetriever

from evaluation.metrics import (
    evaluate_ranking,
    latency_summary
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def candidate_id_from_path(resume_path):
    """
    Example:
    C003_python_backend_developer.pdf -> C003

    This works for the Step 11 synthetic dataset.
    """
    filename = str(resume_path).replace("\\", "/").split("/")[-1]
    candidate_id = filename.split("_", 1)[0]

    if not (
        len(candidate_id) == 4
        and candidate_id.startswith("C")
        and candidate_id[1:].isdigit()
    ):
        raise ValueError(
            f"Invalid synthetic resume filename: {filename}"
        )

    return candidate_id


def ranking_to_ids(candidates):
    ids = [
        candidate_id_from_path(candidate["resume_path"])
        for candidate in candidates
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate candidates found in ranking"
        )

    return ids


def get_relevant_ids(labels, job_id):
    judgments = labels[job_id]["judgments"]

    return {
        judgment["candidate_id"]
        for judgment in judgments
        if judgment["relevant"]
    }


def validate_dataset(jobs, candidates, labels):
    if not jobs:
        raise ValueError("No job descriptions found")

    if not candidates:
        raise ValueError("No candidates found")

    candidate_ids = {
        candidate["candidate_id"]
        for candidate in candidates
    }

    if len(candidate_ids) != len(candidates):
        raise ValueError("Duplicate candidate IDs")

    job_ids = [job["job_id"] for job in jobs]

    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Duplicate job IDs")

    for job in jobs:
        job_id = job["job_id"]

        if job_id not in labels:
            raise ValueError(
                f"Missing relevance labels for {job_id}"
            )

        judgments = labels[job_id]["judgments"]

        labeled_ids = [
            item["candidate_id"]
            for item in judgments
        ]

        if len(labeled_ids) != len(set(labeled_ids)):
            raise ValueError(
                f"Duplicate judgments for {job_id}"
            )

        if set(labeled_ids) != candidate_ids:
            raise ValueError(
                f"Incomplete ground truth for {job_id}"
            )


def benchmark(call, repetitions=30, warmups=2):
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    if warmups < 0:
        raise ValueError("warmups cannot be negative")

    for _ in range(warmups):
        call()

    latencies = []
    final_result = None

    for _ in range(repetitions):
        start = time.perf_counter()

        final_result = call()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

    return final_result, latency_summary(latencies)


def make_metric_row(
    job_id,
    method,
    ranking,
    relevant_ids,
    latency,
    eligible_count=None
):
    scores = evaluate_ranking(
        ranking,
        relevant_ids,
        ks=(5, 10)
    )

    return {
        "job_id": job_id,
        "method": method,
        "relevant_count": len(relevant_ids),
        "returned_count": len(ranking),
        "eligible_count": eligible_count,
        **scores,
        **latency
    }


def macro_average(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["method"]].append(row)

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "hit_rate_at_5",
        "precision_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "reciprocal_rank",
        "mean_ms",
        "median_ms",
        "p95_ms"
    ]

    summary = []

    for method, method_rows in grouped.items():
        result = {
            "method": method,
            "jobs_evaluated": len(method_rows)
        }

        for metric in metric_names:
            values = [
                row[metric]
                for row in method_rows
                if row[metric] is not None
            ]

            result[metric] = (
                round(mean(values), 4)
                if values else None
            )

        # MRR is the mean of per-job reciprocal ranks.
        result["mrr"] = result.pop(
            "reciprocal_rank"
        )

        summary.append(result)

    return summary


def save_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    dataset_dir=DATASET_DIR,
    results_dir=RESULTS_DIR,
    repetitions=30,
    warmups=2,
    semantic_retriever=None,
    job_matcher=None
):
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)

    jobs = load_json(
        dataset_dir / "job_descriptions.json"
    )

    candidates = load_json(
        dataset_dir / "candidates.json"
    )

    labels = load_json(
        dataset_dir / "relevance_labels.json"
    )

    validate_dataset(jobs, candidates, labels)

    retriever = (
        semantic_retriever
        if semantic_retriever is not None
        else SemanticRetriever()
    )

    matcher = (
        job_matcher
        if job_matcher is not None
        else JobMatcher()
    )

    # Confirm the expected dataset was indexed.
    expected_ids = {
        candidate["candidate_id"]
        for candidate in candidates
    }

    # This intentionally retrieves all candidates for the
    # small synthetic dataset. It is not a scalable strategy.
    rows = []
    rankings = []

    for job in jobs:
        job_id = job["job_id"]
        description = job["job_description"]

        relevant_ids = get_relevant_ids(
            labels, job_id
        )

        print(f"\nEvaluating {job_id}: {job['title']}")

        def semantic_call():
            return retriever.search_candidates(
                job_description=description,
                retrieve_all=True
            )

        def hybrid_call():
            return matcher.match_candidates(
                job_description=description,
                must_have_skills=job["must_have_skills"],
                preferred_skills=job["preferred_skills"],
                min_experience=job["min_experience"],
                top_n=len(candidates)
            )

        semantic_results, semantic_latency = benchmark(
            semantic_call,
            repetitions=repetitions,
            warmups=warmups
        )

        invalid_paths = []

        for candidate in semantic_results:
            path = candidate["resume_path"]

            try:
                candidate_id_from_path(path)
            except ValueError:
                invalid_paths.append(path)

        if invalid_paths:
            raise ValueError(
                "The ChromaDB collection contains resumes "
                "that do not belong to the Step 11 dataset:\n"
                + "\n".join(invalid_paths)
                + "\nReindex the synthetic resumes into a "
                "dedicated ChromaDB collection."
            )

        hybrid_results, hybrid_latency = benchmark(
            hybrid_call,
            repetitions=repetitions,
            warmups=warmups
        )

        semantic_ids = ranking_to_ids(
            semantic_results
        )

        hybrid_ids = ranking_to_ids(
            hybrid_results["top_matches"]
        )

        # A semantic-only ranking must contain all
        # 35 indexed resumes for a fair baseline.
        if set(semantic_ids) != expected_ids:
            missing = sorted(
                expected_ids - set(semantic_ids)
            )
            unexpected = sorted(
                set(semantic_ids) - expected_ids
            )

            raise ValueError(
                f"Indexed dataset mismatch. "
                f"Missing: {missing}; "
                f"Unexpected: {unexpected}"
            )

        if not set(hybrid_ids).issubset(expected_ids):
            raise ValueError(
                "Hybrid output contains unknown candidates"
            )

        if not set(hybrid_ids).issubset(set(semantic_ids)):
            raise ValueError(
                "Hybrid results are not in the retrieval pool"
            )

        rows.append(
            make_metric_row(
                job_id,
                "semantic_only",
                semantic_ids,
                relevant_ids,
                semantic_latency
            )
        )

        rows.append(
            make_metric_row(
                job_id,
                "hybrid",
                hybrid_ids,
                relevant_ids,
                hybrid_latency,
                eligible_count=hybrid_results["eligible_count"]
            )
        )

        rankings.extend([
            {
                "job_id": job_id,
                "method": "semantic_only",
                "candidate_ids": semantic_ids
            },
            {
                "job_id": job_id,
                "method": "hybrid",
                "candidate_ids": hybrid_ids
            }
        ])

        print(
            "Semantic Precision@5:",
            round(rows[-2]["precision_at_5"], 4)
        )

        print(
            "Hybrid Precision@5:",
            round(rows[-1]["precision_at_5"], 4)
        )

    summary = macro_average(rows)

    report = {
        "experiment": "RAG Profile Matching Evaluation",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "resume_count": len(candidates),
            "job_count": len(jobs),
            "ground_truth_judgments": sum(
                len(labels[job["job_id"]]["judgments"])
                for job in jobs
            )
        },
        "configuration": {
            "repetitions": repetitions,
            "warmups": warmups,
            "cutoffs": [5, 10],
            "semantic_retrieve_all": True,
            "hybrid_top_n": len(candidates)
        },
        "per_job": rows,
        "summary": summary,
        "rankings": rankings
    }

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with (
        results_dir / "performance_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    save_csv(
        results_dir / "per_job_metrics.csv",
        rows
    )

    save_csv(
        results_dir / "summary_metrics.csv",
        summary
    )

    print("\nEvaluation complete.")
    print("Reports saved to:", results_dir)

    return report


if __name__ == "__main__":
    run_evaluation()
