# RAG Profile Matching System

A Python-based resume matching system that combines semantic retrieval, vector search, eligibility filtering, and weighted candidate scoring to rank resumes against job descriptions.

## 1. Project Overview

Traditional keyword-based resume matching can overlook semantically related skills and experience. This project explores a retrieval-based approach that represents resume passages and job descriptions using sentence embeddings.

The system retrieves relevant resume content, checks mandatory job requirements, and ranks eligible candidates using a configurable weighted score.

## 2. Key Features

- Read PDF, DOCX, and TXT resumes.
- Extract candidate names, skills, and experience.
- Apply section-aware resume chunking.
- Generate HuggingFace sentence embeddings.
- Store and search embeddings using ChromaDB.
- Retrieve and group resume chunks by candidate.
- Apply mandatory skill and experience checks.
- Calculate weighted candidate match scores.
- Generate explainable JSON matching results.
- Evaluate semantic-only versus hybrid matching.
- Generate JSON/CSV evaluation reports and notebook visualizations.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Resume processing | pypdf, python-docx |
| Embeddings | sentence-transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Data analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Testing | unittest |
| Experiments | Jupyter Notebook |

## 4. Architecture

**Indexing pipeline**

Resume files → Text extraction → Metadata extraction → Section-aware chunking → Embeddings → ChromaDB

**Matching pipeline**

Job description → Job requirement extraction → Semantic retrieval → Candidate grouping → Mandatory eligibility checks → Weighted scoring → Ranked candidate results

## 5. Dataset

The synthetic evaluation dataset contains:

- 35 fictional candidate resumes.
- 15 PDF, 10 DOCX, and 10 TXT documents.
- Seven technical role categories.
- Five job descriptions.
- 175 candidate-job relevance judgments.

The labels are generated using mandatory skill coverage and minimum experience requirements. They are not independent human hiring assessments.

## 6. Installation

### Prerequisites

- A compatible Python installation.
- Git, if cloning from a repository.
- Internet access for the initial HuggingFace model download.

### Setup on Windows

Clone or extract the project, then open PowerShell in the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Generate the dataset

```powershell
python -m dataset.generate_dataset
```

### Index the synthetic resumes

Ensure `resume_rag.py` points to `dataset/resumes` and that indexing and retrieval use the same ChromaDB collection.

```powershell
python resume_rag.py
```

### Run the matching demonstration

```powershell
python -m demo.run_demo
```

### Run performance evaluation

```powershell
python -m evaluation.evaluate
```

### Run automated tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Open the experiment notebook

```powershell
jupyter notebook notebooks/rag_experiments.ipynb
```

## 7. Candidate Scoring

The configured weighted score uses:

| Component | Weight |
|---|---:|
| Semantic similarity | 40% |
| Mandatory skills | 30% |
| Preferred skills | 20% |
| Experience | 10% |

Mandatory requirements can also act as eligibility filters.

## 8. Evaluation Framework

The evaluation compares:

1. **Semantic-only:** Candidates ranked by embedding-based retrieval.
2. **Hybrid:** Semantic retrieval combined with eligibility filtering and weighted scoring.

Metrics include:

- Precision@5 and Precision@10
- Recall@5 and Recall@10
- Hit Rate@5 and Hit Rate@10
- Mean Reciprocal Rank (MRR)
- Mean, median, and per-job P95 latency

### Measured Results

Replace the table below with the values from `results/summary_metrics.csv`.

| Metric | Semantic-only | Hybrid |
|---|---|---|
| Precision@5 | TBD | TBD |
| Recall@5 | TBD | TBD |
| Precision@10 | TBD | TBD |
| Recall@10 | TBD | TBD |
| MRR | TBD | TBD |
| Mean latency (ms) | TBD | TBD |

The evaluation reports are stored in `results/`.

## 9. Experimental Notebook

`notebooks/rag_experiments.ipynb` includes:

- Dataset exploration
- Metadata extraction analysis
- Chunk size and section distribution
- Embedding similarity experiments
- Semantic and hybrid retrieval comparison
- Accuracy and latency visualizations
- Failure analysis
- Final experimental conclusions

Individual charts are saved in `results/figures/`.

## 10. Testing

The project includes automated tests for dataset generation, metric calculations, and evaluation pipeline behavior, along with the existing project tests.

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Record the actual number of passing tests in the final submission.

## 11. Limitations

- The dataset is synthetic and relatively small.
- Relevance labels are based on deterministic eligibility rules.
- Metadata extraction may miss skill synonyms and unusual resume formats.
- Full-index retrieval is not designed for large-scale deployment.
- Embedding similarity alone does not establish candidate suitability.
- The current pipeline focuses on retrieval and ranking rather than LLM-generated answers.

## 12. Future Improvements

- Cross-encoder reranking.
- Skill normalization and synonym expansion.
- Larger and more diverse resume datasets.
- Human-annotated relevance judgments.
- Embedding model and chunking ablation studies.
- Scalable vector retrieval with bounded top-K search.
- Grounded LLM explanations with supporting resume excerpts.

## 13. Demo

The project demonstration is available in `demo/demo_video.mp4` if included in the submission.

## 14. Conclusion

This project demonstrates a practical semantic resume retrieval and ranking pipeline. It combines vector search with explicit job requirements and evaluates the resulting rankings using standard information retrieval metrics.

The experiment notebook and evaluation reports document the system's measured behavior, limitations, and opportunities for improvement.
