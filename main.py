import csv
import hashlib
import json
import math
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def load_golden_set(file_path: str) -> list[dict]:
    golden_set = []

    # "r" = read mode
    # with creates a context manager. Python automatically closes the file once we're done with it
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # transform the JSON text into a python object (dictionary)
            example = json.loads(line)

            golden_set.append(example)

    return golden_set

def recall_at_k(
        relevant_passage_ids: list[str],
        retrieved_passage_ids: list[str],
        k: int,
) -> float:
    relevant = set(relevant_passage_ids)

    if not relevant:
        return 0.0

    # the ':k' makes python ignore everything after the k position.
    retrieved_at_k = set(retrieved_passage_ids[:k])

    found = relevant.intersection(retrieved_at_k)

    return len(found) / len(relevant)

# RR -> evaluates ONE query
def reciprocal_rank(
        relevant_passage_ids: list[str],
        retrieved_passage_ids: list[str],
) -> float:
    relevant = set(relevant_passage_ids)

    # enumerate give us the position and the value. That's why we don't use directly retrieved_passage_ids
    for rank, passage_id in enumerate(retrieved_passage_ids, start = 1):
        if passage_id in relevant:
            return 1 / rank
    return 0.0

# MRR -> evaluates a collection of queries by taking the average of their Reciprocal ranks

# Our score is higher when the relevant chunk is near (or) on top
def dcg_at_k(
        relevant_scores: list[int],
        k: int,
) -> float:
    score = 0.0

    for rank, relevance in enumerate(relevant_scores[:k], start = 1):
        score += relevance / math.log2(rank + 1)

    return score

# we sort the relevant scores, because we want all relevant scores near to top, that's the ideal

# Measures how good the retrieved ranking is compared with the ideal ranking
def ndcg_at_k(
        relevant_scores: list[int],
        total_relevant: int,
        k: int,
) -> float:
    actual_dcg = dcg_at_k(
        relevant_scores,
        k,
    )

    ideal_relevant_count = min(
        total_relevant,
        k,
    )

    ideal_scores = [1] * ideal_relevant_count

    ideal_dcg = dcg_at_k(
        ideal_scores,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def build_relevance_scores(
        relevant_passage_ids: list[str],
        retrieved_passage_ids: list[str],
) -> list[int]:
    relevant = set(relevant_passage_ids)
    scores = []

    for passage_id in retrieved_passage_ids:
        if passage_id in relevant:
            scores.append(1)
        else:
            scores.append(0)

    return scores

def evaluate_query(
        relevant_passage_ids: list[str],
        retrieved_passage_ids: list[str],
        k: int
) -> dict:
    relevance_scores = build_relevance_scores(relevant_passage_ids, retrieved_passage_ids)

    return {
        "recall_at_k": recall_at_k(
            relevant_passage_ids,
            retrieved_passage_ids,
            k,
        ),
        "reciprocal_rank": reciprocal_rank(
            relevant_passage_ids,
            retrieved_passage_ids,
        ),
        "ndcg_at_k": ndcg_at_k(
            relevance_scores,
            len(relevant_passage_ids),
            k,
        )
    }

def evaluate_dataset(
        dataset: list[dict],
        k: int
) -> list[dict]:
    results = []

    for example in dataset:
        metrics = evaluate_query(
            example["relevant_passage_ids"],
            example["retrieved_passage_ids"],
            k
        )

        metrics["category"] = example["category"]

        results.append(metrics)

    return results

def aggregate_metrics(
        results: list[dict],
) -> dict:
    if not results:
        return {
            "mean_recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_ndcg_at_k": 0.0,
        }

    mean_recall = sum(
        # like for each result get result["recall_at_k"]
        result["recall_at_k"]
        for result in results
    ) / len(results)

    mrr = sum(
        result["reciprocal_rank"]
        for result in results
    ) / len(results)

    mean_ndcg = sum(
        result["ndcg_at_k"]
        for result in results
    ) / len(results)

    return {
        "mean_recall_at_k": mean_recall,
        "mrr": mrr,
        "mean_ndcg_at_k": mean_ndcg,
    }

def passes_recall_gate(
        baseline_recall: float,
        candidate_recall: float,
        max_drop: float = 0.01,
) -> bool:
    drop = baseline_recall - candidate_recall

    return drop < max_drop or math.isclose(
        drop,
        max_drop,
        abs_tol=1e-9,
    )

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text,
    )

    return response.data[0].embedding

def embed_chunks(
        chunks: list[dict],
) -> list[dict]:
    embedded_chunks = []

    for chunk in chunks:
        embedding = create_embedding(chunk["text"])

        embedded_chunks.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "embedding": embedding,
        })

    return embedded_chunks

def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b
        # zip: pairs values in the same positon
        for a,b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (magnitude_a * magnitude_b)

def retrieve_dense(
        query: str,
        embedded_chunks: list[dict],
        k: int,
) -> list[str]:
    query_embeddings = create_embedding_cached(query)

    scored_chunks = []

    for chunk in embedded_chunks:
        score = cosine_similarity(
            query_embeddings,
            chunk["embedding"],
        )

        scored_chunks.append(
            (score, chunk["id"])
        )
    scored_chunks.sort(reverse=True)

    return [
        chunk_id
        for score, chunk_id in scored_chunks[:k]
    ]

def retrieve_for_example(
        example: dict,
        retriever,
        chunks: list[dict],
        k: int,
) -> dict:
    retrieved_ids = retriever(
        example["query"],
        chunks,
        k,
    )

    return {
        "query": example["query"],
        "category": example["category"],
        "relevant_passage_ids": example["relevant_passage_ids"],
        "retrieved_passage_ids": retrieved_ids,
    }

def retrieve_dataset(
        golden_set: list[dict],
        retriever,
        chunks: list[dict],
        k: int,
) -> list[dict]:
    results = []

    for example in golden_set:
        retrieved_example = retrieve_for_example(
            example,
            retriever,
            chunks,
            k,
        )

        results.append(retrieved_example)

    return results

def load_chunks(file_path: str) -> list[dict]:
    chunks = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            chunks.append(chunk)

    return chunks

def run_experiment(
        golden_set: list[dict],
        retriever,
        chunks: list[dict],
        k: int,
) -> dict:
    retrieved_dataset = retrieve_dataset(
        golden_set,
        retriever,
        chunks,
        k,
    )

    results = evaluate_dataset(
        retrieved_dataset,
        k,
    )

    overall_metrics = aggregate_metrics(results)

    category_metrics = aggregate_metrics_by_category(results)

    details = []

    for retrieved_example, metrics in zip(
            retrieved_dataset,
            results,
    ):
        details.append({
            "query": retrieved_example["query"],
            "category": retrieved_example["category"],
            "relevant_passage_ids": (
                retrieved_example["relevant_passage_ids"]
            ),
            "retrieved_passage_ids": (
                retrieved_example["retrieved_passage_ids"]
            ),
            "recall_at_k": metrics["recall_at_k"],
            "reciprocal_rank": metrics["reciprocal_rank"],
            "ndcg_at_k": metrics["ndcg_at_k"],
        })

    return {
        "overall": overall_metrics,
        "by_category": category_metrics,
        "details": details,
    }

def retrieve_hybrid(
        query: str,
        embedded_chunks: list[dict],
        k: int,
        alpha: float = 0.5,
):
    query_words = set(tokenize(query))
    query_embeddings = create_embedding_cached(query)

    scored_chunks = []

    for chunk in embedded_chunks:
        chunk_words = set(tokenize(chunk["text"]))

        common_words = query_words.intersection(chunk_words)

        if query_words:
            keyword_score = len(common_words) / len(query_words)
        else:
            keyword_score = 0.0

        dense_score = cosine_similarity(
            query_embeddings,
            chunk["embedding"],
        )

        dense_score = (dense_score + 1) / 2

        hybrid_score = (
            alpha * dense_score
            + (1 - alpha) * keyword_score
        )

        scored_chunks.append(
            (hybrid_score, chunk["id"])
        )

    scored_chunks.sort(reverse=True)

    return [
        chunk_id
        for score, chunk_id in scored_chunks[:k]
    ]

def aggregate_metrics_by_category(
    results: list[dict],
) -> dict:
    grouped_results = {}

    for result in results:
        category = result["category"]

        if category not in grouped_results:
            grouped_results[category] = []

        grouped_results[category].append(result)

    category_metrics = {}

    for category, category_results in grouped_results.items():
        category_metrics[category] = aggregate_metrics(
            category_results
        )

    return category_metrics

def normalize_reranked_ids(
    ranked_ids: list[str],
    candidate_ids: list[str],
) -> list[str]:
    valid_candidates = set(candidate_ids)

    normalized = []
    seen = set()

    for passage_id in ranked_ids:
        if (
            passage_id in valid_candidates
            and passage_id not in seen
        ):
            normalized.append(passage_id)
            seen.add(passage_id)

    for passage_id in candidate_ids:
        if passage_id not in seen:
            normalized.append(passage_id)
            seen.add(passage_id)

    return normalized

RERANK_MODEL = "gpt-4.1-mini"
RERANK_PROMPT_VERSION = "v2"

def rerank_with_gpt(
    query: str,
    candidate_ids: list[str],
    chunks: list[dict],
    k: int,
) -> list[str]:
    model = RERANK_MODEL
    prompt_version = RERANK_PROMPT_VERSION

    os.makedirs(
        "cache/reranker",
        exist_ok=True,
    )

    cache_data = {
        "query": query,
        "candidate_ids": candidate_ids,
        "k": k,
        "model": model,
        "prompt_version": prompt_version,
    }

    cache_content = json.dumps(
        cache_data,
        ensure_ascii=False,
        sort_keys=True,
    )

    cache_key = hashlib.sha256(
        cache_content.encode("utf-8")
    ).hexdigest()

    cache_path = (
        f"cache/reranker/{cache_key}.json"
    )

    if os.path.exists(cache_path):
        cached = load_json(cache_path)

        ranked_ids = normalize_reranked_ids(
            cached["ranked_ids"],
            candidate_ids,
        )

        return ranked_ids[:k]

    chunks_by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    candidates = [
        {
            "id": candidate_id,
            "text": chunks_by_id[candidate_id]["text"],
        }
        for candidate_id in candidate_ids
        if candidate_id in chunks_by_id
    ]

    prompt = f"""
    You are a retrieval reranker for a RAG system.

    Your task is to rank passages by how useful they are
    for directly answering the user's query.

    Query:
    {query}

    Candidate passages:
    {json.dumps(candidates, ensure_ascii=False)}

    Ranking criteria, in priority order:

    1. Prefer passages that directly contain the information
       needed to answer the query.

    2. A passage that explicitly states the answer should rank
       above a passage that merely shares similar words or topic.

    3. Prefer specific and actionable information over general
       background information.

    4. Do not reward keyword overlap unless the passage actually
       helps answer the query.

    5. Rank every provided candidate ID exactly once.

    Return the IDs from most relevant to least relevant.
    """

    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "reranking_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": ["ranked_ids"],
                    "additionalProperties": False,
                },
            }
        },
    )

    result = json.loads(
        response.output_text
    )

    ranked_ids = normalize_reranked_ids(
        result["ranked_ids"],
        candidate_ids,
    )

    save_json(
        {
            "ranked_ids": ranked_ids,
        },
        cache_path,
    )

    return ranked_ids[:k]

def retrieve_dense_reranked(
        query: str,
        embedded_chunks: list[dict],
        k: int,
) -> list[str]:
    candidate_k = max(k * 3, k)

    candidate_ids = retrieve_dense(
        query,
        embedded_chunks,
        candidate_k,
    )

    reranked_ids = rerank_with_gpt(
        query,
        candidate_ids,
        embedded_chunks,
        k,
    )

    return reranked_ids

def retrieve_broken(
    query: str,
    chunks: list[dict],
    k: int,
) -> list[str]:
    return []

def validate_golden_set(
        golden_set: list[dict],
        minimum_examples: int = 100,
) -> None:
    if len(golden_set) < minimum_examples:
        raise ValueError(
            f"Golden set has {len(golden_set)} examples. "
            f"Expected at least {minimum_examples}."
        )

    required_fields = {
        "query",
        "relevant_passage_ids",
        "category",
    }

    for index, example in enumerate(golden_set):
        missing_fields = required_fields - example.keys()

        if missing_fields:
            raise ValueError(
                f"Example {index} is missing fields: "
                f"{missing_fields}"
            )

        if not example["relevant_passage_ids"]:
            raise ValueError(
                f"Example {index} has no relevant passages."
            )













def chunk_document(
        text: str,
        chunk_size: int,
        overlap: int,
        prefix: str,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    step = chunk_size - overlap

    chunks = []

    for index, start in enumerate(
        range(0, len(words), step),
        start = 1
    ):
        end = min(
            start + chunk_size,
            len(words),
        )

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append({
            "id": f"{prefix}_{index:03d}",
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": end,
        })

    return chunks

def find_reference_span(
        document: str,
        reference_text: str,
) -> tuple[int, int]:
    document_words = document.split()
    reference_words = reference_text.split()

    if not reference_words:
        raise ValueError("reference_text must not be empty.")

    max_start = len(document_words) - len(reference_words) + 1

    for start in range(max_start):
        end = start + len(reference_words)

        if document_words[start:end] == reference_words:
            return start, end

    raise ValueError(
        "Reference text was not found in the source document."
    )

def find_covering_chunks(
        corpus: list[dict],
        reference_start: int,
        reference_end: int,
) -> list[str]:
    covering_chunks = []

    current_position = reference_start

    while current_position < reference_end:
        candidates = [
            chunk
            for chunk in corpus
            if chunk["start_word"] <= current_position < chunk["end_word"]
        ]

        if not candidates:
            raise ValueError(
                f"No chunk convers word position {current_position}."
            )

        best_chunk = max(
            candidates,
            key=lambda chunk: chunk["end_word"]
        )

        covering_chunks.append(best_chunk["id"])

        current_position = best_chunk["end_word"]

    return covering_chunks

def remap_golden_set_by_spans(
        golden_set: list[dict],
        document: str,
        corpus: list[dict],
) -> list[dict]:
    remapped_examples = []

    for example in golden_set:
        relevant_ids = set()

        for reference_text in example["reference_texts"]:
            start, end = find_reference_span(
                document,
                reference_text
            )

            covering_ids = find_covering_chunks(
                corpus,
                start,
                end,
            )

            relevant_ids.update(covering_ids)

        remapped_example = example.copy()

        remapped_example["relevant_passage_ids"] = sorted(
            relevant_ids
        )

        remapped_examples.append(
            remapped_example
        )

    return remapped_examples

def save_json(
    data: dict,
    file_path: str,
) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_json(
    file_path: str,
) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def create_embedding_cached(
        text: str,
        model: str = "text-embedding-3-small"
) -> list[float]:
    os.makedirs(
        "cache/text_embeddings",
        exist_ok=True,
    )

    cache_data = {
        "text": text,
        "model": model,
    }

    cache_content = json.dumps(
        cache_data,
        ensure_ascii=False,
        sort_keys=True,
    )

    cache_key = hashlib.sha256(
        cache_content.encode("utf-8")
    ).hexdigest()

    cache_path = (
        f"cache/text_embeddings/{cache_key}.json"
    )

    if os.path.exists(cache_path):
        cached = load_json(cache_path)

        return cached["embedding"]

    response = client.embeddings.create(
        model=model,
        input=text,
    )

    embedding = response.data[0].embedding

    save_json(
        {
            "embedding": embedding,
        },
        cache_path,
    )

    return embedding



def build_results_table(
    dense_100: dict,
    dense_200: dict,
    hybrid_200: dict,
    reranked_200: dict,
) -> list[dict]:
    return [
        {
            "configuration": "dense_chunk100",
            "recall_at_10": dense_100["overall"]["mean_recall_at_k"],
            "mrr": dense_100["overall"]["mrr"],
            "ndcg_at_10": dense_100["overall"]["mean_ndcg_at_k"],
        },
        {
            "configuration": "dense_chunk200",
            "recall_at_10": dense_200["overall"]["mean_recall_at_k"],
            "mrr": dense_200["overall"]["mrr"],
            "ndcg_at_10": dense_200["overall"]["mean_ndcg_at_k"],
        },
        {
            "configuration": "hybrid_chunk200",
            "recall_at_10": hybrid_200["overall"]["mean_recall_at_k"],
            "mrr": hybrid_200["overall"]["mrr"],
            "ndcg_at_10": hybrid_200["overall"]["mean_ndcg_at_k"],
        },
        {
            "configuration": (
                f"reranker_{RERANK_MODEL}_"
                f"{RERANK_PROMPT_VERSION}_chunk200"
            ),
            "recall_at_10": reranked_200["overall"]["mean_recall_at_k"],
            "mrr": reranked_200["overall"]["mrr"],
            "ndcg_at_10": reranked_200["overall"]["mean_ndcg_at_k"],
        },
    ]

def save_results_csv(
    rows: list[dict],
    file_path: str,
) -> None:
    if not rows:
        return

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

def find_problematic_queries(
        experiment: dict,
        limit: int = 5,
) -> list[dict]:
    problematic = [
        detail
        for detail in experiment["details"]
        if (
            detail["recall_at_k"] < 1.0
            or detail["reciprocal_rank"] < 1.0
            or detail["ndcg_at_k"] < 1.0
        )
    ]

    problematic.sort(
        key=lambda detail: (
            detail["recall_at_k"],
            detail["ndcg_at_k"],
            detail["reciprocal_rank"],
        )
    )

    return problematic[:limit]

def main() -> None:
    k = 10
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    golden_set = load_golden_set("golden_set.jsonl")
    validate_golden_set(golden_set)
    with open("docs/cast_ai_engineering.md", "r", encoding="utf-8") as file:
        document = file.read()

    corpus_100 = chunk_document(document, chunk_size=100, overlap=20, prefix="chunk100")
    corpus_200 = chunk_document(document, chunk_size=200, overlap=40, prefix="chunk200")
    save_json(corpus_200, "cache/chunks_chunk200.json")

    golden_100 = remap_golden_set_by_spans(golden_set, document, corpus_100)
    golden_200 = remap_golden_set_by_spans(golden_set, document, corpus_200)

    def load_or_embed(corpus: list[dict], cache_path: str) -> list[dict]:
        if os.path.exists(cache_path):
            return load_json(cache_path)
        embedded = embed_chunks(corpus)
        save_json(embedded, cache_path)
        return embedded

    embedded_100 = load_or_embed(corpus_100, "cache/embeddings_chunk100.json")
    embedded_200 = load_or_embed(corpus_200, "cache/embeddings_chunk200.json")

    def load_or_run(path: str, golden: list[dict], retriever, chunks: list[dict]) -> dict:
        if os.path.exists(path):
            return load_json(path)
        result = run_experiment(golden, retriever, chunks, k=k)
        save_json(result, path)
        return result

    dense_100 = load_or_run(
        f"results/dense_chunk100_k{k}.json", golden_100, retrieve_dense, embedded_100
    )
    dense_200 = load_or_run(
        f"results/dense_chunk200_k{k}.json", golden_200, retrieve_dense, embedded_200
    )
    hybrid_200 = load_or_run(
        f"results/hybrid_chunk200_k{k}.json", golden_200, retrieve_hybrid, embedded_200
    )

    reranked_path = (
        f"results/reranked_{RERANK_MODEL}_{RERANK_PROMPT_VERSION}_chunk200_k{k}.json"
    )
    reranked_200 = load_or_run(
        reranked_path, golden_200, retrieve_dense_reranked, embedded_200
    )
    reranked_200.setdefault("config", {
        "retriever": "dense_reranker",
        "reranker_model": RERANK_MODEL,
        "prompt_version": RERANK_PROMPT_VERSION,
        "chunk_size": 200,
        "overlap": 40,
        "k": k,
        "candidate_k": k * 3,
    })
    save_json(reranked_200, reranked_path)

    results_table = build_results_table(dense_100, dense_200, hybrid_200, reranked_200)
    save_results_csv(results_table, "results/retrieval_comparison.csv")

    print("Retrieval results:")
    for row in results_table:
        print(row)

    print("\nFive queries to inspect:")
    for index, item in enumerate(find_problematic_queries(reranked_200), start=1):
        print(
            f"#{index} [{item['category']}] {item['query']} "
            f"(Recall={item['recall_at_k']:.3f}, "
            f"RR={item['reciprocal_rank']:.3f}, nDCG={item['ndcg_at_k']:.3f})"
        )

def main() -> None:
    k = 10
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    golden_set = load_golden_set("golden_set.jsonl")
    validate_golden_set(golden_set)
    with open("docs/sources/ai_engineering.md", "r", encoding="utf-8") as file:
        document = file.read()

    corpus_100 = chunk_document(document, chunk_size=100, overlap=20, prefix="chunk100")
    corpus_200 = chunk_document(document, chunk_size=200, overlap=40, prefix="chunk200")
    save_json(corpus_200, "cache/chunks_chunk200.json")

    golden_100 = remap_golden_set_by_spans(golden_set, document, corpus_100)
    golden_200 = remap_golden_set_by_spans(golden_set, document, corpus_200)

    def load_or_embed(corpus: list[dict], cache_path: str) -> list[dict]:
        if os.path.exists(cache_path):
            return load_json(cache_path)
        embedded = embed_chunks(corpus)
        save_json(embedded, cache_path)
        return embedded

    embedded_100 = load_or_embed(corpus_100, "cache/embeddings_chunk100.json")
    embedded_200 = load_or_embed(corpus_200, "cache/embeddings_chunk200.json")

    def load_or_run(path: str, golden: list[dict], retriever, chunks: list[dict]) -> dict:
        if os.path.exists(path):
            return load_json(path)
        result = run_experiment(golden, retriever, chunks, k=k)
        save_json(result, path)
        return result

    dense_100 = load_or_run(
        f"results/dense_chunk100_k{k}.json", golden_100, retrieve_dense, embedded_100
    )
    dense_200 = load_or_run(
        f"results/dense_chunk200_k{k}.json", golden_200, retrieve_dense, embedded_200
    )
    hybrid_200 = load_or_run(
        f"results/hybrid_chunk200_k{k}.json", golden_200, retrieve_hybrid, embedded_200
    )

    reranked_path = (
        f"results/reranked_{RERANK_MODEL}_{RERANK_PROMPT_VERSION}_chunk200_k{k}.json"
    )
    reranked_200 = load_or_run(
        reranked_path, golden_200, retrieve_dense_reranked, embedded_200
    )
    reranked_200.setdefault("config", {
        "retriever": "dense_reranker",
        "reranker_model": RERANK_MODEL,
        "prompt_version": RERANK_PROMPT_VERSION,
        "chunk_size": 200,
        "overlap": 40,
        "k": k,
        "candidate_k": k * 3,
    })
    save_json(reranked_200, reranked_path)

    results_table = build_results_table(dense_100, dense_200, hybrid_200, reranked_200)
    save_results_csv(results_table, "results/retrieval_comparison.csv")

    print("Retrieval results:")
    for row in results_table:
        print(row)

    print("\nFive queries to inspect:")
    for index, item in enumerate(find_problematic_queries(reranked_200), start=1):
        print(
            f"#{index} [{item['category']}] {item['query']} "
            f"(Recall={item['recall_at_k']:.3f}, "
            f"RR={item['reciprocal_rank']:.3f}, nDCG={item['ndcg_at_k']:.3f})"
        )


if __name__ == "__main__":
    main()
