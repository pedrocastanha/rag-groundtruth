import hashlib
import json
import math
import os
import re
from collections import Counter

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
def mean_reciprocal_rank(
        queries: list[dict],
) -> float:
    if not queries:
        return 0.0

    scores = []

    for query in queries:
        score = reciprocal_rank(
            query["relevant_passage_ids"],
            query["retrieved_passage_ids"],
        )

        scores.append(score)

    return sum(scores) / len(queries)

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
def ideal_dcg_at_k(
        relevant_scores: list[int],
        k: int,
) -> float:
    ideal_scores = sorted(
        relevant_scores,
        reverse = True,
    )

    return dcg_at_k(ideal_scores, k)

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

def compare_metrics(
        baseline: dict,
        candidate: dict,
) -> dict:
    return {
        "recall_delta": (
            candidate["mean_recall_at_k"]
            - baseline["mean_recall_at_k"]
        ),
        "mrr_delta": (
            candidate["mrr"]
            - baseline["mrr"]
        ),
        "ndcg_delta": (
            candidate["mean_ndcg_at_k"]
            - baseline["mean_ndcg_at_k"]
        ),
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

def retrieve_keyword(
        query: str,
        chunks: list[dict],
        k: int
) -> list[str]:
    query_words = set(tokenize(query))

    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(tokenize(chunk["text"]))

        score = len(
            query_words.intersection(chunk_words)
        )

        scored_chunks.append(
            (score, chunk["id"])
        )

    scored_chunks.sort(reverse=True)

    return [
        chunk_id
        for score, chunk_id in scored_chunks[:k]
    ]

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

def generate_candidate_example(
    chunk: dict,
) -> dict:
    allowed_categories = [
        "negligence",
        "contract",
        "duty",
        "damages",
        "liability",
    ]

    prompt = f"""
You are creating an evaluation dataset for a legal retrieval system.

Passage ID:
{chunk["id"]}

Passage:
{chunk["text"]}

Create ONE realistic user question that can be answered by this passage.

Requirements:
- Paraphrase the passage instead of copying its wording.
- The question should test semantic retrieval.
- Do not mention the passage ID.
- Choose exactly ONE category from this list:
{allowed_categories}
- Return only valid JSON in this format:

{{
    "query": "...",
    "category": "..."
}}
"""

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
    )

    generated = json.loads(response.output_text)

    return {
        "query": generated["query"],
        "relevant_passage_ids": [chunk["id"]],
        "category": chunk["category"],
        "review_status": "pending",
    }

def validate_corpus_size(
        chunks: list[dict],
        k: int,
        minimum_multiplier: int = 5,
) -> None:
    minimum_chunks = k * minimum_multiplier

    if len(chunks) < minimum_chunks:
        raise ValueError(
            f"Corpus has only {len(chunks)} chunks. "
            f"For Recall@{k}, use at least {minimum_chunks} chunks."
        )

def generate_corpus_passage(
        chunk_id: str,
        category: str,
) -> dict:
    prompt = f"""
    You are creating a synthetic corpus for evaluating a legal research retrieval system.

    Category: {category}

    Create ONE short legal passage.

    Requirements:
    - Write 2 to 4 sentences.
    - The passage should contain one specific legal rule, principle, exception, or scenario.
    - Make it meaningfully different from generic textbook definitions.
    - Do not mention that the passage is synthetic.
    - Do not include a title.
    - Return only the passage text.
    """

    response = client.responses.create(
        model="gpt-4.1-nano",
        input = prompt,
    )

    return {
        "id": chunk_id,
        "text": response.output_text.strip(),
        "category": category,
    }

def generate_synthetic_corpus(
        categories: list[str],
        passages_per_category: int,
        start_index: int = 100
) -> list[dict]:
    generated_chunks = []
    next_index = start_index

    for category in categories:
        for _ in range(passages_per_category):
            chunk_id = f"chunk_{next_index}"

            chunk = generate_corpus_passage(
                chunk_id = chunk_id,
                category = category,
            )

            generated_chunks.append(chunk)
            next_index += 1

    return generated_chunks

def save_jsonl(
        data: list[dict],
        file_path:str
) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

def generate_candidates_for_corpus(
        chunks: list[dict],
        queries_per_chunk: int = 2,
) -> list[dict]:
    candidates = []
    candidate_index = 1

    for chunk in chunks:
        for _ in range(queries_per_chunk):
            candidate = generate_candidate_example(chunk)

            candidate["candidate_id"] = f"candidate_{candidate_index}"

            candidates.append(candidate)

            candidate_index += 1

    return candidates

def review_candidates(
        candidates: list[dict],
        chunks: list[dict],
        target_approved: int = 100,
) -> list[dict]:
    chunks_by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    approved = []

    for candidate in candidates:
        if len(approved) >= target_approved:
            break

        passage_id = candidate["relevant_passage_ids"][0]
        passage = chunks_by_id[passage_id]

        print("\n" + "=" * 80)
        print(f'Candidate: {candidate["candidate_id"]}')
        print(f'Category: {candidate["category"]}')
        print(f'\nQuery:\n{candidate["query"]}')
        print(f'\nRelevant passage ({passage_id}):\n{passage["text"]}')

        decision = input(
            "\nApprove? [y/n/q]: "
        ).strip().lower()

        if decision == "y":
            candidate["review_status"] = "approved"
            approved.append(candidate)

        elif decision == "n":
            candidate["review_status"] = "rejected"

        elif decision == "q":
            break

    return approved

def chunk_text(
        text: str,
        chunk_size: int,
        overlap: int = 0,
) -> list[str]:
    words = text.split()

    chunks = []

    step = chunk_size - overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    for start in range(0, len(words), step):
        end = start + chunk_size

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks

def attach_reference_texts(
        golden_set: list[dict],
        chunks: list[dict],
) -> list[dict]:
    chunks_by_id = {
        chunk["id"]: chunk["text"]
        for chunk in chunks
    }

    enriched_golden_set = []

    for example in golden_set:
        reference_texts = []

        for passage_id in example["relevant_passage_ids"]:
            if passage_id not in chunks_by_id:
                raise ValueError(
                    f"Relevant passage not found: {passage_id}"
                )

            reference_texts.append(
                chunks_by_id[passage_id]
            )

        enriched_example = example.copy()

        enriched_example["reference_texts"] = reference_texts

        enriched_golden_set.append(enriched_example)

    return enriched_golden_set

def build_chunk_records(
        chunk_texts: list[str],
        prefix: str,
) -> list[dict]:
    chunks = []

    for index, text in enumerate(chunk_texts, start=1):
        chunks.append({
            "id": f"{prefix}_{index:03d}",
            "text": text,
        })

    return chunks

def text_containment_score(
        reference_text: str,
        chunk_text: str,
) -> float:
    reference_tokens = tokenize(reference_text.lower())
    chunk_tokens = tokenize(chunk_text.lower())

    if not reference_tokens or not chunk_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    chunk_counts = Counter(chunk_tokens)

    common_tokens = reference_counts & chunk_counts

    overlap = sum(common_tokens.values())

    return overlap / min(
        len(reference_tokens),
        len(chunk_tokens),
    )

def remap_golden_set_to_corpus(
        golden_set: list[dict],
        corpus: list[dict],
        minimum_score: float = 0.5,
) -> list[dict]:
    remapped_examples = []

    for example in golden_set:
        relevant_ids = set()

        for reference_text in example["reference_texts"]:
            scored_chunks = []

            for chunk in corpus:
                score = text_containment_score(
                    reference_text,
                    chunk["text"],
                )

                if score >= minimum_score:
                    scored_chunks.append(
                        (score, chunk["id"])
                    )

            scored_chunks.sort(
                reverse=True,
            )

            for _, chunk_id in scored_chunks:
                relevant_ids.add(chunk_id)

        remapped_example = example.copy()

        remapped_example["relevant_passage_ids"] = sorted(
            relevant_ids
        )

        remapped_examples.append(
            remapped_example
        )

    return remapped_examples

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

def find_largest_regressions(
        baseline: dict,
        candidate: dict,
        metric: str,
        limit: int = 5,
) -> list[dict]:
    baseline_by_query = {
        item["query"]: item
        for item in baseline["details"]
    }

    regressions = []

    for candidate_item in candidate["details"]:
        query = candidate_item["query"]

        baseline_item = baseline_by_query[query]

        delta = (
            candidate_item[metric]
            - baseline_item[metric]
        )

        regressions.append({
            "query": query,
            "category": candidate_item["category"],
            "delta": delta,
            "baseline_score": baseline_item[metric],
            "candidate_score": candidate_item[metric],
            "relevant_passage_ids": (
                candidate_item["relevant_passage_ids"]
            ),
            "baseline_retrieved": (
            baseline_item["retrieved_passage_ids"]
            ),
            "candidate_retrieved": (
                candidate_item["retrieved_passage_ids"]
            )
        })

    regressions.sort(
        key=lambda item: item["delta"]
    )

    return regressions[:limit]

def print_ranking_with_text(
        retrieved_ids: list[str],
        chunks: list[dict],
) -> None:
    chunks_by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    for rank, passage_id in enumerate(
        retrieved_ids,
        start = 1,
    ):
        chunk = chunks_by_id[passage_id]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {passage_id}")
        print("Text:")
        print(chunk["text"])

if __name__ == "__main__":
    K = 10

    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    golden_set = load_golden_set(
        "golden_set.jsonl"
    )

    with open(
        "docs/cast_ai_engineering.md",
        "r",
        encoding="utf-8",
    ) as file:
        document = file.read()

    # ---------------------------------
    # Build corpora
    # ---------------------------------

    corpus_100 = chunk_document(
        document,
        chunk_size=100,
        overlap=20,
        prefix="chunk100",
    )

    corpus_200 = chunk_document(
        document,
        chunk_size=200,
        overlap=40,
        prefix="chunk200",
    )

    # ---------------------------------
    # Remap ground truth
    # ---------------------------------

    golden_100 = remap_golden_set_by_spans(
        golden_set,
        document,
        corpus_100,
    )

    golden_200 = remap_golden_set_by_spans(
        golden_set,
        document,
        corpus_200,
    )

    # ---------------------------------
    # Embeddings - chunk size 100
    # ---------------------------------

    embeddings_100_path = (
        "cache/embeddings_chunk100.json"
    )

    if os.path.exists(embeddings_100_path):
        print("Loading cached embeddings - chunk 100...")

        embedded_100 = load_json(
            embeddings_100_path
        )
    else:
        print("Creating embeddings - chunk 100...")

        embedded_100 = embed_chunks(
            corpus_100
        )

        save_json(
            embedded_100,
            embeddings_100_path,
        )

    # ---------------------------------
    # Embeddings - chunk size 200
    # ---------------------------------

    embeddings_200_path = (
        "cache/embeddings_chunk200.json"
    )

    if os.path.exists(embeddings_200_path):
        print("Loading cached embeddings - chunk 200...")

        embedded_200 = load_json(
            embeddings_200_path
        )
    else:
        print("Creating embeddings - chunk 200...")

        embedded_200 = embed_chunks(
            corpus_200
        )

        save_json(
            embedded_200,
            embeddings_200_path,
        )

    # ---------------------------------
    # Dense - chunk 100
    # ---------------------------------

    dense_100_path = (
        f"results/dense_chunk100_k{K}.json"
    )

    if os.path.exists(dense_100_path):
        print("Loading Dense chunk 100...")

        dense_100 = load_json(
            dense_100_path
        )
    else:
        print("Running Dense chunk 100...")

        dense_100 = run_experiment(
            golden_100,
            retrieve_dense,
            embedded_100,
            k=K,
        )

        save_json(
            dense_100,
            dense_100_path,
        )

    # ---------------------------------
    # Dense - chunk 200
    # ---------------------------------

    dense_200_path = (
        f"results/dense_chunk200_k{K}.json"
    )

    if os.path.exists(dense_200_path):
        print("Loading Dense chunk 200...")

        dense_200 = load_json(
            dense_200_path
        )
    else:
        print("Running Dense chunk 200...")

        dense_200 = run_experiment(
            golden_200,
            retrieve_dense,
            embedded_200,
            k=K,
        )

        save_json(
            dense_200,
            dense_200_path,
        )

    # ---------------------------------
    # Hybrid - chunk 200
    # ---------------------------------

    hybrid_200_path = (
        f"results/hybrid_chunk200_k{K}.json"
    )

    if os.path.exists(hybrid_200_path):
        print("Loading Hybrid chunk 200...")

        hybrid_200 = load_json(
            hybrid_200_path
        )
    else:
        print("Running Hybrid chunk 200...")

        hybrid_200 = run_experiment(
            golden_200,
            retrieve_hybrid,
            embedded_200,
            k=K,
        )

        save_json(
            hybrid_200,
            hybrid_200_path,
        )

    # ---------------------------------
    # Dense + Reranker - chunk 200
    # ---------------------------------

    reranked_200_path = (
        f"results/"
        f"reranked_{RERANK_MODEL}_"
        f"{RERANK_PROMPT_VERSION}_"
        f"chunk200_k{K}.json"
    )

    if os.path.exists(reranked_200_path):
        print("Loading Dense + Reranker...")

        reranked_200 = load_json(
            reranked_200_path
        )
    else:
        print("Running Dense + Reranker...")

        reranked_200 = run_experiment(
            golden_200,
            retrieve_dense_reranked,
            embedded_200,
            k=K,
        )

        reranked_200["config"] = {
            "retriever": "dense_reranker",
            "reranker_model": RERANK_MODEL,
            "prompt_version": RERANK_PROMPT_VERSION,
            "chunk_size": 200,
            "overlap": 40,
            "k": K,
            "candidate_k": K * 3,
        }

        save_json(
            reranked_200,
            reranked_200_path,
        )

    # ---------------------------------
    # Results
    # ---------------------------------

    print("\nDense - chunk 100:")
    print(dense_100["overall"])

    print("\nDense - chunk 200:")
    print(dense_200["overall"])

    print("\nHybrid - chunk 200:")
    print(hybrid_200["overall"])

    print("\nDense + Reranker - chunk 200:")
    print(reranked_200["overall"])

    print("\nChunk 100 -> Chunk 200:")
    print(
        compare_metrics(
            dense_100["overall"],
            dense_200["overall"],
        )
    )

    print("\nDense -> Hybrid:")
    print(
        compare_metrics(
            dense_200["overall"],
            hybrid_200["overall"],
        )
    )

    print("\nDense -> Dense + Reranker:")
    print(
        compare_metrics(
            dense_200["overall"],
            reranked_200["overall"],
        )
    )

    worst_reranker_queries = find_largest_regressions(
        dense_200,
        reranked_200,
        metric="reciprocal_rank",
        limit=5,
    )

    print("\nWorst reranker regressions:")

    for regression in worst_reranker_queries:
        print("\n" + "=" * 80)
        print("Query:", regression["query"])
        print("Category:", regression["category"])
        print("Delta:", regression["delta"])
        print(
            "Baseline score:",
            regression["baseline_score"],
        )
        print(
            "Reranker score:",
            regression["candidate_score"],
        )
        print(
            "Relevant:",
            regression["relevant_passage_ids"],
        )
        print(
            "Dense:",
            regression["baseline_retrieved"],
        )
        print(
            "Reranked:",
            regression["candidate_retrieved"],
        )

    print("\nDEBUG DENSE DETAIL:")

    dense_detail = dense_200["details"][0]

    print("Query:")
    print(dense_detail["query"])

    print("Relevant:")
    print(dense_detail["relevant_passage_ids"])

    print("Retrieved:")
    print(dense_detail["retrieved_passage_ids"])

    print(
        "Retrieved count:",
        len(dense_detail["retrieved_passage_ids"]),
    )

    print(
        "Saved RR:",
        dense_detail["reciprocal_rank"],
    )

    print(
        "Recomputed RR:",
        reciprocal_rank(
            dense_detail["relevant_passage_ids"],
            dense_detail["retrieved_passage_ids"],
        ),
    )

    print("\nDEBUG RERANKER DETAIL:")

    reranked_detail = reranked_200["details"][0]

    print("Query:")
    print(reranked_detail["query"])

    print("Relevant:")
    print(reranked_detail["relevant_passage_ids"])

    print("Retrieved:")
    print(reranked_detail["retrieved_passage_ids"])

    print(
        "Retrieved count:",
        len(reranked_detail["retrieved_passage_ids"]),
    )

    print(
        "Saved RR:",
        reranked_detail["reciprocal_rank"],
    )

    print(
        "Recomputed RR:",
        reciprocal_rank(
            reranked_detail["relevant_passage_ids"],
            reranked_detail["retrieved_passage_ids"],
        ),
    )

    debug_query = (
        "Para uma tarefa de extração de dados estruturados que alimenta outro sistema, "
        "que configuração de temperatura e formato de saída é recomendada?"
    )

    dense_debug = next(
        item
        for item in dense_200["details"]
        if item["query"] == debug_query
    )

    reranked_debug = next(
        item
        for item in reranked_200["details"]
        if item["query"] == debug_query
    )

    print("\nDENSE PROBLEM QUERY:")
    print(dense_debug)
    print(
        "Recomputed RR:",
        reciprocal_rank(
            dense_debug["relevant_passage_ids"],
            dense_debug["retrieved_passage_ids"],
        ),
    )

    print("\nRERANKER PROBLEM QUERY:")
    print(reranked_debug)
    print(
        "Recomputed RR:",
        reciprocal_rank(
            reranked_debug["relevant_passage_ids"],
            reranked_debug["retrieved_passage_ids"],
        ),
    )

    print("\nDENSE RANKING TEXTS:")

    print_ranking_with_text(
        dense_debug["retrieved_passage_ids"],
        embedded_200,
    )

    print("\nRERANKED RANKING TEXTS:")

    print_ranking_with_text(
        reranked_debug["retrieved_passage_ids"],
        embedded_200,
    )

    print("\nGROUND TRUTH:")

    print_ranking_with_text(
        reranked_debug["relevant_passage_ids"],
        embedded_200,
    )

    worst_query = worst_reranker_queries[0]["query"]

    dense_worst = next(
        item
        for item in dense_200["details"]
        if item["query"] == worst_query
    )

    reranked_worst = next(
        item
        for item in reranked_200["details"]
        if item["query"] == worst_query
    )

    print("\nWORST QUERY:")
    print(worst_query)

    print("\nGROUND TRUTH:")
    print_ranking_with_text(
        reranked_worst["relevant_passage_ids"],
        embedded_200,
    )

    print("\nDENSE:")
    print_ranking_with_text(
        dense_worst["retrieved_passage_ids"][:3],
        embedded_200,
    )

    print("\nRERANKER:")
    print_ranking_with_text(
        reranked_worst["retrieved_passage_ids"][:5],
        embedded_200,
    )

    print("\nTESTING WORST QUERY WITH MINI:")

    mini_result = retrieve_dense_reranked(
        worst_query,
        embedded_200,
        K,
    )

    print("Relevant:")
    print(reranked_worst["relevant_passage_ids"])

    print("Mini ranking:")
    print(mini_result)

    print_ranking_with_text(
        mini_result[:5],
        embedded_200,
    )