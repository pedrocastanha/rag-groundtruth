import re
import json
import math
from openai import OpenAI
import os
from dotenv import load_dotenv
from collections import Counter

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
    query_embeddings = create_embedding(query)

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

    return {
        "overall": overall_metrics,
        "by_category": category_metrics,
    }

def retrieve_hybrid(
        query: str,
        embedded_chunks: list[dict],
        k: int,
        alpha: float = 0.5,
):
    query_words = set(tokenize(query))
    query_embeddings = create_embedding(query)

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

def rerank_with_gpt(
        query: str,
        candidate_ids: list[str],
        chunks: list[dict],
        k: int,
) -> list[str]:
    candidates = []

    for chunk in chunks:
        if chunk["id"] in candidate_ids:
            candidates.append({
                "id": chunk["id"],
                "text": chunk["text"],
            })

    prompt = f"""
You are a retrieval reranker.

Query:
{query}

Candidate passages:
{json.dumps(candidates, ensure_ascii=False)}

Rank the candidate passage IDs from most relevant
to least relevant for answering the query.

Return ONLY a JSON array of IDs.

Example:
["chunk_10", "chunk_42"]
"""

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
    )

    ranked_ids = json.loads(response.output_text)

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

if __name__ == "__main__":
    golden_set = load_golden_set("golden_set.jsonl")

    with open(
        "docs/cast_ai_engineering.md",
        "r",
        encoding="utf-8",
    ) as file:
        document = file.read()

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

    print("Corpus 100:")
    print(len(corpus_100))
    print(corpus_100[0])
    print(corpus_100[1])

    print("\nCorpus 200:")
    print(len(corpus_200))
    print(corpus_200[0])
    print(corpus_200[1])

    reference_text = golden_set[0]["reference_texts"][0]

    start, end = find_reference_span(
        document,
        reference_text,
    )

    print("\nReference:")
    print(reference_text)

    print("\nReference position:")
    print("Start:", start)
    print("End:", end)
    print("Length:", end - start)

    print("\nFound text:")
    print(
        " ".join(
            document.split()[start:end]
        )
    )

    relevant_100 = find_covering_chunks(
        corpus_100,
        start,
        end,
    )

    relevant_200 = find_covering_chunks(
        corpus_200,
        start,
        end,
    )

    print("\nRelevant chunks - 100:")
    print(relevant_100)

    print("\nRelevant chunks - 200:")
    print(relevant_200)

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

    print("\nGolden 100:")
    print(len(golden_100))
    print(golden_100[0]["relevant_passage_ids"])

    print("\nGolden 200:")
    print(len(golden_200))
    print(golden_200[0]["relevant_passage_ids"])

    print("\nEmbedding corpus 100...")
    embedded_100 = embed_chunks(corpus_100)

    print("Embedding corpus 200...")
    embedded_200 = embed_chunks(corpus_200)

    print("\nRunning Dense Retrieval - chunk size 100...")
    dense_100 = run_experiment(
        golden_100,
        retrieve_dense,
        embedded_100,
        k=10,
    )

    print("\nRunning Dense Retrieval - chunk size 200...")
    dense_200 = run_experiment(
        golden_200,
        retrieve_dense,
        embedded_200,
        k=10,
    )

    print("\nDense - Chunk size 100:")
    print(dense_100)

    print("\nDense - Chunk size 200:")
    print(dense_200)

    comparison = compare_metrics(
        dense_100["overall"],
        dense_200["overall"],
    )

    print("\nChunk 100 -> Chunk 200:")
    print(comparison)