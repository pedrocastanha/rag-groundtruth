import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
)
from main import (
    load_json,
    save_json,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

evaluation_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

evaluation_llm = llm_factory(
    "gpt-4.1-mini",
    client=evaluation_client,
    max_tokens=4096,
)

evaluation_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=evaluation_client,
)

faithfulness_scorer = Faithfulness(
    llm=evaluation_llm,
)

answer_relevancy_scorer = AnswerRelevancy(
    llm=evaluation_llm,
    embeddings=evaluation_embeddings,
)

def generate_answer(
        query: str,
        contexts: list[str],
        model: str = "gpt-4.1-mini"
) -> str:
    formatted_context = "\n\n".join(
        f"[Passage {index}]\n{context}"
        for index, context in enumerate(
            contexts,
            start=1
        )
    )

    prompt = f"""
Answer the user's question using only the provided context.

Rules:
- Do not use information outside the context.
- If the context does not contain enough information,
  say that the available context is insufficient.
- Be concise and directly answer the question.

User question:
{query}

Context:
{formatted_context}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text.strip()

async def evaluate_generation(
    query: str,
    response: str,
    contexts: list[str],
) -> dict:
    faithfulness_result, relevancy_result = await asyncio.gather(
        faithfulness_scorer.ascore(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
        ),
        answer_relevancy_scorer.ascore(
            user_input=query,
            response=response,
        ),
    )

    return {
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
    }

def get_contexts_from_ids(
    retrieved_ids: list[str],
    chunks: list[dict],
    limit: int = 5,
) -> list[str]:
    chunks_by_id = {
        chunk["id"]: chunk["text"]
        for chunk in chunks
    }

    return [
        chunks_by_id[chunk_id]
        for chunk_id in retrieved_ids[:limit]
        if chunk_id in chunks_by_id
    ]

def select_representative_sample(
        details: list[dict],
        sample_size: int = 20,
) -> list[dict]:
    by_category = {}

    for item in details:
        category = item["category"]

        by_category.setdefault(
            category,
            []
        ).append(item)

    sample = []

    while len(sample) < sample_size:
        added = False

        for category_items in by_category.values():
            if category_items:
                sample.append(
                    category_items.pop(0)
                )

                added = True

                if len(sample) == sample_size:
                    break

        if not added:
            break

    return sample

async def evaluate_sample_item(
    item: dict,
    chunks: list[dict],
) -> dict:
    query = item["query"]

    retrieved_ids = item[
        "retrieved_passage_ids"
    ]

    contexts = get_contexts_from_ids(
        retrieved_ids,
        chunks,
        limit=5,
    )

    answer = generate_answer(
        query,
        contexts,
    )

    try:
        metrics = await evaluate_generation(
            query,
            answer,
            contexts,
        )

        return {
            "query": query,
            "category": item["category"],
            "answer": answer,
            "faithfulness": metrics["faithfulness"],
            "answer_relevancy": metrics["answer_relevancy"],
            "error": None,
        }

    except Exception as error:
        return {
            "query": query,
            "category": item["category"],
            "answer": answer,
            "faithfulness": None,
            "answer_relevancy": None,
            "error": str(error),
        }

async def evaluate_sample(
    sample: list[dict],
    chunks: list[dict],
    output_path: str,
) -> list[dict]:
    results = []

    for index, item in enumerate(
        sample,
        start=1,
    ):
        print(
            f"\n[{index}/{len(sample)}] "
            f"{item['category']} -> {item['query']}"
        )

        result = await evaluate_sample_item(
            item,
            chunks,
        )

        results.append(result)

        save_json(
            results,
            output_path,
        )

        if result["error"]:
            print(
                "Evaluation error:",
                result["error"],
            )
        else:
            print(
                "Faithfulness:",
                result["faithfulness"],
            )

            print(
                "Answer relevancy:",
                result["answer_relevancy"],
            )

    return results

def aggregate_generation_results(
        results: list[dict],
) -> dict:
    valid_results = [
        result
        for result in results
        if result["error"] is None
        and result["faithfulness"] is not None
        and result["answer_relevancy"] is not None
    ]

    failed_results = [
        result
        for result in results
        if result not in valid_results
    ]

    if not valid_results:
        return {
            "total": len(results),
            "successful": 0,
            "failed": len(failed_results),
            "mean_faithfulness": None,
            "mean_answer_relevancy": None,
        }

    mean_faithfulness = sum(
        result["faithfulness"]
        for result in valid_results
    ) / len(valid_results)

    mean_answer_relevancy = sum(
        result["answer_relevancy"]
        for result in valid_results
    ) / len(valid_results)

    return {
        "total": len(results),
        "successful": len(valid_results),
        "failed": len(failed_results),
        "mean_faithfulness": mean_faithfulness,
        "mean_answer_relevancy": mean_answer_relevancy,
    }

if __name__ == "__main__":
    chunks = load_json(
        "cache/chunks_chunk200.json"
    )

    experiment = load_json(
        "results/reranked_gpt-4.1-mini_v2_chunk200_k10.json"
    )

    sample = select_representative_sample(
        experiment["details"],
        sample_size=12,
    )

    # results = asyncio.run(
    #     evaluate_sample(
    #         sample,
    #         chunks,
    #         "results/generation_eval_sample12.json",
    #     )
    # )

    results = load_json(
        "results/generation_eval_sample12.json"
    )

    summary = aggregate_generation_results(
        results
    )

    save_json(
        summary,
        "results/generation_eval_summary.json",
    )

    print("\nGeneration evaluation summary:")
    print(summary)

