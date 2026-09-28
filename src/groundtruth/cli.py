from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundtruth.corpus.chunking import attach_relevant_chunk_ids
from groundtruth.experiments.config import (
    load_config, load_query_cases, unresolved_reviews, validate_dataset_references,
)
from groundtruth.evaluation.agent_matrix import (
    build_agent_matrix,
    render_agent_matrix_markdown,
)
from groundtruth.experiments.runner import load_corpus, run_agent_experiment, run_retrieval_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="groundtruth")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-dataset", "prepare-corpus", "evaluate-retrieval", "evaluate-agents"):
        command = commands.add_parser(name)
        default_config = (
            "configs/experiments/agent_controlled.yaml"
            if name == "evaluate-agents"
            else "configs/experiments/retrieval_controlled.yaml"
        )
        command.add_argument("--config", default=default_config)
        if name == "evaluate-agents":
            command.add_argument("--without-generation-eval", action="store_true",
                                 help="Skip RAGAS Faithfulness and Answer Relevancy calls")
    matrix_command = commands.add_parser(
        "compare-agents", help="Build and validate the full 18-cell agent comparison"
    )
    matrix_command.add_argument("--results-dir", default="results/multidoc")
    matrix_command.add_argument("--baseline", default="baselines/multidoc_v1.json")
    matrix_command.add_argument("--expected-generation-samples", type=int, default=4)
    matrix_command.add_argument(
        "--output-json", default="results/multidoc/agent_matrix.json"
    )
    matrix_command.add_argument(
        "--output-markdown", default="results/multidoc/agent_matrix.md"
    )
    args = parser.parse_args()
    if args.command == "compare-agents":
        matrix = build_agent_matrix(
            args.results_dir,
            args.baseline,
            expected_generation_samples=args.expected_generation_samples,
        )
        json_path = Path(args.output_json)
        markdown_path = Path(args.output_markdown)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(render_agent_matrix_markdown(matrix), encoding="utf-8")
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
        if not matrix["complete"]:
            raise SystemExit(1)
        return
    config = load_config(args.config)
    cases = load_query_cases(config)

    if args.command == "validate-dataset":
        _, chunks, _ = load_corpus(config)
        problems = validate_dataset_references(cases, chunks)
        mapped = attach_relevant_chunk_ids(cases, chunks) if not problems else {}
        print(json.dumps({
            "queries": len(cases), "pending_review": unresolved_reviews(cases),
            "problems": problems,
            "queries_without_matching_reference": [key for key, ids in mapped.items() if not ids],
            "mapped_relevant_chunks": sum(len(ids) for ids in mapped.values()),
        }, ensure_ascii=False, indent=2))
    elif args.command == "prepare-corpus":
        documents, chunks, fingerprint = load_corpus(config)
        print(json.dumps({"documents": len(documents), "chunks": len(chunks),
                          "corpus_fingerprint": fingerprint}, indent=2))
    elif args.command == "evaluate-retrieval":
        print(json.dumps(run_retrieval_experiment(config), ensure_ascii=False, indent=2))
    else:
        result = run_agent_experiment(
            config, include_generation_eval=not args.without_generation_eval
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
