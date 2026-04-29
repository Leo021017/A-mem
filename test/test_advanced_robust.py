"""
Robust evaluation harness.

Modes:
  - vanilla: only no-attention retrieval, and persist per-sample JSONL under results/
  - attention: run attention retrieval and compare against saved vanilla JSONL (no vanilla re-run)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any

# Allow running from repo root or from test/ directory.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mem_agent import RobustAdvancedMemAgent
from load_dataset import load_locomo_dataset
from utils import calculate_metrics, aggregate_metrics

logger = logging.getLogger("amem_robust")


def _safe_name(s: str) -> str:
    s = str(s)
    for ch in ["/", "\\", " ", ":", "|", "\t", "\n", "\r"]:
        s = s.replace(ch, "_")
    return s


def _dataset_tag(dataset_path: str) -> str:
    p = Path(dataset_path)
    name = p.name
    if name.lower().endswith(".json"):
        name = name[: -len(".json")]
    return _safe_name(name)


def _parse_int_list(csv: Optional[str]) -> Optional[List[int]]:
    if csv is None:
        return None
    items = [x.strip() for x in csv.split(",") if x.strip()]
    if not items:
        return None
    return [int(x) for x in items]


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    eval_logger = logging.getLogger("locomo_eval_robust")
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in eval_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def _basic_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "count": int(len(values)),
    }


def _aggregate_compare(
    metrics_no: List[Dict[str, float]],
    metrics_attn: List[Dict[str, float]],
    categories: List[int],
) -> Dict[str, Dict]:
    all_keys = set()
    for m in metrics_no + metrics_attn:
        all_keys.update(m.keys())
    keys_sorted = sorted(all_keys)

    def _summarize(idxs: List[int]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for k in keys_sorted:
            va = [float(metrics_no[i].get(k, 0.0)) for i in idxs]
            vb = [float(metrics_attn[i].get(k, 0.0)) for i in idxs]
            a_stats = _basic_stats(va)
            b_stats = _basic_stats(vb)
            out[k] = {
                "no_attention": a_stats,
                "attention": b_stats,
                "delta_mean": float(b_stats["mean"] - a_stats["mean"]) if va and vb else 0.0,
            }
        return out

    overall = _summarize(list(range(len(metrics_no))))
    by_category: Dict[str, Dict] = {}
    for cat in sorted(set(categories)):
        idxs = [i for i, c in enumerate(categories) if int(c) == int(cat)]
        by_category[str(cat)] = _summarize(idxs)

    return {"overall": overall, "by_category": by_category}


def _vanilla_path(
    results_dir: Path,
    model: str,
    backend: str,
    dataset_path: str,
    sample_id: int,
    retrieve_k: int,
) -> Path:
    return results_dir / (
        f"Vanilla_{_safe_name(model)}_{_safe_name(backend)}_{_dataset_tag(dataset_path)}"
        f"_sample_{sample_id}_k_{int(retrieve_k)}.jsonl"
    )


def _attention_path(
    results_dir: Path,
    model: str,
    backend: str,
    dataset_path: str,
    sample_id: int,
    retrieve_k: int,
) -> Path:
    return results_dir / (
        f"Attention_{_safe_name(model)}_{_safe_name(backend)}_{_dataset_tag(dataset_path)}"
        f"_sample_{sample_id}_k_{int(retrieve_k)}.jsonl"
    )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue


def _done_qa_idx_set(path: Path) -> set:
    """Return qa_idx values already persisted in a JSONL file.

    Robust to partial lines (e.g. Ctrl+C during write): invalid JSON lines are ignored.
    """
    done = set()
    if not path.exists():
        return done
    for obj in _iter_jsonl(path):
        qa_idx = obj.get("qa_idx", None)
        try:
            if qa_idx is not None:
                done.add(int(qa_idx))
        except Exception:
            continue
    return done


def run_vanilla(
    *,
    dataset_path: str,
    model: str,
    backend: str,
    retrieve_k: int,
    temperature_c5: float,
    sglang_host: str,
    sglang_port: int,
    attention_model: str,
    attention_max_length: int,
    results_dir: Path,
    sample_ids: Optional[List[int]],
    max_questions_per_sample: Optional[int],
    max_questions_total: Optional[int],
    output_path: Optional[str],
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = _REPO_ROOT / "logs" / f"eval_vanilla_{_safe_name(model)}_{backend}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_logger = setup_logger(str(log_path))

    eval_logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    memories_dir = _REPO_ROOT / f"cached_memories_robust_{backend}_{_safe_name(model)}"
    memories_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    allow_categories = {1, 2, 3, 4, 5}
    total_questions = 0
    category_counts = defaultdict(int)
    all_metrics: List[Dict[str, float]] = []
    all_categories: List[int] = []
    individual_results: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(samples):
        sid = int(sample.sample_id)
        if sample_ids is not None and sid not in sample_ids:
            continue

        agent = RobustAdvancedMemAgent(
            model,
            backend,
            retrieve_k,
            temperature_c5,
            sglang_host,
            sglang_port,
            attention_model=attention_model,
            attention_max_length=attention_max_length,
        )

        memory_cache_file = memories_dir / f"memory_cache_sample_{sample_idx}.pkl"
        retriever_cache_file = memories_dir / f"retriever_cache_sample_{sample_idx}.pkl"
        retriever_cache_embeddings_file = memories_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"

        if memory_cache_file.exists():
            eval_logger.info(f"Loading cached memories for sample_idx={sample_idx} sample_id={sid}")
            with memory_cache_file.open("rb") as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if retriever_cache_file.exists():
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    str(retriever_cache_file), str(retriever_cache_embeddings_file)
                )
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, "all-MiniLM-L6-v2"
                )
        else:
            eval_logger.info(f"No cached memories for sample_idx={sample_idx} sample_id={sid} — building.")
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp, time=turns.date_time)
            with memory_cache_file.open("wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(str(retriever_cache_file), str(retriever_cache_embeddings_file))

        vanilla_path = _vanilla_path(results_dir, model, backend, dataset_path, sid, retrieve_k)
        done_qa = _done_qa_idx_set(vanilla_path)
        if done_qa:
            eval_logger.info(f"Vanilla cache exists, will resume. done={len(done_qa)} -> {vanilla_path}")
        else:
            eval_logger.info(f"Writing vanilla cache: {vanilla_path}")
        written = 0
        with vanilla_path.open("a", encoding="utf-8") as fp:
            for qa_idx, qa in enumerate(sample.qa):
                if max_questions_total is not None and total_questions >= max_questions_total:
                    break
                if max_questions_per_sample is not None and written >= max_questions_per_sample:
                    break
                if int(qa.category) not in allow_categories:
                    continue
                if qa_idx in done_qa:
                    continue

                total_questions += 1
                category_counts[int(qa.category)] += 1

                keywords = agent.generate_query_llm(qa.question)
                raw_context = agent.retrieve_memory_no_attention(keywords, k=agent.retrieve_k)
                pred, user_prompt, _ = agent.answer_question_with_context(
                    qa.question, int(qa.category), qa.final_answer, raw_context
                )
                tok = getattr(agent.memory_system.llm_controller.llm, "last_prompt_tokens", None)
                metrics = calculate_metrics(pred, qa.final_answer) if qa.final_answer else {"exact_match": 0, "f1": 0.0}

                row = {
                    "sample_idx": sample_idx,
                    "sample_id": sid,
                    "qa_idx": qa_idx,
                    "question": qa.question,
                    "reference": qa.final_answer,
                    "category": int(qa.category),
                    "keywords": keywords,
                    "retrieval_mode": "no_attention",
                    "raw_context": raw_context,
                    "user_prompt": user_prompt,
                    "prediction": pred,
                    "user_prompt_tokens": tok,
                    "metrics": metrics,
                }
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                fp.flush()
                written += 1

                all_metrics.append(metrics)
                all_categories.append(int(qa.category))
                individual_results.append(row)

        if max_questions_total is not None and total_questions >= max_questions_total:
            break

    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    final_results: Dict[str, Any] = {
        "mode": "vanilla",
        "model": model,
        "backend": backend,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {str(k): int(v) for k, v in category_counts.items()},
        "aggregate_metrics": aggregate_results,
        "individual_results": individual_results,
        "results_dir": str(results_dir),
    }

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(final_results, indent=2, ensure_ascii=False), encoding="utf-8")
        eval_logger.info(f"Summary saved to {outp}")

    return final_results


def run_attention_compare(
    *,
    dataset_path: str,
    model: str,
    backend: str,
    retrieve_k: int,
    temperature_c5: float,
    sglang_host: str,
    sglang_port: int,
    attention_model: str,
    attention_max_length: int,
    results_dir: Path,
    sample_ids: Optional[List[int]],
    max_questions_per_sample: Optional[int],
    max_questions_total: Optional[int],
    output_path: Optional[str],
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = _REPO_ROOT / "logs" / f"eval_attention_{_safe_name(model)}_{backend}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_logger = setup_logger(str(log_path))

    eval_logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    memories_dir = _REPO_ROOT / f"cached_memories_robust_{backend}_{_safe_name(model)}"
    memories_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    allow_categories = {1, 2, 3, 4, 5}
    total_questions = 0
    category_counts = defaultdict(int)

    cmp_prompt_len = {"no_attention": [], "attention": []}
    cmp_metrics = {"no_attention": [], "attention": []}
    cmp_categories: List[int] = []

    individual_results: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(samples):
        sid = int(sample.sample_id)
        if sample_ids is not None and sid not in sample_ids:
            continue

        vanilla_path = _vanilla_path(results_dir, model, backend, dataset_path, sid, retrieve_k)
        if not vanilla_path.exists():
            eval_logger.warning(f"Skip sample_id={sid}: vanilla cache not found at {vanilla_path}")
            continue

        vanilla_rows = [r for r in _iter_jsonl(vanilla_path) if int(r.get("sample_id", -1)) == sid]
        vanilla_rows = [r for r in vanilla_rows if int(r.get("category", -1)) in allow_categories]
        if max_questions_per_sample is not None:
            vanilla_rows = vanilla_rows[: max_questions_per_sample]

        if not vanilla_rows:
            eval_logger.warning(f"Skip sample_id={sid}: vanilla cache empty")
            continue

        agent = RobustAdvancedMemAgent(
            model,
            backend,
            retrieve_k,
            temperature_c5,
            sglang_host,
            sglang_port,
            attention_model=attention_model,
            attention_max_length=attention_max_length,
        )

        memory_cache_file = memories_dir / f"memory_cache_sample_{sample_idx}.pkl"
        retriever_cache_file = memories_dir / f"retriever_cache_sample_{sample_idx}.pkl"
        retriever_cache_embeddings_file = memories_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"

        if memory_cache_file.exists():
            with memory_cache_file.open("rb") as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if retriever_cache_file.exists():
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    str(retriever_cache_file), str(retriever_cache_embeddings_file)
                )
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, "all-MiniLM-L6-v2"
                )
        else:
            eval_logger.info(f"No cached memories for sample_idx={sample_idx} sample_id={sid} — building.")
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp, time=turns.date_time)
            with memory_cache_file.open("wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(str(retriever_cache_file), str(retriever_cache_embeddings_file))

        eval_logger.info(f"Processing sample_id={sid} with {len(vanilla_rows)} vanilla rows")
        attention_path = _attention_path(results_dir, model, backend, dataset_path, sid, retrieve_k)
        done_attn = _done_qa_idx_set(attention_path)
        if done_attn:
            eval_logger.info(f"Attention cache exists, will resume. done={len(done_attn)} -> {attention_path}")
        else:
            eval_logger.info(f"Writing attention cache: {attention_path}")
        with attention_path.open("a", encoding="utf-8") as attn_fp:
            for r in vanilla_rows:
                if max_questions_total is not None and total_questions >= max_questions_total:
                    break

                q = r.get("question", "")
                ref = r.get("reference", "")
                cat = int(r.get("category", 0))
                qa_idx = r.get("qa_idx", None)
                try:
                    qa_idx_int = int(qa_idx) if qa_idx is not None else None
                except Exception:
                    qa_idx_int = None
                if not q or cat not in allow_categories:
                    continue
                if qa_idx_int is not None and qa_idx_int in done_attn:
                    continue

                # Use the same keywords from vanilla to keep compare aligned.
                keywords = r.get("keywords", "")
                if not keywords:
                    keywords = agent.generate_query_llm(q)

                ctx_attn = agent.retrieve_memory(keywords, k=agent.retrieve_k)
                pred_attn, prompt_attn, _ = agent.answer_question_with_context(q, cat, ref, ctx_attn)
                tok_attn = getattr(agent.memory_system.llm_controller.llm, "last_prompt_tokens", None)
                metrics_attn = calculate_metrics(pred_attn, ref) if ref else {"exact_match": 0, "f1": 0.0}

                # Baseline fields from vanilla (for comparison aggregation)
                pred_no = r.get("prediction", "")
                tok_no = r.get("user_prompt_tokens", None)
                metrics_no = r.get("metrics", {"exact_match": 0, "f1": 0.0})

                total_questions += 1
                category_counts[cat] += 1

                if tok_no is not None:
                    cmp_prompt_len["no_attention"].append(float(tok_no))
                if tok_attn is not None:
                    cmp_prompt_len["attention"].append(float(tok_attn))
                if isinstance(metrics_no, dict) and isinstance(metrics_attn, dict):
                    cmp_metrics["no_attention"].append(metrics_no)
                    cmp_metrics["attention"].append(metrics_attn)
                    cmp_categories.append(cat)

                # Persist attention per-question cache (Vanilla-aligned)
                attn_row = {
                    "sample_id": sid,
                    "qa_idx": qa_idx_int,
                    "question": q,
                    "reference": ref,
                    "category": cat,
                    "keywords": keywords,
                    "retrieval_mode": "attention",
                    "raw_context": ctx_attn,
                    "user_prompt": prompt_attn,
                    "prediction": pred_attn,
                    "user_prompt_tokens": tok_attn,
                    "metrics": metrics_attn,
                }
                attn_fp.write(json.dumps(attn_row, ensure_ascii=False) + "\n")
                attn_fp.flush()

                # Keep detailed compare record in the summary JSON
                individual_results.append(
                    {
                        "sample_id": sid,
                        "qa_idx": r.get("qa_idx", None),
                        "question": q,
                        "reference": ref,
                        "category": cat,
                        "keywords": keywords,
                        "no_attention": {
                            "prediction": pred_no,
                            "user_prompt_tokens": tok_no,
                            "metrics": metrics_no,
                        },
                        "attention": {
                            "prediction": pred_attn,
                            "user_prompt_tokens": tok_attn,
                            "metrics": metrics_attn,
                            "raw_context": ctx_attn,
                            "user_prompt": prompt_attn,
                        },
                    }
                )

        if max_questions_total is not None and total_questions >= max_questions_total:
            break

    compare_summary = {
        "user_prompt_tokens": {
            "overall": {
                "no_attention": _basic_stats(cmp_prompt_len["no_attention"]),
                "attention": _basic_stats(cmp_prompt_len["attention"]),
            }
        },
        "metrics": _aggregate_compare(
            cmp_metrics["no_attention"],
            cmp_metrics["attention"],
            cmp_categories,
        )
        if cmp_categories
        else {"overall": {}, "by_category": {}},
    }

    final_results: Dict[str, Any] = {
        "mode": "attention_compare",
        "model": model,
        "backend": backend,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {str(k): int(v) for k, v in category_counts.items()},
        "compare_attention": compare_summary,
        "individual_results": individual_results,
        "results_dir": str(results_dir),
    }

    if output_path is None:
        output_path = str(_REPO_ROOT / "logs" / f"compare_{_safe_name(model)}_{backend}_{timestamp}.json")
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(final_results, indent=2, ensure_ascii=False), encoding="utf-8")
    eval_logger.info(f"Compare summary saved to {outp}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Robust LoComo evaluation (vanilla + attention compare)")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--attention_model", type=str, default="/data/hyc/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--attention_max_length", type=int, default=4096)

    # New controls
    parser.add_argument("--mode", type=str, choices=["vanilla", "attention"], default="vanilla",
                        help="vanilla: run no-attention and save results; attention: run attention and compare vs saved vanilla")
    parser.add_argument("--sample_ids", type=str, default=None,
                        help="Comma-separated sample_id list (from dataset). Example: 0,1,2")
    parser.add_argument("--max_questions_per_sample", type=int, default=None,
                        help="Max QA rows per sample (after category filter). Default: all available (vanilla) / all cached rows (attention)")
    parser.add_argument("--max_questions_total", type=int, default=None,
                        help="Global max questions across all samples (after filtering/alignment). Default: no limit")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to store vanilla caches (JSONL). Default: results/")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to save a JSON summary. For attention mode, default is logs/compare_*.json")

    args = parser.parse_args()

    dataset_path = str((_REPO_ROOT / args.dataset).resolve()) if not os.path.isabs(args.dataset) else args.dataset
    results_dir = (_REPO_ROOT / args.results_dir).resolve()
    sample_ids = _parse_int_list(args.sample_ids)

    if args.mode == "vanilla":
        run_vanilla(
            dataset_path=dataset_path,
            model=args.model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
            attention_model=args.attention_model,
            attention_max_length=args.attention_max_length,
            results_dir=results_dir,
            sample_ids=sample_ids,
            max_questions_per_sample=args.max_questions_per_sample,
            max_questions_total=args.max_questions_total,
            output_path=str((_REPO_ROOT / args.output).resolve()) if args.output else None,
        )
    else:
        run_attention_compare(
            dataset_path=dataset_path,
            model=args.model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
            attention_model=args.attention_model,
            attention_max_length=args.attention_max_length,
            results_dir=results_dir,
            sample_ids=sample_ids,
            max_questions_per_sample=args.max_questions_per_sample,
            max_questions_total=args.max_questions_total,
            output_path=str((_REPO_ROOT / args.output).resolve()) if args.output else None,
        )


if __name__ == "__main__":
    main()

