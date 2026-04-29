#!/usr/bin/env python3
"""
Parse vanilla result JSONL and generate aggregate analysis.
python test/parse_vanilla_result.py \
  --result_jsonl results/Vanilla_deepseek-v4-flash_openai_locomo10_sample_0_k_10.jsonl \
  --dataset data/locomo10.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Allow running from repo root or from test/ directory.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from load_dataset import LoCoMoSample, load_locomo_dataset


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "count": int(len(values)),
    }


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    overall_values: Dict[str, List[float]] = defaultdict(list)
    category_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        cat = int(row.get("category", -1))
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            if not _is_number(metric_value):
                continue
            v = float(metric_value)
            overall_values[metric_name].append(v)
            category_values[cat][metric_name].append(v)

    overall_stats = {k: _stats(vs) for k, vs in sorted(overall_values.items())}
    by_category_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cat, metric_map in sorted(category_values.items(), key=lambda x: x[0]):
        by_category_stats[str(cat)] = {k: _stats(vs) for k, vs in sorted(metric_map.items())}

    return overall_stats, by_category_stats


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _build_dia_text_map(sample: LoCoMoSample) -> Dict[str, str]:
    dia_to_text: Dict[str, str] = {}
    for session in sample.conversation.sessions.values():
        for turn in session.turns:
            dia_to_text[turn.dia_id] = turn.text
    return dia_to_text


def _extract_k_from_name(path: Path) -> Optional[int]:
    m = re.search(r"_k_(\d+)(?:\.[^.]+)?$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"_k_(\d+)", path.name)
    if m:
        return int(m.group(1))
    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse vanilla result JSONL and output aggregate analysis JSON.")
    parser.add_argument("--result_jsonl", type=str, required=True, help="Path to vanilla result JSONL.")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to LoCoMo dataset json.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    parser.add_argument("--retrieve_k", type=int, default=None, help="Optional retrieve_k override in output metadata.")
    args = parser.parse_args()

    result_path = Path(args.result_jsonl).resolve()
    dataset_path = Path(args.dataset).resolve()

    rows = list(_iter_jsonl(result_path))
    if not rows:
        raise ValueError(f"No valid JSON rows found in {result_path}")

    samples = load_locomo_dataset(dataset_path)
    sample_map: Dict[int, LoCoMoSample] = {int(s.sample_id): s for s in samples}

    total_questions_in_result = len(rows)
    sample_ids = sorted({int(r.get("sample_id", -1)) for r in rows if "sample_id" in r})
    valid_sample_ids = [sid for sid in sample_ids if sid in sample_map]

    total_questions_in_samples = sum(len(sample_map[sid].qa) for sid in valid_sample_ids)

    overall_metrics, category_metric_stats = _aggregate_metrics(rows)

    category_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        category_counts[str(int(r.get("category", -1)))] += 1

    category_counts_in_sample: Dict[str, int] = defaultdict(int)
    for sid in valid_sample_ids:
        for qa in sample_map[sid].qa:
            category_counts_in_sample[str(int(qa.category))] += 1

    category_summary: Dict[str, Dict[str, Any]] = {}
    for category in sorted(set(list(category_counts.keys()) + list(category_counts_in_sample.keys())), key=lambda x: int(x)):
        count_result = int(category_counts.get(category, 0))
        count_sample = int(category_counts_in_sample.get(category, 0))
        category_summary[category] = {
            "question_count": count_result,
            "ratio_in_result_questions": _safe_ratio(count_result, total_questions_in_result),
            "ratio_in_sample_total_questions": _safe_ratio(count_result, total_questions_in_samples),
            "sample_question_count": count_sample,
            "category_coverage_in_sample": _safe_ratio(count_result, count_sample),
            "metrics": category_metric_stats.get(category, {}),
        }

    evidence_question_hits = 0
    evidence_question_total_hits = 0
    evidence_questions_total = 0
    evidence_turn_hits = 0
    evidence_turn_total = 0
    unresolved_evidence_refs = 0
    evidence_by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    dia_cache: Dict[int, Dict[str, str]] = {}
    for row in rows:
        sid = int(row.get("sample_id", -1))
        qa_idx = int(row.get("qa_idx", -1))
        cat = str(int(row.get("category", -1)))
        raw_context = str(row.get("raw_context", ""))

        if sid not in sample_map:
            continue
        sample = sample_map[sid]
        if qa_idx < 0 or qa_idx >= len(sample.qa):
            continue

        qa = sample.qa[qa_idx]
        evidence_ids = list(qa.evidence or [])
        evidence_questions_total += 1
        evidence_by_category[cat]["questions_total"] += 1

        if sid not in dia_cache:
            dia_cache[sid] = _build_dia_text_map(sample)
        dia_map = dia_cache[sid]

        ctx_norm = _normalize_text(raw_context)
        this_question_hit = False
        this_question_turn_hit_count = 0

        for ev in evidence_ids:
            dia_id = str(ev)
            evidence_turn_total += 1
            evidence_by_category[cat]["evidence_turns_total"] += 1

            ev_text = dia_map.get(dia_id)
            if ev_text is None and ":" in dia_id:
                ev_text = dia_map.get(dia_id.split(":")[-1], None)

            if not ev_text:
                unresolved_evidence_refs += 1
                evidence_by_category[cat]["evidence_unresolved"] += 1
                continue

            ev_norm = _normalize_text(ev_text)
            if ev_norm and ev_norm in ctx_norm:
                evidence_turn_hits += 1
                evidence_by_category[cat]["evidence_turns_hit"] += 1
                this_question_hit = True
                this_question_turn_hit_count += 1

        if this_question_hit:
            evidence_question_hits += 1
            evidence_by_category[cat]["questions_hit"] += 1
        if evidence_ids and this_question_turn_hit_count == len(evidence_ids):
            evidence_question_total_hits += 1
            evidence_by_category[cat]["questions_total_hit"] += 1

    evidence_by_category_summary: Dict[str, Dict[str, Any]] = {}
    for cat, stats in sorted(evidence_by_category.items(), key=lambda x: int(x[0])):
        q_total = int(stats.get("questions_total", 0))
        q_hit = int(stats.get("questions_hit", 0))
        e_total = int(stats.get("evidence_turns_total", 0))
        e_hit = int(stats.get("evidence_turns_hit", 0))
        e_unresolved = int(stats.get("evidence_unresolved", 0))
        evidence_by_category_summary[cat] = {
            "question_level": {
                "hit_questions": q_hit,
                "total_questions": q_total,
                "hit_rate": _safe_ratio(q_hit, q_total),
            },
            "question_level_total": {
                "hit_questions": int(stats.get("questions_total_hit", 0)),
                "total_questions": q_total,
                "hit_rate": _safe_ratio(int(stats.get("questions_total_hit", 0)), q_total),
            },
            "evidence_turn_level": {
                "hit_turns": e_hit,
                "total_turns": e_total,
                "hit_rate": _safe_ratio(e_hit, e_total),
                "unresolved_turn_refs": e_unresolved,
            },
        }

    retrieve_k = args.retrieve_k if args.retrieve_k is not None else _extract_k_from_name(result_path)

    analysis = {
        "metadata": {
            "result_jsonl": str(result_path),
            "dataset": str(dataset_path),
            "retrieve_k": retrieve_k,
            "sample_ids": valid_sample_ids,
            "total_questions_in_result": total_questions_in_result,
            "total_questions_in_samples": total_questions_in_samples,
        },
        "overall_metrics": overall_metrics,
        "category_metrics": category_summary,
        "evidence_coverage": {
            "question_level": {
                "hit_questions": evidence_question_hits,
                "total_questions": evidence_questions_total,
                "hit_rate": _safe_ratio(evidence_question_hits, evidence_questions_total),
            },
            "question_level_total": {
                "hit_questions": evidence_question_total_hits,
                "total_questions": evidence_questions_total,
                "hit_rate": _safe_ratio(evidence_question_total_hits, evidence_questions_total),
            },
            "evidence_turn_level": {
                "hit_turns": evidence_turn_hits,
                "total_turns": evidence_turn_total,
                "hit_rate": _safe_ratio(evidence_turn_hits, evidence_turn_total),
                "unresolved_turn_refs": unresolved_evidence_refs,
            },
            "by_category": evidence_by_category_summary,
        },
    }

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        k_suffix = f"_k_{retrieve_k}" if retrieve_k is not None else ""
        output_path = result_path.with_name(f"{result_path.stem}_analysis{k_suffix}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
