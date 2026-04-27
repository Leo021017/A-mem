"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import json
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime

# Download required NLTK data
# Use repo-local nltk_data to avoid network downloads/hangs
_REPO_NLTK_DATA = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.insert(0, _REPO_NLTK_DATA)
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/wordnet")
except LookupError as e:
    raise RuntimeError(
        f"未在项目路径找到 NLTK 资源：{_REPO_NLTK_DATA}（缺 punkt/wordnet）。"
    ) from e

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(
        self,
        model,
        backend,
        retrieve_k,
        temperature_c5,
        sglang_host="http://localhost",
        sglang_port=30000,
        attention_model: str = "/data/hyc/models/Qwen2.5-14B-Instruct",
        attention_max_length: int = 4096,
    ):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.attention_model = attention_model
        self.attention_max_length = attention_max_length

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(
            content,
            k=k,
            use_attention=True,
            attention_model_name=self.attention_model,
            attention_max_length=self.attention_max_length,
        )

    def retrieve_memory_no_attention(self, content, k=10):
        return self.memory_system.find_related_memories_raw(
            content,
            k=k,
            use_attention=False,
            attention_model_name=self.attention_model,
            attention_max_length=self.attention_max_length,
        )

    @staticmethod
    def _category5_options(question: str, answer: str) -> list:
        """Stable options order for fair comparisons."""
        import hashlib
        h = hashlib.md5(question.encode("utf-8")).digest()[0]
        if h % 2 == 0:
            return ["Not mentioned in the conversation", answer]
        return [answer, "Not mentioned in the conversation"]

    def build_user_prompt(self, question: str, category: int, answer: str, context: str) -> tuple:
        """Build user_prompt and temperature given a context."""
        assert category in [1, 2, 3, 4, 5]
        if category == 5:
            answer_tmp = self._category5_options(question, answer)
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        return user_prompt, temperature

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        user_prompt, temperature = self.build_user_prompt(question, category, answer, raw_context)

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context

    def answer_question_with_context(self, question: str, category: int, answer: str, raw_context: str) -> tuple:
        """Answer with provided context (used for attention/no-attention comparisons)."""
        user_prompt, temperature = self.build_user_prompt(question, category, answer, raw_context)
        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question_with_context failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     compare_attention: bool = False,
                     attention_model: str = "/data/hyc/models/Qwen2.5-14B-Instruct",
                     attention_max_length: int = 4096):
    """Evaluate the robust agent on the LoComo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    results = []
    all_metrics = []
    all_categories = []
    cmp_prompt_len = {"no_attention": [], "attention": []}  # prompt_tokens
    cmp_metrics = {"no_attention": [], "attention": []}
    cmp_categories: List[int] = []
    total_questions = 0
    category_counts = defaultdict(int)

    i = 0
    error_num = 0
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(backend, model),
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]

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
        """Aggregate all metric keys: overall + by_category, no_attention vs attention."""
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

    for sample_idx, sample in enumerate(samples):
        if sample_idx > 0:
            break
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port,
                                       attention_model=attention_model,
                                       attention_max_length=attention_max_length)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                eval_logger.info(f"Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(f"No retriever cache found, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, 'all-MiniLM-L6-v2'
                )
            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")

            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp, time=turn_datatime)

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")

        num_runs = 0
        for qa in sample.qa:
            if num_runs == 3:
                break
            num_runs += 1
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1
                eval_logger.info(f"Question {total_questions}: {qa.question}")

                if compare_attention:
                    keywords = agent.generate_query_llm(qa.question)
                    ctx_no = agent.retrieve_memory_no_attention(keywords, k=agent.retrieve_k)
                    ctx_attn = agent.retrieve_memory(keywords, k=agent.retrieve_k)

                    pred_no, prompt_no, _ = agent.answer_question_with_context(
                        qa.question, qa.category, qa.final_answer, ctx_no
                    )
                    tok_no = getattr(agent.memory_system.llm_controller.llm, "last_prompt_tokens", None)
                    pred_attn, prompt_attn, _ = agent.answer_question_with_context(
                        qa.question, qa.category, qa.final_answer, ctx_attn
                    )
                    tok_attn = getattr(agent.memory_system.llm_controller.llm, "last_prompt_tokens", None)
                    pred_no = parse_plain_text_answer(pred_no)
                    pred_attn = parse_plain_text_answer(pred_attn)

                    metrics_no = calculate_metrics(pred_no, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0, "f1": 0.0
                    }
                    metrics_attn = calculate_metrics(pred_attn, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0, "f1": 0.0
                    }

                    # Prompt length: use API-reported prompt_tokens (token count), not char length.
                    if tok_no is None or tok_attn is None:
                        raise RuntimeError(
                            "未拿到 API usage.prompt_tokens，无法按 token 统计 prompt 长度。"
                            "请确认 backend=openai 且 OpenAI-compatible 返回 usage 字段。"
                        )
                    cmp_prompt_len["no_attention"].append(float(tok_no))
                    cmp_prompt_len["attention"].append(float(tok_attn))
                    cmp_metrics["no_attention"].append(metrics_no)
                    cmp_metrics["attention"].append(metrics_attn)
                    cmp_categories.append(int(qa.category))

                    # Keep "attention" run as default aggregate (backward compatible)
                    all_metrics.append(metrics_attn)
                    all_categories.append(qa.category)

                    results.append({
                        "question_id": f"{sample.sample_id}:{total_questions}",
                        "sample_id": sample_idx,
                        "question": qa.question,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "no_attention": {
                            "prediction": pred_no,
                            "user_prompt_tokens": tok_no,
                            "metrics": metrics_no,
                        },
                        "attention": {
                            "prediction": pred_attn,
                            "user_prompt_tokens": tok_attn,
                            "metrics": metrics_attn,
                        },
                    })

                    # "最终正确答案对比"：逐题输出 reference / 两种 prediction（不算准确率）
                    eval_logger.info(f"Reference: {qa.final_answer}")
                    eval_logger.info(f"NoAttn pred: {pred_no} | prompt_tokens={tok_no} | f1={metrics_no.get('f1')}")
                    eval_logger.info(f"Attn   pred: {pred_attn} | prompt_tokens={tok_attn} | f1={metrics_attn.get('f1')}")
                else:
                    prediction, user_prompt, raw_context = agent.answer_question(
                        qa.question, qa.category, qa.final_answer
                    )
                    prediction = parse_plain_text_answer(prediction)

                    eval_logger.info(f"Prediction: {prediction}")
                    eval_logger.info(f"Reference: {qa.final_answer}")
                    eval_logger.info(f"User Prompt: {user_prompt}")
                    eval_logger.info(f"Category: {qa.category}")
                    eval_logger.info(f"Raw Context: {raw_context}")

                    metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                        "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                        "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                        "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                    }
                    all_metrics.append(metrics)
                    all_categories.append(qa.category)
                    results.append({
                        "sample_id": sample_idx,
                        "question": qa.question,
                        "prediction": prediction,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "metrics": metrics,
                    })

                if total_questions % 10 == 0:
                    eval_logger.info(f"Processed {total_questions} questions")

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }

    if compare_attention:
        # Prompt tokens: overall + per-category
        prompt_tokens_summary = {
            "overall": {
                "no_attention": _basic_stats(cmp_prompt_len["no_attention"]),
                "attention": _basic_stats(cmp_prompt_len["attention"]),
            },
            "by_category": {},
        }
        for cat in sorted(set(cmp_categories)):
            idxs = [i for i, c in enumerate(cmp_categories) if int(c) == int(cat)]
            prompt_tokens_summary["by_category"][str(cat)] = {
                "no_attention": _basic_stats([cmp_prompt_len["no_attention"][i] for i in idxs]),
                "attention": _basic_stats([cmp_prompt_len["attention"][i] for i in idxs]),
            }

        compare_summary = {
            "user_prompt_tokens": prompt_tokens_summary,
            "metrics": _aggregate_compare(
                cmp_metrics["no_attention"],
                cmp_metrics["attention"],
                cmp_categories,
            ),
        }
        final_results["compare_attention"] = compare_summary
        eval_logger.info("=== Attention vs No-Attention Comparison ===")
        eval_logger.info(f"Prompt tokens overall: {compare_summary['user_prompt_tokens']['overall']}")
        # Print headline metrics if present
        if "f1" in compare_summary["metrics"]["overall"]:
            eval_logger.info(f"F1 overall: {compare_summary['metrics']['overall']['f1']}")
        if "exact_match" in compare_summary["metrics"]["overall"]:
            eval_logger.info(f"Exact_match overall: {compare_summary['metrics']['overall']['exact_match']}")
    eval_logger.info(f"Error number: {error_num}")

    # Always persist compare results to JSON (even if --output not provided).
    if compare_attention and not output_path:
        output_path = os.path.join(
            os.path.dirname(__file__),
            "logs",
            f"compare_{model}_{backend}_ratio{ratio}_{timestamp}.json",
        )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--compare_attention", action="store_true",
                        help="Compare retrieval with/without attention (double API calls)")
    parser.add_argument("--attention_model", type=str, default="/data/hyc/models/Qwen2.5-14B-Instruct",
                        help="Local HF model path/name for attention-based filtering")
    parser.add_argument("--attention_max_length", type=int, default=4096,
                        help="Max length for attention model input (context+query)")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port,
        compare_attention=args.compare_attention,
        attention_model=args.attention_model,
        attention_max_length=args.attention_max_length,
    )


if __name__ == "__main__":
    main()
