from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


_WS_RE = re.compile(r"\s+")

_HF_CACHE: Dict[str, Tuple[object, object, torch.device]] = {}


def _get_tokenizer_and_model(model_name: str) -> tuple:
    """
    Load once, reuse across calls.
    Also force eager attention so output_attentions=True works (sdpa/flash won't).
    """
    if model_name in _HF_CACHE:
        tok, mdl, dev = _HF_CACHE[model_name]
        return tok, mdl, dev

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    # Force eager attention implementation if supported.
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(cfg, "attn_implementation"):
        try:
            cfg.attn_implementation = "eager"
        except Exception:
            pass

    model_kwargs = {"trust_remote_code": True}
    # Newer transformers accepts attn_implementation at load time.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=cfg,
            attn_implementation="eager",
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=cfg,
            **model_kwargs,
        )

    model.eval()
    model.to(device)

    _HF_CACHE[model_name] = (tokenizer, model, device)
    return tokenizer, model, device


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


def _transformer_attention_select(
    query: str,
    raw_context: str,
    topp: float,
    model_name: str,
    max_length: int,
) -> Optional[str]:

    if topp <= 0:
        return ""
    if topp >= 1:
        return raw_context

    tokenizer, model, device = _get_tokenizer_and_model(model_name)

    # 单独 tokenize query / context，确保 context 的 offset_mapping 是相对 raw_context 的
    q_enc = tokenizer(
        query,
        add_special_tokens=False,
        return_tensors=None,
        truncation=True,
        max_length=max_length,
    )
    c_enc = tokenizer(
        raw_context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
        truncation=True,
        max_length=max_length,
    )
    q_ids: List[int] = list(q_enc["input_ids"])
    c_ids: List[int] = list(c_enc["input_ids"])
    c_offsets: List[List[int]] = [list(x) for x in c_enc["offset_mapping"]]

    # 连接符：尽量减少对 token 边界的干扰，用纯换行
    sep_ids: List[int] = list(tokenizer("\n\n", add_special_tokens=False)["input_ids"])

    # 组装最终输入（保留 BOS 如果 tokenizer 有）：context + query
    input_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(int(tokenizer.bos_token_id))
    ctx_start = len(input_ids)
    input_ids.extend(c_ids)
    ctx_end = len(input_ids)  # exclusive
    input_ids.extend(sep_ids)
    q_start = len(input_ids)
    input_ids.extend(q_ids)
    q_end = len(input_ids)  # exclusive

    # 截断到 max_length：保留 query，裁剪最前面的 context tokens
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        cut_from_ctx = min(overflow, max(0, len(c_ids) - 1))
        if cut_from_ctx > 0:
            c_ids = c_ids[cut_from_ctx:]
            c_offsets = c_offsets[cut_from_ctx:]
            # 重新拼接（ctx_start 不变，ctx_end/q_start 需要重算）
            input_ids = input_ids[:ctx_start] + c_ids
            ctx_end = len(input_ids)
            input_ids.extend(sep_ids)
            q_start = len(input_ids)
            input_ids.extend(q_ids)
            q_end = len(input_ids)

    ctx_len = len(c_ids)
    if ctx_len != len(c_offsets):
        raise RuntimeError("context token 与 offset_mapping 长度不一致，无法映射回原文。")
    if ctx_len <= 0 or q_end <= q_start:
        raise RuntimeError("query/context token 切分失败（可能输入为空或被截断）。")

    enc = {"input_ids": torch.tensor([input_ids], device=device)}
    if tokenizer.pad_token_id is not None:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)

    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attentions = out.attentions  # tuple[num_layers] of (batch, heads, seq, seq)
    if not attentions:
        raise RuntimeError(
            "模型未返回 attentions。通常是 transformers 使用了 sdpa/flash attention。"
            "请确认已强制 attn_implementation='eager'，或使用支持 output_attentions 的注意力实现。"
        )

    query_idx = list(range(q_start, q_end))
    ctx_idx = list(range(ctx_start, min(ctx_start + ctx_len, ctx_end)))
    if not query_idx or not ctx_idx:
        raise RuntimeError("query/context token 切分失败（可能是输入被截断或 tokenizer 行为异常）。")

    # 计算每个 context token 的累积注意力分数
    # decoder-only: score[s] = sum_layers mean_heads sum_{t in query} attn[t, s]
    seq_len = len(input_ids)
    scores = torch.zeros(seq_len, dtype=torch.float32)
    for layer_attn in attentions:
        # (heads, seq, seq)
        a = layer_attn[0]  # batch 0
        a = a.mean(dim=0)  # head 之间取平均，得到 (seq, seq)
        # 只给 context token 累积分数：被 query 关注越多，分数越高
        scores += a[query_idx, :].sum(dim=0).detach().cpu()

    ctx_scores = [(i, float(scores[i].item())) for i in ctx_idx]
    ctx_scores.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(1, int(len(ctx_scores) * topp))
    keep_set = {i for i, _ in ctx_scores[:keep_n]}

    # 将保留的 token 映射回 raw_context 的字符 span
    spans: List[_Span] = []
    for pos in sorted(keep_set):
        c_i = pos - ctx_start
        if c_i < 0 or c_i >= len(c_offsets):
            continue
        start, end = c_offsets[c_i]
        if start == 0 and end == 0:
            continue
        spans.append(_Span(int(start), int(end)))

    if not spans:
        raise RuntimeError("offset 映射为空，无法把 token 映射回 raw_context 文本片段。")

    # 不做片段合并：按 token 顺序逐个取子串并拼接
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    parts = [raw_context[s.start:s.end] for s in spans]
    text = " ".join(p.strip() for p in parts if p.strip())
    return _WS_RE.sub(" ", text).strip()


def retrieve_attention(
    query: str,
    raw_context: str,
    topp: float = 0.5,
    model_name: str = "/data/hyc/models/Qwen2.5-14B-Instruct",
    max_length: int = 20000,
) -> str:
    """
    计算 raw_context 对 query 的“累积注意力分数”，选取 topk=topp 的 token，
    仅返回这些 token 对应的原文片段。

    - **topp**: 0~1 之间，表示保留的 context token 比例（默认 0.3）。
    - **model_name**: 用于 attention 的本地 transformer。
    - **max_length**: pair 编码最大长度（包含 query+context+special tokens）。
    """
    text = _transformer_attention_select(
        query=query,
        raw_context=raw_context,
        topp=topp,
        model_name=model_name,
        max_length=max_length,
    )
    return text or ""
