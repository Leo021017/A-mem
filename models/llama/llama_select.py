# coding=utf-8
from typing import Callable, Optional, Union

import os
import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from .llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


def selective_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    selective_query_indices: Optional[torch.LongTensor] = None,
    collect_attention_scores: bool = False,
    attention_score_buffer: Optional[torch.Tensor] = None,
    context_length: Optional[int] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    """Compute attention only on selected query rows and scatter back."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    full_q_len = query.shape[-2]
    if selective_query_indices is None:
        selective_query_indices = torch.arange(
            full_q_len, device=query.device, dtype=torch.long
        )
    else:
        selective_query_indices = selective_query_indices.to(
            device=query.device, dtype=torch.long
        )

    query_sel = query.index_select(dim=2, index=selective_query_indices)
    attn_weights = torch.matmul(query_sel, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[
            :, :, selective_query_indices, : key_states.shape[-2]
        ]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    # (b, heads, q_selected, d)
    attn_output_sel = torch.matmul(attn_weights, value_states)

    # Scatter selected query outputs back to full query length.
    # Unselected rows keep zero attention output (residual path still preserves input).
    attn_output_full = torch.zeros(
        query.shape[0],
        query.shape[1],
        full_q_len,
        query.shape[-1],
        device=query.device,
        dtype=query.dtype,
    )
    attn_output_full[:, :, selective_query_indices, :] = attn_output_sel
    attn_output_full = attn_output_full.transpose(1, 2).contiguous()

    if collect_attention_scores and attention_score_buffer is not None:
        # Aggregate: mean over heads, sum over selected query rows.
        layer_scores = attn_weights.mean(dim=(0, 1)).sum(dim=0).detach()
        limit = layer_scores.shape[0]
        if context_length is not None:
            limit = min(limit, int(context_length))
        attention_score_buffer[:limit] += layer_scores[:limit].to(
            device=attention_score_buffer.device, dtype=attention_score_buffer.dtype
        )

    # Return selected-query attention weights to keep memory usage low.
    return attn_output_full, attn_weights


class LlamaSelectAttention(LlamaAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        selective_enabled = bool(kwargs.pop("selective_attention", False))
        if selective_enabled and self.config._attn_implementation != "eager":
            raise ValueError(
                "selective_attention 仅支持 eager 注意力实现，请设置 _attn_implementation='eager'."
            )

        attention_interface: Callable = selective_eager_attention_forward
        if not selective_enabled:
            attention_interface = (
                ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                if self.config._attn_implementation != "eager"
                else selective_eager_attention_forward
            )
            if attention_interface is not selective_eager_attention_forward:
                # 非 selective 模式走原生后端，移除 selective 私有参数避免不兼容
                kwargs.pop("selective_query_indices", None)
                kwargs.pop("collect_attention_scores", None)
                kwargs.pop("attention_score_buffer", None)
                kwargs.pop("context_length", None)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaSelectDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = LlamaSelectAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


@auto_docstring
class LlamaSelectModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaSelectDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

        self.pp_enabled = False
        self.layer_gpu_list = []
        self.devices = []

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        selective_attention = bool(kwargs.pop("selective_attention", False))
        selective_query_start = kwargs.pop("selective_query_start", None)
        selective_query_indices = kwargs.pop("selective_query_indices", None)
        collect_attention_scores = bool(kwargs.pop("collect_attention_scores", False))
        attention_score_buffer = kwargs.pop("attention_score_buffer", None)
        context_length = kwargs.pop("context_length", None)

        if self.pp_enabled:
            self.norm = self.norm.to(self.devices[0])
            self.embed_tokens = self.embed_tokens.to(self.devices[0])
            self.rotary_emb = self.rotary_emb.to(self.devices[0])

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if selective_attention:
            if selective_query_indices is None:
                if selective_query_start is None:
                    raise ValueError(
                        "启用 selective_attention 时，需提供 selective_query_start 或 selective_query_indices."
                    )
                seq_len = inputs_embeds.shape[1]
                selective_query_indices = torch.arange(
                    int(selective_query_start),
                    seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
            else:
                selective_query_indices = selective_query_indices.to(
                    device=inputs_embeds.device, dtype=torch.long
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                selective_attention=selective_attention,
                selective_query_indices=selective_query_indices,
                collect_attention_scores=collect_attention_scores,
                attention_score_buffer=attention_score_buffer,
                context_length=context_length,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaSelectForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.pp_enabled = os.environ.get("PP_ENABLED", "0")
        self.pp_enabled = False if self.pp_enabled == "0" else True

        self.model = LlamaSelectModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.model.pp_enabled = self.pp_enabled
        self.gpu_list = []
        self.num_gpus = 0
        self.layers_per_gpu = len(self.model.layers)
        self.devices = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]
        self.model.devices = self.devices
        if self.pp_enabled:
            self.gpu_list = list(range(torch.cuda.device_count()))
            self.num_gpus = len(self.gpu_list)
            self.layers_per_gpu = len(self.model.layers) // self.num_gpus
            for i in range(self.num_gpus):
                start_idx = i * self.layers_per_gpu
                end_idx = (
                    (i + 1) * self.layers_per_gpu
                    if i != self.num_gpus - 1
                    else len(self.model.layers)
                )
                layer_chunk = self.model.layers[start_idx:end_idx]
                self.model.layer_gpu_list.append(layer_chunk)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaSelectModel",
    "LlamaSelectForCausalLM",
]
 