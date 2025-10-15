# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Gemma transformer."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import enum
import functools
from typing import Any

from flax import nnx
from gemma import helpers
from gemma import layers
from gemma import modules
from gemma import params as params_lib
from gemma import sow_lib
import jax
import jax.debug
import jax.numpy as jnp
from jaxtyping import Array  # pylint: disable=g-importing-member,g-multiple-import

Cache = dict[str, modules.LayerCache]


def make_attention_layers_types(
    pattern: tuple[modules.AttentionType, ...],
    num_layers: int,
) -> tuple[modules.AttentionType, ...]:
  """Returns the list of attention types for every layers."""

  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return tuple(out)


class QueryPreAttentionNormalisation(enum.Enum):
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `embed_dim // num_heads`
  BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46
_NUM_LAYERS_GEMMA3_1B = 26
_NUM_LAYERS_GEMMA3_4B = 34
_NUM_LAYERS_GEMMA3_12B = 48
_NUM_LAYERS_GEMMA3_27B = 62
GEMMA3_ATTENTION_PATTERN = (
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.LOCAL_SLIDING,
    modules.AttentionType.GLOBAL,
)


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Configuration for the gemma transformer."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attention_types: Iterable[modules.AttentionType]
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  attn_logits_soft_cap: float | None = None
  transpose_gating_einsum: bool = False
  local_base_frequency: int = modules.DEFAULT_ROPE_BASE_FREQUENCY
  global_base_frequency: int = modules.DEFAULT_ROPE_BASE_FREQUENCY
  local_scale_factor: float = modules.DEFAULT_ROPE_SCALE_FACTOR
  global_scale_factor: float = modules.DEFAULT_ROPE_SCALE_FACTOR
  use_qk_norm: bool = False
  sliding_window_size: int | None = None

  def query_pre_attn_scalar(self) -> float:
    """Returns the scalar to multiply the query by before attention."""
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
        return self.embed_dim // self.num_heads
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
        return (self.embed_dim // self.num_heads) ** -0.5
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
        return self.head_dim**-0.5

  @classmethod
  def from_path(cls, path: str) -> TransformerConfig:
    """Creates a TransformerConfig from loaded parameters."""
    params = params_lib.load_params(path)

    return cls.from_params(params)

  @classmethod
  def from_params(cls, params: params_lib.Params) -> TransformerConfig:
    """Creates a TransformerConfig from loaded parameters.

    Args:
      params: Model parameters

    Returns:
      TransformerConfig.
    """
    # TODO verify Gemma 2 and 3 module names
    # Post Attn Norm is only used starting from Gemma 2.
    use_post_attn_norm = (
        'post_attention_norm' in params['llm']['layers']
    )

    # QK Norm is only used starting from Gemma 3.
    use_qk_norm = '_query_norm' in params['llm']['layers']['attn']

    # TODO: Test that this correctly assigns the number of model layers to num_layers
    first_leaf = next(iter(jax.tree_util.tree_leaves(params['llm']['layers'])))
    num_layers = first_leaf.shape[0]

    if not use_post_attn_norm:  # Gemma 1.
      if num_layers == _NUM_LAYERS_GEMMA_2B:
        return cls.gemma_2b()
      if num_layers == _NUM_LAYERS_GEMMA_7B:
        return cls.gemma_7b()
      raise ValueError(
          'Guessing Gemma 1 model, but could not determine size from params.'
      )
    elif not use_qk_norm:  # Gemma 2.
      if num_layers == _NUM_LAYERS_GEMMA2_2B:
        return cls.gemma2_2b()
      if num_layers == _NUM_LAYERS_GEMMA2_9B:
        return cls.gemma2_9b()
      if num_layers == _NUM_LAYERS_GEMMA2_27B:
        return cls.gemma2_27b()
      raise ValueError(
          'Guessing Gemma 2 model but could not determine size from params.'
      )
    else:  # Gemma 3.
      if num_layers == _NUM_LAYERS_GEMMA3_1B:
        return cls.gemma3_1b()
      if num_layers == _NUM_LAYERS_GEMMA3_4B:
        return cls.gemma3_4b()
      if num_layers == _NUM_LAYERS_GEMMA3_12B:
        return cls.gemma3_12b()
      if num_layers == _NUM_LAYERS_GEMMA3_27B:
        return cls.gemma3_27b()

    raise ValueError('Could not determine Gemma variant from params.')

  @classmethod
  def gemma_2b(cls):
    num_layers = _NUM_LAYERS_GEMMA_2B
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_7b(cls):
    num_layers = _NUM_LAYERS_GEMMA_7B
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(modules.AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma2_2b(cls):
    num_layers = _NUM_LAYERS_GEMMA2_2B
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2304,
        hidden_dim=9216,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_9b(cls):
    num_layers = _NUM_LAYERS_GEMMA2_9B
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=28672,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_27b(cls):
    num_layers = _NUM_LAYERS_GEMMA2_27B
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=4608,
        hidden_dim=72728,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        final_logit_softcap=30.0,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attention_types=(
            modules.AttentionType.LOCAL_SLIDING,
            modules.AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma3_1b(cls):
    num_layers = _NUM_LAYERS_GEMMA3_1B
    return cls(
        num_layers=num_layers,
        final_logit_softcap=None,
        num_embed=262144,
        embed_dim=1152,
        hidden_dim=6 * 1152,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=512,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
    )

  @classmethod
  def gemma3_4b(cls):
    num_layers = _NUM_LAYERS_GEMMA3_4B
    return cls(
        num_layers=num_layers,
        final_logit_softcap=None,
        num_embed=262_144,
        embed_dim=2560,
        hidden_dim=2560 * 8 // 2,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
    )

  @classmethod
  def gemma3_12b(cls):
    num_layers = _NUM_LAYERS_GEMMA3_12B
    return cls(
        num_layers=num_layers,
        final_logit_softcap=None,
        num_embed=262144,
        embed_dim=30 * 128,
        hidden_dim=8 * 30 * 128 // 2,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
    )

  @classmethod
  def gemma3_27b(cls):
    num_layers = _NUM_LAYERS_GEMMA3_27B
    return cls(
        num_layers=num_layers,
        final_logit_softcap=None,
        num_embed=262144,
        embed_dim=5376,
        hidden_dim=5376 * 8 // 2,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attention_types=make_attention_layers_types(
            GEMMA3_ATTENTION_PATTERN, num_layers
        ),
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        attn_logits_soft_cap=None,
        sliding_window_size=1024,
        transpose_gating_einsum=True,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
    )


def _map_linen_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
  """Maps linen variable names to nnx variable names."""
  new_key = []
  for k in key:
    if k.startswith('layer_'):
      prefix, suffix = k.split('layer_')
      assert not prefix, prefix
      new_key.append('layers')
      new_key.append(int(suffix))
    elif k == 'gating_einsum':
      new_key.append('gate_proj')
      new_key.append('kernel')
    elif k == 'linear':
      new_key.append('down_proj')
      new_key.append('kernel')
    else:
      new_key.append(k)

  return tuple(new_key)


def _assign_linen_params_to_nnx_state(
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any,
    transpose_gating_einsum: bool,
) -> dict[tuple[str, ...], Any]:
  """Splits and maybe transposes gate_proj."""
  if 'gate_proj' in mapped_path:
    if transpose_gating_einsum:
      # TODO: verify if swapping axes is correctly implemented for npz params
      val = jnp.swapaxes(val, 1, 2)
    
    # val: (L, 2, in_dim, out_dim)
    gate_proj_val = val[:, 0, ...]  # (L, in_dim, out_dim)
    up_proj_val   = val[:, 1, ...]  # (L, in_dim, out_dim)

    state[mapped_path].value = gate_proj_val
    state[mapped_path[:-2] + ('up_proj', 'kernel')].value = up_proj_val

  else:
    state[mapped_path].value = val
  return state


class Transformer(nnx.Module):
  """Gemma transformer."""

  @classmethod
  def from_params(
      cls,
      params: params_lib.Params,
      config: None | TransformerConfig = None,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
  ) -> Transformer:
    if config is None:
      config = TransformerConfig.from_params(params)
    assign_val_fn = functools.partial(
        _assign_linen_params_to_nnx_state,
        transpose_gating_einsum=config.transpose_gating_einsum,
    )
    return helpers.module_from_linen_variables(
        module_factory=lambda: cls(
            config, rngs=nnx.Rngs(params=0), sow_config=sow_config
        ),
        variables=params['llm'],
        map_key_fn=_map_linen_var_names,
        assign_val_fn=assign_val_fn,
    )

  def compute_logits(self, pre_logits):
    logits = self.embedder.decode(pre_logits)
    return logits

  def __init__(
      self,
      config: TransformerConfig,
      *,
      rngs: nnx.Rngs,
      sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
  ):
    self.config = config
    self.embedder = modules.Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        rngs=rngs,
    )

    # Modified layers to be stacked.
    @nnx.split_rngs(splits=config.num_layers)
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def make_block(attn_type, rngs: nnx.Rngs):
      return modules.Block(
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        embed_dim=config.embed_dim,
        head_dim=config.head_dim,
        hidden_dim=config.hidden_dim,
        sliding_window_size=config.sliding_window_size,
        use_post_attn_norm=config.use_post_attn_norm,
        use_post_ffw_norm=config.use_post_ffw_norm,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        attn_type=attn_type,
        query_pre_attn_scalar=config.query_pre_attn_scalar(),
        rngs=rngs,
        rope_base_frequency=config.local_base_frequency
        if attn_type == modules.AttentionType.LOCAL_SLIDING
        else config.global_base_frequency,
        rope_scale_factor=config.local_scale_factor
        if attn_type == modules.AttentionType.LOCAL_SLIDING
        else config.global_scale_factor,
        use_qk_norm=config.use_qk_norm,
        sow_config=sow_config,
      )
    
    attn_type_ids = [attn_type.value for attn_type in config.attention_types]
    self.layers = make_block(attn_type_ids, rngs)
    self.final_norm = layers.RMSNorm(config.embed_dim, rngs=rngs)
    self.final_logits_softcap = config.final_logit_softcap
    self.sow_config = sow_config

  def __call__(
      self,
      last_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: modules.LayerCache | None,  # (sequence length L')
      attention_mask: Array,  # [B, L, L']
      embeddings: Array = None,
      encode_tokens: bool = True,
      return_pre_logits: bool = False,
  ) -> tuple[Array, modules.LayerCache | None]:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    @nnx.split_rngs(splits=self.num_layers)
    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def forward_block(x, block: modules.Block, cache):
      # Each layer should get its own slice of the cache
      # The scan automatically handles this when cache is passed as scan input
      # print(f"k shape: {k.shape}")
      # print(f"v shape: {v.shape}")
      # print(f"end_index shape: {end_index.shape}")
      # cache: modules.LayerCache = {'k': k, 'v': v, 'end_index': end_index}
      # jax.debug.print("Processing layer with cache: {}", cache)
      new_cache, x = block(x, positions, cache, attention_mask)
      # print(f"new_cache shape: {new_cache['k'].shape}")
      # print(f"new_cache shape: {new_cache['v'].shape}")
      # print(f"new_cache shape: {new_cache['end_index'].shape}")
      # jax.debug.print("Layer output cache: {}", new_cache)
      return x, new_cache#new_cache['k'], new_cache['v'], new_cache['end_index']

    new_cache = None if cache is None else {}

    x = embeddings if not encode_tokens else self.embedder.encode(last_tokens)
    self.sow_config.maybe_sow_embeddings(x, self)

    # jax.debug.print("Initial cache: {}", cache.keys if cache else None)
    # jax.debug.print("Initial layers: {}", self.layers)
    # jax.debug.print("Cache shapes: {}", cache['v'].shape)
    # jax.debug.print("x {}", x.shape)
    # jax.debug.print("layers {}", self.layers)

    # print(f"Cache shapes: {cache['v'].shape}")
    # print(f"Cache shapes: {cache['end_index'].shape}")
    # print(f"Attention mask shape: {attention_mask.shape} ")
    

    x, cache = forward_block(x, self.layers, cache)
    if cache is not None:
      new_cache = cache

    # jax.debug.print("Final new cache: {}", new_cache.keys if new_cache else None)
    # jax.debug.print("Final  cache: {}", cache.keys if cache else None)
    # jax.debug.print("Post Block X: {}", x)
    # print(f"Post Block X shape: {x.shape}")

    x = self.final_norm(x)
    if return_pre_logits:
      return x, new_cache
    logits = self.compute_logits(x)

    if self.final_logits_softcap is not None:
      logits /= self.final_logits_softcap
      logits = jnp.tanh(logits) * self.final_logits_softcap

    # jax.debug.print("final logits: {}", logits)
    return logits, new_cache  # pytype: disable=bad-return-type

  @property
  def embed_dim(self) -> int:
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    return self.embedder.num_embed

  @property
  def num_layers(self) -> int:
    return self.config.num_layers

  def init_cache(
      self,
      cache_size: int,
      batch_size: int,
      dtype: jnp.dtype = jnp.bfloat16,
  ) -> modules.LayerCache:
    """Initializes a new Transformer cache."""

    @nnx.split_rngs(splits=self.num_layers)
    @nnx.scan(in_axes=0, out_axes=0)
    def scan_cache_init(block: modules.Block):
      cache = block.init_cache(cache_size, batch_size, dtype)
      return cache

    return scan_cache_init(self.layers)


  def init_intermediates(
      self,
      batch_size: int,
      buffer_size: int,
      sow_config: sow_lib.SowConfig,
      dtype: jnp.dtype = jnp.float32,
  ) -> sow_lib.TransformerIntermediates:
    """Initializes the intermediate activations that will be filled."""
    # print("Initializing Sow Intermediates")
    intermediates = sow_lib.TransformerIntermediates()
    residual_stream_dummy = jnp.zeros(
        (batch_size, buffer_size, self.embed_dim),
        dtype=dtype,
    )
    residual_stream_dummy_layers = jnp.zeros(
        (self.num_layers, batch_size, buffer_size, self.embed_dim),
        dtype=dtype,
    )
    if sow_config.embeddings:
      intermediates.embeddings = residual_stream_dummy
    # for _ in range(self.num_layers):
    layer_intermediates = sow_lib.LayerIntermediates()
    # print("Initializing layer intermediates")
    # print("Sow config:", sow_config)
    if sow_config.rs_after_attention:
      layer_intermediates.rs_after_attention = residual_stream_dummy_layers
    if sow_config.rs_after_ffw:
      layer_intermediates.rs_after_ffw = residual_stream_dummy_layers
    if sow_config.attn_logits_topk:
      shape = (
          self.num_layers, # Fist dimension is the num layers in stacked intermediates
          batch_size,
          buffer_size,
          self.config.num_heads,
          sow_config.attn_logits_topk,
      )
      layer_intermediates.attn_logits_topk_values = jnp.zeros(
          shape,
          dtype=dtype,
      )
      layer_intermediates.attn_logits_topk_indices = jnp.zeros(
          shape,
          dtype=jnp.int32,
      )
    if sow_config.mlp_hidden_topk:
      shape = (
          self.num_layers, # Fist dimension is the num layers in stacked intermediates
          batch_size,
          buffer_size,
          sow_config.mlp_hidden_topk,
      )
      layer_intermediates.mlp_hidden_topk_values = jnp.zeros(
          shape,
          dtype=dtype,
      )
      layer_intermediates.mlp_hidden_topk_indices = jnp.zeros(
          shape,
          dtype=jnp.int32,
      )
    # print("Setting Layer intermediates")
    intermediates.layers = layer_intermediates
    return intermediates


def make_causal_attn_mask(
    input_mask: Array,
) -> Array:
  """Attention mask in batch mode.

  Args:
    input_mask: Input mask for the input. True for non-padded tokens only, else
      False.

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def build_positions_from_mask(input_mask: Array) -> Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)
