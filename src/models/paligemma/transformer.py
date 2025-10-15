"""PaliGemma Transformer"""

# from typing import Optional
from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Any

from flax import nnx
import helpers
import jax.numpy as jnp
import jax
from jaxtyping import Array

import gemma.sow_lib as sow_lib
import params as params_lib
# import gemma.modules as gemma_modules
from gemma.transformer import Transformer as GemmaTransformer
from gemma.transformer import TransformerConfig as GemmaConfig
from vit.transformer import VisionTransformer, VisionTransformerConfig as ViTConfig


# TODO add init cache for gemma 
@dataclass(frozen=True)
class PaliGemmaConfig:
  """Config for PaliGemma."""
  vit_config: ViTConfig
  gemma_config: GemmaConfig
  # max_seq_length: int = 4096

  @classmethod
  def from_params(cls, params: params_lib.Params) -> PaliGemmaConfig:
    gemma_config = GemmaConfig.from_params(params)
    vit_config = ViTConfig.from_params(params)
    return cls(gemma_config=gemma_config,
                vit_config=vit_config)

def _assign_npz_params_to_nnx_state(
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


def _map_npz_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
  """Maps npz variable names to nnx variable names."""
  new_key = []
  for k in key:
    if k == 'MultiHeadDotProductAttention_0':
      new_key.append('attn')
    elif k == 'MlpBlock_0':
      new_key.append('mlp')
    elif k == 'Dense_0':
      new_key.append('linear1')
    elif k == 'Dense_1':
      new_key.append('linear2')
    elif k == 'LayerNorm_0':
      new_key.append('norm1')
    elif k == 'LayerNorm_1':
      new_key.append('norm2')
    elif k.startswith('layer_'):
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


class PaliGemmaTransformer(nnx.Module):
  """PaliGemma Transformer model"""

  @classmethod
  def from_params(
    cls,
    params: params_lib.Params,
    config: None | PaliGemmaConfig = None,
    sow_config: None | sow_lib.SowConfig = None,
  ) -> PaliGemmaTransformer:
    if config is None:
      config = PaliGemmaConfig.from_params(params)
    
    assign_val_fn = functools.partial(
      _assign_npz_params_to_nnx_state,
      transpose_gating_einsum = config.gemma_config.transpose_gating_einsum
    )
    return helpers.module_from_linen_variables(
      module_factory = lambda: cls(
        config, rngs=nnx.Rngs(params=0), sow_config=sow_config
      ),
      variables=params,
      map_key_fn= _map_npz_var_names,
      assign_val_fn=assign_val_fn,
    )

  def __init__(
      self,
      config: PaliGemmaConfig,
      *,
      rngs: nnx.Rngs,
      sow_config=None, # TODO sow config
  ):
    self.img = VisionTransformer(config.vit_config, rngs=rngs)
    self.llm = GemmaTransformer(config.gemma_config, rngs=rngs)
    self.sow_config = sow_config

  def embed_text(self, tokens):
    ztxt = self.llm.embedder.encode(tokens)
    return ztxt

  def embed_image(self, image):
    # if we have video, fold frame dimension into the batch dimension
    image_shape = image.shape
    if len(image_shape) == 5:  # video frames
      image = jnp.reshape(image, (-1, *image.shape[-3:]))

    # Do we want to normalize? are they huge?
    zimg, _ = self.img(image)

    if len(image_shape) == 5:  # concatenate tokens from all video frames
      zimg = jnp.reshape(zimg, (image_shape[0], -1, zimg.shape[-1]))

    return zimg

  def embed_image_and_text(self, image, text, *, input_mask=None, mask_ar=None):
    zimg = self.embed_image(image)
    ztxt = self.embed_text(text)

    if input_mask is None:
      input_mask = jnp.full(jnp.array(text).shape, True)
    if mask_ar is None:
      mask_ar = jnp.full(jnp.array(text).shape, 1)

    # Concatenate embeded image and text into a single token sequence.
    # print(zimg.shape, ztxt.shape)
    x = jnp.concatenate([zimg, ztxt], axis=1)
    _, img_len, _ = zimg.shape
    pad_width = ((0, 0), (img_len, 0))
    mask_ar = jnp.pad(mask_ar, pad_width, constant_values=0)
    input_mask = jnp.pad(input_mask, pad_width, constant_values=True)

    return x, input_mask, mask_ar

  def make_attn_mask(self, input_mask, mask_ar):
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = (cumsum[:, None, :] <= cumsum[:, :, None])
    valid_mask = (input_mask[:, None, :] * input_mask[:, :, None])
    return jnp.logical_and(attn_mask, valid_mask)


  # TODO verify
  def prefill_cache(self, x, input_mask, mask_ar, cache, *, cache_size, num_patches):
      """Initializes decoding cache with `x` [B, N, E] as prompt.

      IMPORTANT: Inputs MUST be left-aligned and attn_mask should not allow
      input tokens to attend to padding tokens.

      TODO: Relax left-align requirement by converting any input into
      a right aligned input with no attention to padding tokens. """
      # TODO right align
      # x, input_mask, mask_ar = _left_to_right_align(x, input_mask, mask_ar)
      attn_mask = self.make_attn_mask(input_mask, mask_ar)
      # seq_len = jnp.sum(input_mask, axis=-1)
      positions = jnp.cumsum(input_mask, axis=-1) - 1
      batch_size, prefill_len, _ = x.shape

       # Pad attention to set the cache size.
      mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, cache_size - prefill_len)))

      # print("prefill cache")
      # print("x", x.shape)
      # print("prefill_len", prefill_len, "cache_size", cache_size)
      # print("input_mask", input_mask.shape, "mask_ar", mask_ar.shape)
      # print("mask", mask.shape,  "attn_mask", attn_mask.shape, "positions", positions.shape)
      pre_logits, cache = self.llm(
          last_tokens=None,
          positions=positions,
          cache=cache,
          attention_mask=mask,
          embeddings=x,
          encode_tokens=False,
          return_pre_logits=True
      )
      # print("post llm cache keys", cache.keys if cache else None)
      # print("compute logits")
      return self.llm.compute_logits(pre_logits), cache



  # def prefill_vision(self, images, cache):
  #   """Write image K/V into cache; returns updated cache.
  #   images: [B, H, W, C] or [B, T, H, W, C] (your embed_image already handles video)
  #   """
  #   zimg = self.embed_image(images)             # [B, Zi, D]
  #   batch_size, img_len, _ = zimg.shape
  #   cache_size = _cache_size(cache) # TODO rewrite this function or use attribute

  #   # Positions 0..Zi-1 (global indices for image tokens)
  #   pos_img = jnp.arange(img_len, dtype=jnp.int32)[None, :].repeat(batch_size, 0)  # [B, Zi]

  #   # Keep-mask lets image tokens see earlier image tokens (triangular is safe)
  #   keep_img = _img_keep_mask(batch_size, img_len, cache_size, causal=False)                 # [B, Zi, S] # TODO verify full attention

  #   # Run LLM once to *write* K/V for image into cache
  #   # We ignore logits here; only cache matters for later steps.
  #   _, cache = self.llm(zimg, pos_img, cache, keep_img)
  #   return cache

  # # TODO verify
  # def prefill_text(self, prompt_mat, global_positions, cache):
  #   """Write text-prompt K/V into cache; returns updated cache.
  #   prompt_mat: [B, P] token IDs
  #   global_positions: [B, P] = (Zi + local_positions)
  #   """
  #   x_text, _ = self.embed_text(prompt_mat)     # [B, P, D]
  #   batch_size, seq_len, _ = x_text.shape
  #   end_index = _cache_end_index(cache)                # should equal image length after prefill_vision()
  #   cache_size  = _cache_size(cache)

  #   # Keep-mask: image fully visible, text causal
  #   keep_txt = _text_keep_mask(batch_size, end_index, seq_len, cache_size)     # [B, P, S]

  #   # Run LLM once to *append* prompt K/V after image K/V
  #   _, cache = self.llm(x_text, global_positions, cache, keep_txt)
  #   return cache


  def __call__(
    self,
    image = None,
    text = None,
    mask_ar = None,

    # last_tokens: Array,  # [B, L]
    # positions: Array,  # [B, L]
    # cache: modules.LayerCache | None,  # (sequence length L')
    # attention_mask: Array,  # [B, L, L']
    # embeddings: Array = None,
    # encode_tokens: bool = True,
    # return_pre_logits: bool = False,
    
  ):
    # Embed the image and text.
    x, input_mask, mask_ar = self.embed_image_and_text(image, text, mask_ar=mask_ar)

    # Call transformer on the embedded token sequence.
    attn_mask = self.make_attn_mask(input_mask, mask_ar)
    # Compute RoPE positions
    positions = build_positions_from_mask(input_mask)  # [B, T+I]

    # Run the transformer (no cache for now)
    pre_logits, cache = self.llm(
        last_tokens=None,
        positions=positions,
        cache=None,
        attention_mask=attn_mask,
        embeddings=x,
        encode_tokens=False,
        return_pre_logits=True,
    )

    # Slice only text portion of hidden states
    zimg = x[:, :x.shape[1] - text.shape[1], :]
    text_pre_logits, _ = pre_logits[:, zimg.shape[1]:, :]

    # Final projection to vocabulary
    text_logits = self.llm.compute_logits(text_pre_logits)
    return text_logits

  # TODO Complete initializing intermediates
  def init_intermediates(self, batch_size, buffer_size,sow_config):
    pass

  

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


# # TODO validate keep masks
# def _tri_mask(n: int) -> jnp.ndarray:
#     """Lower-triangular boolean [n,n] with True where j<=i."""
#     return jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))

# def _img_keep_mask(
#   batch_size: int,
#   img_len: int,
#   cache_size: int, *, causal: bool = True) -> jnp.ndarray:
#     """
#     Keep-mask for image prefill: [B, Zi, S].
#     If causal=True, image tokens attend to previous image tokens (triangular).
#     If causal=False, image tokens can see all image tokens (full).
#     """
#     keep = jnp.zeros((batch_size, img_len, cache_size), dtype=jnp.bool_)
#     block = _tri_mask(img_len) if causal else jnp.ones((img_len, img_len), dtype=jnp.bool_)
#     return keep.at[:, :, :img_len].set(block)

# def _text_keep_mask(batch_size: int, end_index: int, seq_len: int, cache_size: int) -> jnp.ndarray:
#     """
#     Keep-mask for text prefill: [B, P, S]. batch_size, seq_len, cache_size
#     - Text attends to all image keys [0..Zi-1]
#     - Text attends causally to previous text keys [Zi..Zi+P-1]
#     - Everything beyond Zi+P is masked.
#     """
#     keep = jnp.zeros((batch_size, seq_len, cache_size), dtype=jnp.bool_)
#     keep = keep.at[:, :, :end_index].set(True)                 # see all image keys
#     keep = keep.at[:, :, end_index:end_index+seq_len].set(_tri_mask(seq_len))     # causal over prompt
#     return keep


# def _cache_size(cache) -> int:
#     return cache['k'].shape[1]

# # TODO: rewrite function
# def _cache_end_index(cache) -> int:
#     any_layer = next(iter(cache.values()))
#     # end_index is per-batch; all batches share same index during prefill
#     return int(any_layer["end_index"][0])

def _left_to_right_align(x, input_mask, attn_mask):
  """Converts input from left-align to right-aligned."""
  # Due to vmap, this is operating in a single example (not batch level).
  assert x.ndim == 2 and input_mask.ndim == 1 and attn_mask.ndim == 2
  assert x.shape[0] == input_mask.shape[0]
  assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
  seqlen = jnp.sum(input_mask)
  x = jnp.roll(x, -seqlen, axis=0)
  input_mask = jnp.roll(input_mask, -seqlen, axis=0)
  attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
  return x, input_mask, attn_mask