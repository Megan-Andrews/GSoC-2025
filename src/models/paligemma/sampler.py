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
"""Sampler for PaliGemma transformer.

An example of a sampling class for a PaliGemma model.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import flax
from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp

from gemma import modules
# from vit import sow_lib
from PIL import Image
import numpy as np
import transformer as transformer_lib

import sentencepiece as spm


def _sample_top_p(probs: jnp.ndarray, p: float, key: jax.Array) -> jnp.ndarray:
  """Sample a token using top-p sampling."""
  probs_sorted, indices = jax.lax.top_k(probs, k=probs.shape[-1])
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > p
  probs_sorted = jnp.where(mask, 0.0, probs_sorted)
  probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

  next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

  next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)
  return next_token

#TODO: Validate this.
def _compute_paligemma_attention_mask(
    time_step: jax.Array,
    seq_len: int, #text_cache_size
    img_len: int,
    input_mask: jax.Array
) -> jnp.ndarray:
  """Return [B,1,img_len+seq_len] with True=keep."""
  # print("input mask shape:", input_mask.shape)
  # print("time step:", time_step.shape)
  # print("img len", img_len)
  batch_size = input_mask.shape[0]
  batch_time_step = jnp.full(
    (batch_size, 1),
    time_step,
    dtype=jnp.uint32
  ) # (B,1) a column vector where all values are timestep.
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0),
      # Expand to (1,S) and compare with (B,1) → broadcasts to (B,S)
      batch_time_step
  ) # (B,S) vector with true for Future keys -> causal mask
  max_seq_len = min(input_mask.shape[-1], seq_len)
  # lets you cut out a sub-array whose start index is computed at runtime
  input_mask = jax.lax.dynamic_slice(
      input_mask,
      # length-N tuple/array of integer starts
      (0, jnp.maximum(time_step - seq_len + 1, 0)),
      (batch_size, max_seq_len), # slice sizes
  ) # t: extract the window of PAD flags that aligns with the last seq_len
    # keys ending at time_step, producing [B, max_seq_len
  input_mask = (
      jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
      .at[:, :max_seq_len]
      .set(input_mask)
  ) # paste this into a zero array to make it exactly [B, seq_len]
  causal_padding = ~jnp.logical_or(causal_padding, input_mask)
  visible_img = jnp.ones((batch_size, img_len), dtype=jnp.bool_)
  keep = jnp.concatenate([visible_img, causal_padding], axis=-1)
  # Add a singleton query axis → (B, 1, S). Many attention kernels expect [B, Q, K]
  attention_mask = keep[:, jnp.newaxis, :].astype(jnp.bool_)

  return attention_mask

@flax.struct.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Number of tokens in the prompt.
  num_input_tokens: jnp.ndarray  # [B]

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: modules.LayerCache # NOTE: Modified
  # cache: dict[str, modules.LayerCache]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: Sequence[int] | None

  # Intermediate activations from the model if requested.
  # TODO: sow intermediates
  intermediates: None #sow_lib.TransformerIntermediates | None

  # Random seed for sampling.
  seed: jax.Array

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Top-p sampling threshold.
  top_p: float = flax.struct.field(pytree_node=False)

  # Image
  zimg: jnp.ndarray | None = None      # [B, Zi, D] image prefix tokens (fixed)

  # Image length.
  # img_len: jnp.int32 = jnp.int32(0)    # Zi

@dataclasses.dataclass
class SamplerOutput:
  """Output of the sampler."""

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[list[float]]

  # Tokens corresponding to the generated samples.
  tokens: list[list[int]]

  # Intermediate activations from the model if requested.
  intermediates: None #sow_lib.TransformerIntermediates | None = None # TODO


class Sampler:
  """Sampler for gemma transformer."""

  def __init__(
      self,
      transformer: transformer_lib.PaliGemmaTransformer,
      vocab: spm.SentencePieceProcessor,
      cache_size: int = 1024,
  ):
    """Initializes a sampler for a PaliGemma model.

    Args:
      transformer: an instance of the PaliGemma transformer.
      vocab: vocabulary of the given model.
      cache_size: size of the cache for the transformer.
    """
    self.vocab = vocab
    self.cache_size = cache_size
    graphdef, state = nnx.split(transformer)
    self._transformer_graphdef: graph.NodeDef = graphdef
    self._transformer_state: statelib.State = state
    # we separate out state and graph def so that the state can be passed as an
    # argument to _sample_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time
    self._compiled_sample_fn = jax.jit(self._sample_fn)

  @property
  def transformer(self) -> transformer_lib.PaliGemmaTransformer:
    return nnx.merge(self._transformer_graphdef, self._transformer_state)

  @property
  def transformer_state(self) -> statelib.State:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> statelib.State:
    def check_tree_structure(tree1, tree2):
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            "New state must have the same structure as the old state."
        )

      def check_shape_dtype(x, y):
        return jnp.shape(x) == jnp.shape(y) and jnp.dtype(x) == jnp.dtype(y)

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype, tree1, tree2)
          )
      ):
        raise ValueError(
            "New state must have the same shape and dtype as the old state."
        )

    check_tree_structure(self._transformer_state, state)
    self._transformer_state = state

  @property
  def dtype(self) -> jnp.dtype:
    return jax.tree_util.tree_leaves(
        nnx.to_flat_state(self._transformer_state)
    )[0].dtype

  # TODO verify correctness
  def _prefill_cache(
      self,
      params: statelib.State,
      images: Sequence[jnp.ndarray],
      all_input_ids: list[jax.Array],
      max_new_tokens: int,
  ):
    """Prefill cache with image + text prompt
     return (cache, zimg, Zi, lens)."""
    transformer = nnx.merge(self._transformer_graphdef, params)

    # 1) Encode image prefix once (fixed across decoding)
    zimg = transformer.embed_image(images)        # [B, Zi, D] TODO check shape
    img_len = zimg.shape[1]
    batch_size = len(all_input_ids)

    # 2) Build dense prompt matrix [B, P] and lengths
    pad = self.vocab.pad_id()
    max_id_len = max(len(ids) for ids in all_input_ids)
    prompt_mat = jnp.full((batch_size, max_id_len), pad, dtype=jnp.int32)
    lens = []
    for i, ids in enumerate(all_input_ids):
      prompt_mat = prompt_mat.at[i, : len(ids)].set(ids)
      lens.append(len(ids))
    lens = jnp.array(lens, dtype=jnp.int32)

    # 3) Global positions for prompt (offset by Zi)
    # text_positions = jnp.arange(
    # max_id_len, dtype=jnp.int32)[None, :].repeat(batch_size, 0)
    # global_positions = text_positions + img_len  # [B, P]

    # 3) Ask model to prefill cache.
    # cache_size must cover Zi + P + max_new_tokens
    cache_size = img_len + max_id_len + max_new_tokens
    cache = transformer.llm.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=self.dtype,
    )
    x, input_mask, mask_ar = \
      transformer.embed_image_and_text(images, all_input_ids)

    _, cache = transformer.prefill_cache( # TODO should the logits be used?
      x, input_mask, mask_ar,
      cache, cache_size=cache_size, num_patches=img_len)

    return cache, zimg, img_len, lens

  def _sample_step(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs a single sampling step."""
    batch_size = sampler_state.token_buffer.shape[0]
    decoding_step = jnp.asarray(sampler_state.decoding_step, dtype=jnp.int32)
    last_token = sampler_state.token_buffer[:, decoding_step]

    # text_keep_mask: True where text keys are valid (not PAD)
    text_keep_mask = (sampler_state.token_buffer != self.vocab.pad_id())
    text_keep_mask = text_keep_mask[:, : self.cache_size]  # safety

    # print("decoding step", decoding_step)
    # print("img len", sampler_state.img_len)
    attention_mask = _compute_paligemma_attention_mask( # TODO change
      time_step=decoding_step,
      seq_len=self.text_cache_len,
      img_len=self.img_len_static,
      input_mask=text_keep_mask, # TODO rename to text_keep_mask
    )

    # GLOBAL position = image offset + local text position
    # TODO: review this
    step_positions = (sampler_state.positions[:, decoding_step]+ self.img_len_static)[..., None]  # [B,1]

    # step_positions = jnp.expand_dims(
    #     sampler_state.positions[:, decoding_step], -1
    # )

    last_token = last_token.reshape((batch_size, 1))

    transformer = nnx.merge(self._transformer_graphdef, params)

    # IMPORTANT: now that image+prompt are already in the cache,
    # we *only* step the LLM with the *new text token*.
    
    logits, new_cache = transformer.llm(
        last_tokens=last_token,
        positions=step_positions,
        cache=sampler_state.cache,
        attention_mask=attention_mask,
    )

    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    def sample_top_p(logits, key):
      probs = jax.nn.softmax(logits[:, -1] / sampler_state.temperature, axis=-1)
      next_token = _sample_top_p(probs, sampler_state.top_p, key)
      return next_token

    def sample_best(logits):
      next_token = jnp.argmax(logits, axis=-1)
      next_token = next_token[:, 0]
      return next_token

    if sampler_state.temperature > 0:
      key = jax.random.fold_in(sampler_state.seed, decoding_step)
      next_token_candidate = sample_top_p(logits, key)
    else:
      next_token_candidate = sample_best(logits)

    # print("next token candidate before", next_token_candidate)
    next_token_candidate = jnp.where(
        decoding_step < sampler_state.num_input_tokens - 1,
        sampler_state.token_buffer[:, decoding_step + 1],
        next_token_candidate,
    )
    # print("next token candidate after", next_token_candidate)

    token_buffer = sampler_state.token_buffer.at[:, decoding_step + 1].set(
        next_token_candidate
    )

    if sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = sampler_state.logits_buffer.at[:, decoding_step + 1].set(
          next_logits
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    if sampler_state.intermediates is not None:
      sampler_state.intermediates.merge(decoding_step, transformer)

    done = sampler_state.done | jnp.equal(
        token_buffer[:, decoding_step + 1], self.vocab.eos_id()
    )
    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        cache=new_cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        intermediates=sampler_state.intermediates,
        temperature=sampler_state.temperature,
        top_p=sampler_state.top_p,
        seed=sampler_state.seed,
        zimg=sampler_state.zimg,
    )

  def init_sample_state(
      self,
      all_input_ids: list[jax.Array],
      total_sampling_steps: int,
      include_logits: bool,
      forbidden_token_ids: Sequence[int] | None,
      temperature: float,
      top_p: float,
      seed: jax.Array,
      decoding_step: jnp.int32,
      cache: modules.LayerCache,
      zimg: jnp.ndarray,
      # img_len: jnp.int32,

  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    batch_size = len(all_input_ids)
    num_input_tokens = [len(input_ids) for input_ids in all_input_ids]
    buffer_size = total_sampling_steps + 1

    token_buffer = jnp.full(
        (
            batch_size,
            buffer_size,
        ),
        self.vocab.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    for i, (input_ids, num_tokens) in enumerate(
        zip(all_input_ids, num_input_tokens)
    ):
      token_buffer = token_buffer.at[i, :num_tokens].set(input_ids)
      input_mask = input_mask.at[i, :num_tokens].set(
          input_ids != self.vocab.pad_id()
      )
    positions = transformer_lib.build_positions_from_mask(input_mask)

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, buffer_size, self.transformer.llm.num_embed),
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None

    return _SamplingState(
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache=cache, # TODO verify
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        intermediates=self.transformer.init_intermediates(
            batch_size, buffer_size, self.transformer.sow_config
        ),
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        decoding_step=decoding_step,
        zimg=zimg,
    )

  def tokenize(self, input_string: str) -> jax.Array:
    """Tokenizes the input string."""
    input_ids = self.vocab.EncodeAsIds(input_string)
    input_ids = jnp.array(
        [self.vocab.bos_id()] + jnp.array(input_ids).tolist(), dtype=jnp.int32
    )
    return input_ids

  # TODO: refactor this function
  def load_and_preprocess_image(self, image_path, target_size=(224, 224)) -> jnp.ndarray:
    """Loads and normalizes an image to [0, 1] float32 tensor."""
    img = Image.open(image_path).convert("RGB").resize(target_size)
    img_np = np.array(img) / 255.0
    return jnp.array(img_np, dtype=jnp.float32)

  def mask_tokens_after_eos_ids(self, token_buffer):
    """Mask token IDs after the EOS token with the padding ID."""
    eos_id = self.vocab.eos_id()
    eos_exists = jnp.any(jnp.equal(token_buffer, eos_id), axis=-1)
    eos_indices = jnp.where(
        eos_exists,
        jnp.argmax(jnp.equal(token_buffer, eos_id), axis=-1),
        token_buffer.shape[-1],
    )
    mask = jnp.less_equal(
        jnp.arange(token_buffer.shape[-1]), eos_indices[:, None]
    )
    masked_token_buffer = token_buffer * mask + self.vocab.pad_id() * (1 - mask)

    return masked_token_buffer

  def _sample_fn(
      self,
      params: statelib.State,
      initial_sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal sampling function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      # print("Sample step ", sampler_state.decoding_step)
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(
        cond_fn, sample_with_params, initial_sampling_state
    )

  def __call__(
      self,
      input_strings: Sequence[str],
      images: Sequence[jnp.ndarray], # TODO verify correctness
      total_generation_steps: int,
      echo: bool = False,
      return_logits: bool = True,
      forbidden_tokens: Sequence[str] | None = None,
      temperature: float = 0.0,
      top_p: float = 0.95,
      seed: jax.Array | None = None,
  ) -> SamplerOutput:
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      echo: whgether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      forbidden_tokens: list of tokens that are forbidden to be generated. Each
        token must map to a single token id in the vocab.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      seed: random seed for sampling.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    # print("Calling Sampler")
    forbidden_token_ids = None
    if forbidden_tokens is not None:
      forbidden_token_ids = []
      for token in forbidden_tokens:
        token_id = self.vocab.EncodeAsIds(token)
        if len(token_id) != 1:
          raise ValueError(
              "Forbidden tokens must map to single token ids in the vocab."
          )
        forbidden_token_ids.extend(token_id)
      forbidden_token_ids = tuple(forbidden_token_ids)
    all_input_ids = [self.tokenize(x) for x in input_strings]
    max_input_length = max(len(input_ids) for input_ids in all_input_ids)
    total_sampling_steps = max_input_length + total_generation_steps

    if seed is None:
      seed = jax.random.PRNGKey(0)
    # print("Initializing sample state")

    cache, zimg, img_len, lens = self._prefill_cache(
        self._transformer_state,
        images=images,
        all_input_ids=all_input_ids,
        max_new_tokens=total_generation_steps,
    )
    self.img_len_static = int(img_len)
    self.text_cache_len = int((img_len + max(lens) + total_generation_steps) - img_len)
    # Start at the max prompt length – 1 for the batch
    start_step = (jnp.max(lens) - 1).astype(jnp.int32)
    init_state = self.init_sample_state(
        all_input_ids,
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        decoding_step=start_step,           # we prefilled up to t = Zi+P-1
        cache=cache,
        zimg=zimg,
        # img_len=img_len
    )

    sampling_state = self._compiled_sample_fn(
        self._transformer_state, init_state
    )

    masked_token_buffer = self.mask_tokens_after_eos_ids(
        sampling_state.token_buffer
    )

    out_tokens = []
    out_logits = []
    for i, (token_buffer, num_tokens) in enumerate(
        zip(
            masked_token_buffer,
            sampling_state.num_input_tokens,
        )
    ):

      start_idx = 0 if echo else num_tokens
      out_tokens.append(token_buffer[start_idx:total_sampling_steps].tolist())
      if return_logits:
        logits_buffer = sampling_state.logits_buffer[i]
        out_logits.append(
            logits_buffer[start_idx:total_sampling_steps].tolist()
        )

    decoded_outputs = [self.vocab.DecodeIds(tokens) for tokens in out_tokens]

    if sampling_state.intermediates is not None:
      sampling_state.intermediates.trim(total_sampling_steps)

    result = SamplerOutput(
        text=decoded_outputs,
        logits=out_logits,
        tokens=out_tokens,
        intermediates=sampling_state.intermediates,
    )
    return result
