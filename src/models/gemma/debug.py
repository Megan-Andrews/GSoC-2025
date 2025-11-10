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
"""Tests for the Gemma transformer."""
from collections.abc import Iterable
from collections import defaultdict
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import modules
import sampler as sampler_lib
import sow_lib
import transformer as transformer_lib
import jax.numpy as jnp
import numpy as np
import jax
import sentencepiece as spm


class MockVocab(spm.SentencePieceProcessor):

  def __init__(self):
    super().__init__()
    self._start_id = 3
    self._mapping_text_to_id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        'input': 3,
        'string': 4,
        'hello': 5,
        'world': 6,
        'Hello': 7,
        'there': 8,
        '!': 9,
        'My': 10,
        'name': 11,
        'is': 12,
        'Morgane': 13,
    }
    self._vocab_size = len(self._mapping_text_to_id)

  def pad_id(self) -> int:
    return 0

  def bos_id(self) -> int:
    return 1

  def eos_id(self) -> int:
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return ' '.join(reverse_mapping[e] for e in ids)

  def EncodeAsIds(self, text: str) -> list[int]:  # pylint: disable=invalid-name
    words = text.split(' ')
    return [self._mapping_text_to_id[word] for word in words]




def create_fake_params(config: transformer_lib.TransformerConfig):
  def nested_defaultdict():
    return defaultdict(nested_defaultdict)

  res = nested_defaultdict()
  res['transformer'] = nested_defaultdict()
  params = res['transformer']
  key = jax.random.PRNGKey(0)
  
  # 1. embedding params
  shape =  (config.num_embed, config.embed_dim)
  params['embedder']['input_embedding'] = jax.random.normal(key, shape)
    # jnp.ones(
    #     (config.num_embed, config.embed_dim)
    # )
  # 2. final norm params
  shape = (config.embed_dim,)
  params['final_norm'] = {'scale':  jax.random.normal(key, shape)} #jnp.ones((config.embed_dim,))}

  # 3. attention block params
  for layer_idx in range(config.num_layers):
    shape = (config.num_heads, config.head_dim, config.embed_dim)
    value = jax.random.normal(key, shape)
    params[f'layer_{layer_idx}']['attn']['attn_vec_einsum']['w'] = value
      # jnp.ones(
      #     (config.num_heads, config.head_dim, config.embed_dim)
      # )
    if config.num_heads == config.num_kv_heads:
      shape = (3, config.num_heads, config.embed_dim, config.head_dim)
      value = jax.random.normal(key, shape)
      params[f'layer_{layer_idx}']['attn']['qkv_einsum']['w'] = value
        # jnp.ones(
        #     (3, config.num_heads, config.embed_dim, config.head_dim)
        # )
    else:
      shape = (config.num_heads, config.embed_dim, config.head_dim)
      value = jax.random.normal(key, shape)
      params[f'layer_{layer_idx}']['attn']['q_einsum']['w'] = value 
        # jnp.ones(
        #     (config.num_heads, config.embed_dim, config.head_dim)
        # )
      params[f'layer_{layer_idx}']['attn']['kv_einsum']['w'] = value
        # jnp.ones(
        #     (config.num_kv_heads, config.embed_dim, config.head_dim)
        # )

    # 4. feedforward block params
    shape = (2, config.embed_dim, config.hidden_dim)
    value = jax.random.normal(key, shape)
    params[f'layer_{layer_idx}']['mlp']['gating_einsum'] = value
      # jnp.ones(
      #     (2, config.embed_dim, config.hidden_dim)
      # )
    shape = (config.hidden_dim, config.embed_dim)
    value = jax.random.normal(key, shape)
    params[f'layer_{layer_idx}']['mlp']['linear'] = value
      # jnp.ones(
      #     (config.hidden_dim, config.embed_dim)
      # )

    # 5. layer norm params
    shape = (config.embed_dim,)
    value = jax.random.normal(key, shape)
    params[f'layer_{layer_idx}']['pre_attention_norm']['scale'] = value
      # jnp.ones((
      #     config.embed_dim,
      # ))
    params[f'layer_{layer_idx}']['pre_ffw_norm']['scale'] = value
    # jnp.ones((
    #     config.embed_dim,
    # ))

    if config.use_post_attn_norm:
      params[f'layer_{layer_idx}']['post_attention_norm']['scale'] = value
      # jnp.ones((
      #     config.embed_dim,
      # ))
    if config.use_post_ffw_norm:
      params[f'layer_{layer_idx}']['post_ffw_norm']['scale'] = value
      # jnp.ones((
      #     config.embed_dim,
      # ))
  return res


class TransformerTest(parameterized.TestCase):

  # @parameterized.parameters(
  #     # Prime number to ease shape tracing
  #     # dict(
  #     #     num_layers=3,
  #     #     num_embed=17,
  #     #     embed_dim=2,
  #     #     num_heads=2,
  #     #     num_kv_heads=2,
  #     #     hidden_dim=11,
  #     #     head_dim=8,
  #     #     cache_size=29,
  #     #     batch_size=7,
  #     #     sequence_length=17,
  #     #     expected_outputs_shape=(7, 17, 17),
  #     #     expected_cache_shape=(7, 29, 2, 8),
  #     # ),
  #     dict(
  #         num_layers=3,
  #         num_embed=4,
  #         embed_dim=2,
  #         num_heads=2,
  #         num_kv_heads=2,
  #         hidden_dim=4,
  #         head_dim=2,
  #         cache_size=3,
  #         batch_size=5,
  #         sequence_length=1,
  #         expected_outputs_shape=(5, 1, 4),
  #         expected_cache_shape=(5, 3, 2, 2), #  batch_size, cache_size, num_heads, head_dim
  #     ),
  # )
  # def test_transformer(
  #     self,
  #     num_layers,
  #     num_embed,
  #     embed_dim,
  #     num_heads,
  #     num_kv_heads,
  #     hidden_dim,
  #     head_dim,
  #     cache_size,
  #     batch_size,
  #     sequence_length,
  #     expected_outputs_shape,
  #     expected_cache_shape,
  # ):

  #   config = transformer_lib.TransformerConfig(
  #       num_layers=num_layers,
  #       num_embed=num_embed,
  #       embed_dim=embed_dim,
  #       hidden_dim=hidden_dim,
  #       num_heads=num_heads,
  #       head_dim=head_dim,
  #       num_kv_heads=num_kv_heads,
  #       final_logit_softcap=None,
  #       attention_types=[modules.AttentionType.GLOBAL] * num_layers,
  #       use_post_attn_norm=False,
  #       use_post_ffw_norm=False,
  #   )
  #   attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
  #   params = create_fake_params(config)
  #   transformer = transformer_lib.Transformer.from_params(params, config)
  #   cache = transformer.init_cache(
  #       cache_size=cache_size,
  #       batch_size=batch_size,
  #       dtype=jnp.float32,
  #   )

  #   outputs, cache = transformer(
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       cache,
  #       attention_mask,
  #   )

  #   outputs, cache = transformer(
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       cache,
  #       attention_mask,
  #   )

  #   outputs, cache = transformer(
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       cache,
  #       attention_mask,
  #   )

  #   self.assertEqual(outputs.shape, expected_outputs_shape)
  #   self.assertEqual(cache['layer_0']['v'].shape, expected_cache_shape)


  def test_samples(self):
    vocab = MockVocab()
    num_layers = 6
    transformer_config = transformer_lib.TransformerConfig(  # pytype: disable=wrong-arg-types
        num_layers=num_layers,
        num_embed=vocab.GetPieceSize(),
        embed_dim=768,
        hidden_dim=6144,
        num_heads=4,
        num_kv_heads=4,
        head_dim=256,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        attn_logits_soft_cap=None,
        use_post_attn_norm=None,
        use_post_ffw_norm=None,
        transpose_gating_einsum=False,
    )
    params = create_fake_params(transformer_config)
    transformer = transformer_lib.Transformer.from_params(
         params, transformer_config,  #, rngs=nnx.Rngs(params=0)
    )
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
        cache_size=1024,
    )

    result = sampler(['input string', 'hello world'], total_generation_steps=10)
    self.assertIsNotNone(result)

    top_p_result = sampler(
        ['input string', 'hello world'],
        total_generation_steps=10,
        temperature=9,
        top_p=0.95,
    )
    self.assertIsNotNone(top_p_result)
    self.assertNotEqual(result.text, top_p_result.text)

    top_p_result_2 = sampler(
        ['input string', 'hello world'],
        total_generation_steps=20,
        temperature=9,
        top_p=0.95,
        seed=jax.random.PRNGKey(42),
    )
    self.assertIsNotNone(top_p_result_2)
    self.assertNotEqual(top_p_result.text, top_p_result_2.text)


if __name__ == '__main__':
  absltest.main()
