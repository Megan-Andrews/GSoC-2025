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

from collections import defaultdict
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from gemma import modules
# from gemma import sow_lib
from gemma import transformer as transformer_lib
from vit import transformer as vision_transformer_lib
import transformer as paligemma_transformer_lib
import jax.numpy as jnp
import numpy as np


def create_fake_params(config: paligemma_transformer_lib.PaliGemmaConfig):
  def nested_defaultdict():
    return defaultdict(nested_defaultdict)

  res = nested_defaultdict()
  res['llm'] = nested_defaultdict()
  res['img'] = nested_defaultdict()
  params = res

  gemma_config = config.gemma_config
  # 1. embedding params
  params['llm']['embedder']['input_embedding'] = jnp.ones(
      (gemma_config.num_embed, gemma_config.embed_dim)
  )
  # 2. final norm params
  params['llm']['final_norm'] = {'scale': jnp.ones((gemma_config.embed_dim,))}

  # 3. attention block params
  # for layer_idx in range(config.num_layers):
  params['llm']['llm']['layers']['attn']['attn_vec_einsum']['w'] = jnp.ones(
      (gemma_config.num_layers, gemma_config.num_heads, gemma_config.head_dim, gemma_config.embed_dim)
  )
  if gemma_config.num_heads == gemma_config.num_kv_heads:
    params['llm']['layers']['attn']['qkv_einsum']['w'] = jnp.ones(
        (gemma_config.num_layers, 3, gemma_config.num_heads, gemma_config.embed_dim, gemma_config.head_dim)
    )
  else:
    params['llm']['layers']['attn']['q_einsum']['w'] = jnp.ones(
        (gemma_config.num_layers, gemma_config.num_heads, gemma_config.embed_dim, gemma_config.head_dim)
    )
    params['llm']['layers']['attn']['kv_einsum']['w'] = jnp.ones(
        (gemma_config.num_layers, gemma_config.num_kv_heads, gemma_config.embed_dim, gemma_config.head_dim)
    )

  # 4. feedforward block params
  params['llm']['layers']['mlp']['gating_einsum'] = jnp.ones(
      (gemma_config.num_layers, 2, gemma_config.embed_dim, gemma_config.hidden_dim)
  )
  params['llm']['layers']['mlp']['linear'] = jnp.ones(
      (gemma_config.num_layers, gemma_config.hidden_dim, gemma_config.embed_dim)
  )

  # 5. layer norm params
  params['llm']['layers']['pre_attention_norm']['scale'] = jnp.ones((
      gemma_config.num_layers, gemma_config.embed_dim,
  ))
  params['llm']['layers']['pre_ffw_norm']['scale'] = jnp.ones((
      gemma_config.num_layers, gemma_config.embed_dim,
  ))

  if gemma_config.use_post_attn_norm:
    params['llm']['layers']['post_attention_norm']['scale'] = jnp.ones((
        gemma_config.num_layers, gemma_config.embed_dim,
    ))
  if gemma_config.use_post_ffw_norm:
    params['llm']['layers']['post_ffw_norm']['scale'] = jnp.ones((
        gemma_config.num_layers, gemma_config.embed_dim,
    ))

  # img params
  vit_config = config.vit_config
  # 1. embedding params
  params['img']['embedding']['bias'] = jnp.ones((vit_config.hidden_size,))
  params['img']['embedding']['kernel'] = jnp.ones((vit_config.patch_size, vit_config.patch_size, vit_config.in_channels, vit_config.hidden_size,))

  # 2. head params
  params['img']['head']['bias'] = jnp.ones((vit_config.num_classes,))
  params['img']['head']['kernel'] = jnp.ones((vit_config.hidden_size, vit_config.num_classes))

  # 3. position embedding 
  height_width = vit_config.img_size // vit_config.patch_size
  num_patches = height_width * height_width
  params['img']['pos_embedding'] = jnp.ones((1, num_patches ,vit_config.hidden_size))

  # 4. encoder norm
  params['img']['Transformer']['encoder_norm']['bias'] = jnp.ones((vit_config.hidden_size,))
  params['img']['Transformer']['encoder_norm']['scale'] = jnp.ones((vit_config.hidden_size,))

  # 5. encoder block norm
  params['img']['Transformer']['encoderblock']['norm1']['bias'] = jnp.ones((vit_config.depth, vit_config.hidden_size))
  params['img']['Transformer']['encoderblock']['norm1']['scale'] = jnp.ones((vit_config.depth, vit_config.hidden_size))
  params['img']['Transformer']['encoderblock']['norm2']['bias'] = jnp.ones((vit_config.depth, vit_config.hidden_size))
  params['img']['Transformer']['encoderblock']['norm2']['scale'] = jnp.ones((vit_config.depth, vit_config.hidden_size))

  # 6. encoder block mlp
  params['img']['Transformer']['encoderblock']['mlp']['linear1']['bias'] = jnp.ones((vit_config.depth, vit_config.mlp_dim))
  params['img']['Transformer']['encoderblock']['mlp']['linear1']['kernel'] = jnp.ones((vit_config.depth, vit_config.hidden_size, vit_config.mlp_dim))
  params['img']['Transformer']['encoderblock']['mlp']['linear2']['bias'] = jnp.ones((vit_config.depth, vit_config.hidden_size))
  params['img']['Transformer']['encoderblock']['mlp']['linear2']['kernel'] = jnp.ones((vit_config.depth, vit_config.mlp_dim, vit_config.hidden_size))

  # 7.  encoderblock attention
  head_dim = vit_config.hidden_size // vit_config.num_heads
  params['img']['Transformer']['encoderblock']['attn']['key']['bias'] = jnp.ones((vit_config.depth, vit_config.num_heads, head_dim))
  params['img']['Transformer']['encoderblock']['attn']['key']['kernel'] = jnp.ones((vit_config.depth, vit_config.hidden_size, vit_config.num_heads, head_dim))
  params['img']['Transformer']['encoderblock']['attn']['out']['bias'] = jnp.ones((vit_config.depth, vit_config.hidden_size))
  params['img']['Transformer']['encoderblock']['attn']['out']['kernel'] = jnp.ones((vit_config.depth, vit_config.num_heads, head_dim, vit_config.hidden_size))
  params['img']['Transformer']['encoderblock']['attn']['query']['bias'] = jnp.ones((vit_config.depth, vit_config.num_heads, head_dim))
  params['img']['Transformer']['encoderblock']['attn']['query']['kernel'] = jnp.ones((vit_config.depth, vit_config.hidden_size, vit_config.num_heads, head_dim))
  params['img']['Transformer']['encoderblock']['attn']['value']['bias'] = jnp.ones((vit_config.depth, vit_config.num_heads, head_dim))
  params['img']['Transformer']['encoderblock']['attn']['value']['kernel'] = jnp.ones((vit_config.depth, vit_config.hidden_size, vit_config.num_heads, head_dim))

  return res


class TransformerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Prime number to ease shape tracing
      dict(
          num_layers=3,
          num_embed=17,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=2,
          hidden_dim=11,
          head_dim=8,
          cache_size=29,
          batch_size=7,
          sequence_length=17,
          expected_outputs_shape=(7, 17, 17),
          expected_cache_shape=(3, 7, 29, 2, 8),
      ),
      dict(
          num_layers=3,
          num_embed=4,
          embed_dim=2,
          num_heads=2,
          num_kv_heads=1,
          hidden_dim=4,
          head_dim=4,
          cache_size=2,
          batch_size=1,
          sequence_length=1,
          expected_outputs_shape=(1, 1, 4),
          expected_cache_shape=(3, 1, 2, 1, 4),
      ),
  )
  def test_transformer(
      self,
      num_layers,
      num_embed,
      embed_dim,
      num_heads,
      num_kv_heads,
      hidden_dim,
      head_dim,
      cache_size,
      batch_size,
      sequence_length,
      expected_outputs_shape,
      expected_cache_shape,
  ):

    gemma_config = transformer_lib.TransformerConfig(
        num_layers=num_layers,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    vit_config = vision_transformer_lib.VisionTransformerConfig(
        num_classes=2048,
        patch_size=14,
        img_size=224,
        depth=27,
        mlp_dim=4304,
        num_heads=16,
        hidden_size=1152,
        in_channels=3,
        dtype_mm= "float32"
    )
    paligemma_config = paligemma_transformer_lib.PaliGemmaConfig(
        gemma_config=gemma_config,
        vit_config=vit_config
    )

    attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
    transformer = paligemma_transformer_lib.PaliGemmaTransformer(
        config=paligemma_config, rngs=nnx.Rngs(params=0)
    )


    cache = transformer.llm.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )

    outputs, cache = transformer.llm(
        jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
        cache,
        attention_mask,
    )

    self.assertEqual(outputs.shape, expected_outputs_shape)
    self.assertEqual(cache['v'].shape, expected_cache_shape)


  @parameterized.parameters(
      # Prime number to ease shape tracing
      dict(
          num_layers=3,
          num_embed=17,
          embed_dim=2048,
          num_heads=2,
          num_kv_heads=2,
          hidden_dim=11,
          head_dim=8,
          cache_size=400,
          batch_size=7,
          sequence_length=17,
          expected_outputs_shape=(7, 256+17, 17),
          expected_cache_shape=(3, 7, 400, 2, 8),
          num_classes=2048,
          patch_size=14,
          img_size=224,
          depth=27,
          mlp_dim=4304,
          vit_num_heads=16,
          hidden_size=1152,
          in_channels=3,
          dtype_mm= "float32",
      ),
      dict(
          num_layers=3,
          num_embed=4,
          embed_dim=2048,
          num_heads=2,
          num_kv_heads=1,
          hidden_dim=4,
          head_dim=4,
          cache_size=300,
          batch_size=1,
          sequence_length=1,
          expected_outputs_shape=(1, 256+1, 4),
          expected_cache_shape=(3, 1, 300, 1, 4),
          num_classes=2048,
          patch_size=14,
          img_size=224,
          depth=27,
          mlp_dim=4304,
          vit_num_heads=16,
          hidden_size=1152,
          in_channels=3,
          dtype_mm= "float32",
      ),
  )
  def test_prefill_cache(
      self,
      num_layers,
      num_embed,
      embed_dim,
      num_heads,
      num_kv_heads,
      hidden_dim,
      head_dim,
      cache_size,
      batch_size,
      sequence_length,
      expected_outputs_shape,
      expected_cache_shape,
      num_classes,
      patch_size,
      img_size,
      depth,
      mlp_dim,
      vit_num_heads,
      hidden_size,
      in_channels,
      dtype_mm,
  ):

    gemma_config = transformer_lib.TransformerConfig(
        num_layers=num_layers,
        num_embed=num_embed,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        final_logit_softcap=None,
        attention_types=[modules.AttentionType.GLOBAL] * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    vit_config = vision_transformer_lib.VisionTransformerConfig(
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        depth=depth,
        mlp_dim=mlp_dim,
        num_heads=vit_num_heads,
        hidden_size=hidden_size,
        in_channels=in_channels,
        dtype_mm= dtype_mm
    )
    paligemma_config = paligemma_transformer_lib.PaliGemmaConfig(
        gemma_config=gemma_config,
        vit_config=vit_config
    )

    key = jax.random.PRNGKey(0)
    image = jax.random.uniform(key, (batch_size, vit_config.img_size,
        vit_config.img_size, vit_config.in_channels), dtype=jnp.float32)
    text = jnp.tile(jnp.arange(sequence_length), (batch_size, 1))
    # logits, _ = transformer(image,)
    # print("start", logits.shape, "end")

    height_width = vit_config.img_size // vit_config.patch_size
    num_patches = height_width * height_width
    # self.assertEqual(logits.shape, (1, num_patches, vit_config.num_classes))

    # attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)
    transformer = paligemma_transformer_lib.PaliGemmaTransformer(
        config=paligemma_config, rngs=nnx.Rngs(params=0)
    )


    cache = transformer.llm.init_cache(
        cache_size=cache_size,
        batch_size=batch_size,
        dtype=jnp.float32,
    )

    x, input_mask, mask_ar = transformer.embed_image_and_text(image, text)
    # print("cache size", cache_size, "num patches", num_patches)
    # print("x", x.shape)
    initial_logits, cache = transformer.prefill_cache(x, input_mask, mask_ar, cache, cache_size=cache_size, num_patches=num_patches)

    # outputs, cache = transformer.llm(
    #     jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
    #     jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
    #     cache,
    #     attention_mask,
    # )
    # print(cache['v'])
    self.assertEqual(initial_logits.shape, expected_outputs_shape)
    self.assertEqual(cache['v'].shape, expected_cache_shape)




  # @parameterized.parameters(
  #     ('final_logit_softcap',),
  #     ('attn_logits_soft_cap',),
  # )
  # def test_logit_softcap(
  #     self,
  #     soft_cap_arg,
  # ):
  #   cache_size = 2
  #   batch_size = 1
  #   soft_cap_val = 0.001

  #   attention_mask = jnp.ones((batch_size, 1, cache_size), dtype=jnp.bool)

  #   params = dict(
  #       num_layers=3,
  #       num_embed=4,
  #       embed_dim=2,
  #       num_heads=2,
  #       num_kv_heads=1,
  #       hidden_dim=4,
  #       head_dim=4,
  #       attention_types=[modules.AttentionType.GLOBAL] * 3,
  #       use_post_attn_norm=False,
  #       use_post_ffw_norm=False,
  #   )

  #   no_soft_cap_args = {
  #       'final_logit_softcap': None,
  #       'attn_logits_soft_cap': None,
  #   }

  #   soft_cap_args = no_soft_cap_args.copy()
  #   soft_cap_args[soft_cap_arg] = soft_cap_val

  #   config_soft_cap = transformer_lib.TransformerConfig(
  #       **(params | soft_cap_args)
  #   )
  #   config_no_soft_cap = transformer_lib.TransformerConfig(
  #       **(params | no_soft_cap_args)
  #   )

  #   all_outputs = []
  #   for config in [config_soft_cap, config_no_soft_cap]:
  #     transformer = transformer_lib.Transformer(
  #         config=config, rngs=nnx.Rngs(params=1)
  #     )
  #     cache = transformer.init_cache(
  #         cache_size=cache_size,
  #         batch_size=batch_size,
  #         dtype=jnp.float32,
  #     )

  #     outputs, _ = transformer(
  #         jnp.array([[1]]), jnp.array([[1]]), cache, attention_mask
  #     )
  #     all_outputs.append(outputs)

  #   soft_cap_outputs, no_soft_cap_outputs = all_outputs  # pylint: disable=unbalanced-tuple-unpacking

  #   # Ensure that values aren't equal coming out of computation
  #   self.assertFalse((soft_cap_outputs == no_soft_cap_outputs).all())

  #   # Run soft capping manually
  #   manual_soft_cap_logits = no_soft_cap_outputs / soft_cap_val
  #   manual_soft_cap_logits = jnp.tanh(manual_soft_cap_logits) * soft_cap_val

  #   np.testing.assert_array_almost_equal(
  #       manual_soft_cap_logits, soft_cap_outputs, 1e-5
  #   )

  # @parameterized.parameters([
  #     dict(
  #         config=transformer_lib.TransformerConfig(
  #             num_layers=2,
  #             num_embed=0,  # unused
  #             embed_dim=0,  # unused
  #             hidden_dim=0,  # unused
  #             num_heads=3,
  #             head_dim=4,
  #             num_kv_heads=3,
  #             final_logit_softcap=None,
  #             attention_types=[modules.AttentionType.GLOBAL] * 4,
  #             use_post_attn_norm=False,
  #             use_post_ffw_norm=False,
  #         ),
  #         cache_size=2,
  #         keys=['end_index', 'k', 'v'],
  #         k_shape=(2, 1, 2, 3, 4), # num_layers, batch_size, cache_size, num_heads, head_dim
  #         v_shape=(2, 1, 2, 3, 4),
  #     )
  # ])
  # def test_creates_cache(self, config, cache_size, keys, k_shape, v_shape):
  #   transformer = transformer_lib.Transformer(
  #       config=config, rngs=nnx.Rngs(params=0)
  #   )
  #   cache = transformer.init_cache(
  #       cache_size=cache_size,
  #       batch_size=1,
  #       dtype=jnp.float32,
  #   )

  #   self.assertEqual(list(cache.keys()), keys)
  #   self.assertEqual(cache['k'].shape, k_shape)
  #   self.assertEqual(cache['v'].shape, v_shape)

  # @parameterized.parameters([
  #     dict(
  #         batch_size=1,
  #         seq_size=4,
  #         config=transformer_lib.TransformerConfig(
  #             num_layers=2,
  #             num_embed=4,  # unused
  #             embed_dim=2,
  #             hidden_dim=12,  # unused
  #             num_heads=3,
  #             head_dim=4,
  #             num_kv_heads=3,
  #             final_logit_softcap=None,
  #             attention_types=[modules.AttentionType.GLOBAL] * 2,
  #             use_post_attn_norm=False,
  #             use_post_ffw_norm=False,
  #         ),
  #     )
  # ])
  # def test_forward_no_cache(
  #     self,
  #     batch_size: int,
  #     seq_size: int,
  #     config: transformer_lib.TransformerConfig,
  # ):
  #   cache_size = 6

  #   token_input = jnp.ones((batch_size, seq_size), dtype=jnp.int32)
  #   transformer = transformer_lib.Transformer(
  #       config=config, rngs=nnx.Rngs(params=0)
  #   )
  #   empty_cache = transformer.init_cache(
  #       cache_size=cache_size,
  #       batch_size=batch_size,
  #       dtype=jnp.float32,
  #   )
  #   attention_mask = jnp.ones(
  #       (batch_size, seq_size, cache_size), dtype=jnp.bool
  #   )
  #   positions = transformer_lib.build_positions_from_mask(token_input != 0)

  #   output_cache, _ = transformer(
  #       token_input, positions, empty_cache, attention_mask
  #   )

  #   attention_mask = jnp.ones((batch_size, seq_size, seq_size), dtype=jnp.bool)
  #   output_none, cache_none = transformer(
  #       token_input, positions, None, attention_mask
  #   )

  #   self.assertIsNone(cache_none)
  #   np.testing.assert_array_almost_equal(output_cache, output_none, 1e-5)

  # def test_attention_types(
  #     self,
  # ):

  #   config = transformer_lib.TransformerConfig(
  #       num_layers=2,
  #       num_embed=4,
  #       embed_dim=2,
  #       hidden_dim=12,
  #       num_heads=3,
  #       head_dim=4,
  #       num_kv_heads=3,
  #       final_logit_softcap=None,
  #       attention_types=[modules.AttentionType.GLOBAL] * 2,
  #       use_post_attn_norm=False,
  #       use_post_ffw_norm=False,
  #   )
  #   transformer = transformer_lib.Transformer(
  #       config=config, rngs=nnx.Rngs(params=0)
  #   )
  #   cache = transformer.init_cache(
  #       cache_size=6,
  #       batch_size=1,
  #       dtype=jnp.float32,
  #   )
  #   self.assertTrue(cache)

  # @parameterized.parameters(
  #     dict(
  #         config=transformer_lib.TransformerConfig(
  #             num_layers=2,
  #             num_embed=4,
  #             embed_dim=2,
  #             hidden_dim=12,
  #             num_heads=3,
  #             head_dim=4,
  #             num_kv_heads=3,
  #             final_logit_softcap=None,
  #             attention_types=[modules.AttentionType.GLOBAL] * 2,
  #             use_post_attn_norm=False,
  #             use_post_ffw_norm=False,
  #         ),
  #     ),
  #     dict(
  #         config=transformer_lib.TransformerConfig(
  #             num_layers=2,
  #             num_embed=4,
  #             embed_dim=2,
  #             hidden_dim=12,
  #             num_heads=3,
  #             head_dim=4,
  #             num_kv_heads=3,
  #             final_logit_softcap=None,
  #             attention_types=[modules.AttentionType.GLOBAL] * 2,
  #             use_post_attn_norm=True,
  #             use_post_ffw_norm=True,
  #         ),
  #     ),
  # )
  # def test_load_from_params(self, config):
  #   params = create_fake_params(config)
  #   transformer = transformer_lib.Transformer.from_params(params, config)
  #   logits, _ = transformer(
  #       last_tokens=jnp.tile(jnp.arange(3), (2, 1)),
  #       positions=jnp.tile(jnp.arange(3), (2, 1)),
  #       cache=None,
  #       attention_mask=jnp.ones((2, 1, 3), dtype=jnp.bool),
  #   )
  #   self.assertEqual(logits.shape, (2, 3, 4))

  # @parameterized.parameters([
  #     sow_lib.SowConfig(embeddings=True),
  #     sow_lib.SowConfig(rs_after_attention=True),
  #     sow_lib.SowConfig(rs_after_ffw=True),
  #     sow_lib.SowConfig(attn_logits_topk=5),
  #     sow_lib.SowConfig(mlp_hidden_topk=11),
  # ])
  # def test_sow_intermediates(self, sow_config):
  #   batch_size = 3
  #   sequence_length = 7
  #   num_layers = 2
  #   config = transformer_lib.TransformerConfig(
  #       num_layers=num_layers,
  #       num_embed=4,
  #       embed_dim=48,
  #       hidden_dim=12,
  #       num_heads=1,
  #       head_dim=4,
  #       num_kv_heads=1,
  #       final_logit_softcap=None,
  #       use_post_attn_norm=False,
  #       use_post_ffw_norm=False,
  #       attention_types=[modules.AttentionType.GLOBAL] * num_layers,
  #   )
  #   attention_mask = jnp.ones(
  #       (batch_size, sequence_length, sequence_length), dtype=jnp.bool
  #   )
  #   transformer = transformer_lib.Transformer(
  #       config=config, rngs=nnx.Rngs(params=0), sow_config=sow_config
  #   )
  #   transformer(
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       jnp.tile(jnp.arange(sequence_length), (batch_size, 1)),
  #       None,
  #       attention_mask,
  #   )

  #   if sow_config.embeddings:
  #     self.assertTrue(hasattr(transformer, 'embeddings'))
  #     embeddings = transformer.embeddings.value[0]
  #     self.assertEqual(
  #         embeddings.shape,
  #         (batch_size, sequence_length, config.embed_dim),
  #     )
  #   else:
  #     self.assertFalse(hasattr(transformer, 'embeddings'))

  #   # for layer in transformer.layers:
  #   layer = transformer.layers
  #   if sow_config.rs_after_attention:
  #     self.assertTrue(hasattr(layer, 'rs_after_attention'))
  #     rs_after_attention = layer.rs_after_attention.value[0]
  #     self.assertIsNotNone(rs_after_attention)
  #     self.assertEqual(
  #         rs_after_attention.shape,
  #         (num_layers, batch_size, sequence_length, config.embed_dim),
  #     )
  #   else:
  #     self.assertFalse(hasattr(layer, 'rs_after_attention'))
  #   if sow_config.rs_after_ffw:
  #     self.assertTrue(hasattr(layer, 'rs_after_ffw'))
  #     rs_after_ffw = layer.rs_after_ffw.value[0]
  #     self.assertIsNotNone(rs_after_ffw)
  #     self.assertEqual(
  #         rs_after_ffw.shape,
  #         (num_layers, batch_size, sequence_length, config.embed_dim),
  #     )
  #   else:
  #     self.assertFalse(hasattr(layer, 'rs_after_ffw'))
  #   if sow_config.attn_logits_topk:
  #     self.assertTrue(hasattr(layer.attn, 'logits_topk_values'))
  #     attn_logits_topk_values = layer.attn.logits_topk_values.value[0]
  #     self.assertIsNotNone(attn_logits_topk_values)
  #     self.assertEqual(
  #         attn_logits_topk_values.shape,
  #         (
  #             num_layers,
  #             batch_size,
  #             sequence_length,
  #             config.num_heads,
  #             sow_config.attn_logits_topk,
  #         ),
  #     )
  #     self.assertTrue(hasattr(layer.attn, 'logits_topk_indices'))
  #     attn_logits_topk_indices = layer.attn.logits_topk_indices.value[0]
  #     self.assertIsNotNone(attn_logits_topk_indices)
  #     self.assertEqual(
  #         attn_logits_topk_indices.shape,
  #         (
  #             num_layers,
  #             batch_size,
  #             sequence_length,
  #             config.num_heads,
  #             sow_config.attn_logits_topk,
  #         ),
  #     )
  #   else:
  #     self.assertFalse(hasattr(layer.attn, 'logits_topk_values'))
  #     self.assertFalse(hasattr(layer.attn, 'logits_topk_indices'))
  #   if sow_config.mlp_hidden_topk:
  #     self.assertTrue(hasattr(layer.mlp, 'hidden_topk_values'))
  #     ffw_hidden_topk_values = layer.mlp.hidden_topk_values.value[0]
  #     self.assertIsNotNone(ffw_hidden_topk_values)
  #     self.assertEqual(
  #         ffw_hidden_topk_values.shape,
  #         (
  #             num_layers,
  #             batch_size,
  #             sequence_length,
  #             sow_config.mlp_hidden_topk,
  #         ),
  #     )
  #     self.assertTrue(hasattr(layer.mlp, 'hidden_topk_indices'))
  #     ffw_hidden_topk_indices = layer.mlp.hidden_topk_indices.value[0]
  #     self.assertIsNotNone(ffw_hidden_topk_indices)
  #     self.assertEqual(
  #         ffw_hidden_topk_indices.shape,
  #         (
  #             num_layers,
  #             batch_size,
  #             sequence_length,
  #             sow_config.mlp_hidden_topk,
  #         ),
  #     )
  #   else:
  #     self.assertFalse(hasattr(layer.mlp, 'hidden_topk_values'))
  #     self.assertFalse(hasattr(layer.mlp, 'hidden_topk_indices'))


if __name__ == '__main__':
  absltest.main()
