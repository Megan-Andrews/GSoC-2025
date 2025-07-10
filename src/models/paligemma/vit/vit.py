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
"""ViT sub-modules."""

from __future__ import annotations

import functools
from typing import Union, Any, Optional
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import helpers
import params as params_lib
import modules
from modules import Encoder

@dataclasses.dataclass(frozen=True)
class VisionTransformerConfig:
  """Configuration for the Vision Transformer."""
  num_classes: int
  patch_size: int
  img_size: int
  depth: int
  mlp_dim: int
  num_heads: int
  hidden_size: int
  in_channels: int
  dtype_mm: str
  
  @classmethod
  def from_path(cls, path: str) -> VisionTransformerConfig:
    """Creates a TransformerConfig from loaded parameters."""
    params = params_lib.load_params(path)
    return cls.from_params(params)

  @classmethod
  def from_params(cls, params: params_lib.Params) -> VisionTransformerConfig:
    """Creates a VisionTransformerConfig from loaded parameters.

    Args:
      params: Model parameters

    Returns:
      VisionTransformerConfig.
    """
    # TODO read in params to determine which ViT config should be used.
    return cls.so400m_14()

    # raise ValueError('Could not determine ViT variant from params.')

  @classmethod
  def so400m_14(cls):
    return cls(
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
    elif k!= 'img': # temporarily disable img variable name while developing vit
      new_key.append(k)

  return tuple(new_key)

def _assign_npz_params_to_nnx_state(
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any
) -> dict[tuple[str, ...], Any]:
  state[mapped_path].value = val
  return state

class VisionTransformer(nnx.Module):
  """Transformer Model."""

  @classmethod
  def from_params(
      cls,
      params: params_lib.Params,
      config: None | VisionTransformerConfig = None,
      # sow_config: sow_lib.SowConfig = sow_lib.SowConfig(),
  ) -> VisionTransformer:
    if config is None:
      config = VisionTransformerConfig.from_params(params)

    return helpers.module_from_linen_variables(
        module_factory=lambda: cls(
            config, rngs=nnx.Rngs(params=0) #, sow_config=sow_config
        ),
        variables=params['img'],
        map_key_fn=_map_npz_var_names,
        assign_val_fn=_assign_npz_params_to_nnx_state,
    )
  
  def __init__(
  self,
  config: VisionTransformerConfig,
  *,
  rngs: nnx.Rngs = nnx.Rngs(0)):

    self.num_classes = config.num_classes
    self.patch_size = config.patch_size
    self.img_size = config.img_size
    self.depth = config.depth
    self.mlp_dim = config.mlp_dim
    self.num_heads = config.num_heads
    self.hidden_size = config.hidden_size
    self.in_channels = config.in_channels
    self.dtype_mm = config.dtype_mm
    self.rngs = rngs
    # Calculate the number of patches generated from the image.
    n_patches = (self.img_size // self.patch_size) ** 2

    self.embedding = nnx.Conv(
        self.in_channels, # in_channels
        self.hidden_size, # out_channels
        kernel_size=(self.patch_size,self.patch_size),
        strides=(self.patch_size,self.patch_size),
        padding="VALID",
        dtype=self.dtype_mm,
        use_bias=True,
        rngs=self.rngs)

    # TODO: initialization doesn't matter initialize to zero
    initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
    self.pos_embedding = nnx.Param(
            initializer(
              rngs.params(),
              (1, n_patches, self.hidden_size),
              dtype=self.dtype_mm))

    self.Transformer = Encoder(
        hidden_size=self.hidden_size,
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        rngs=nnx.Rngs(0))

    self.head = nnx.Linear(
      self.hidden_size,
      self.num_classes,
      dtype=self.dtype_mm,
      rngs=self.rngs)


  def __call__(self, image):
    """Applies Transformer model to image."""
    out = {}
    image = jnp.asarray(image, self.dtype_mm)
    x = out["stem"] = self.embedding(image)
    n, h, w, c = x.shape
    print(x.shape)
    x = jnp.reshape(x, [n, h * w, c])
    x = out["with_posemb"] = x + self.pos_embedding

    x, out["encoder"] = self.Transformer(x)
    encoded = out["encoded"] = x

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
      x_2d = out["logits_2d"] = self.head(x_2d)
      x = out["logits"] = self.head(x)

    return x, out



