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

from collections.abc import Sequence
import enum
from typing import Optional
from typing import Any, Union

from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike  # pylint: disable=g-importing-member,g-multiple-import

LayerCache = dict[str, Array]
Shape = Sequence[Union[int, Any]]

class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      features: int, # Explicitly pass input features for __init__
      mlp_dim: Optional[int] = None, # TODO: should I use Optional?
      dropout_rate: float = 0.0, # TODO: dropout only for training?
      dtype_mm: jnp.dtype = jnp.float32, # TODO: is this necessary
      *,
      rngs: nnx.Rngs
      # sow_config: sow_lib.SowConfig = sow_lib.SowConfig()
  ):

    self.mlp_dim = mlp_dim
    self.dropout_rate = dropout_rate
    self.dtype_mm = dtype_mm

    d = features # Input dimension for the first Dense layer

    self.linear1 = nnx.Linear(
        in_features=d,
        out_features=self.mlp_dim or 4 * d,
        dtype=self.dtype_mm,
        rngs=rngs,
        kernel_init= nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6)
    )
    self.dropout_layer = nnx.Dropout(
        rate=self.dropout_rate,
        rngs=rngs, # rngs are passed to the Dropout layer for dropout mask generation
        # deterministic=False # Set to False for training
    )
    self.linear2 = nnx.Linear(
        in_features=self.mlp_dim or 4 * d,
        out_features=d,
        dtype=self.dtype_mm,
        rngs=rngs,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6)
    )

  def __call__(self, x: jax.Array, deterministic: bool):
    """Applies Transformer MlpBlock module."""

    x = self.linear1(x)
    # x = nnx.with_logical_constraint(x, ("act_batch", "act_len", "act_emb")) # This line is removed.
    x = nnx.gelu(x)
    x = self.dropout_layer(x, deterministic=deterministic)
    x = self.linear2(x)
    return x
  


class Encoder1DBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  # NNX modules are initialized with rngs and the features of the input
  def __init__(
      self,
      features: int,
      *,
      num_heads: int = 12,
      dropout: float = 0.0,
      dtype_mm: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs, # TODO is this necessary
      mlp_dim: Optional[int] = None):

    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.dtype_mm = dtype_mm
    self.features = features # Store features for MlpBlock if mlp_dim is None

    # NNX layers are direct instances of modules
    self.norm1 = nnx.LayerNorm(features, rngs=rngs) # TODO does this need rngs?
    self.mha = nnx.MultiHeadAttention(
        num_heads=self.num_heads,
        qkv_features=features, # Explicitly pass features for query, key, value
        kernel_init=nn.initializers.xavier_uniform(), #
        out_features=features,
        in_features=features,
        dtype=self.dtype_mm,
        rngs=rngs,
        decode=False,
    )
    self.dropout1 = nnx.Dropout(rate=self.dropout) # NNX Dropout is a Module

    self.norm2 = nnx.LayerNorm(features, rngs=rngs)
    # Pass features to MlpBlock. If mlp_dim is None in MlpBlock, it will default.
    self.mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        features=features,
        dropout_rate=self.dropout,
        dtype_mm=self.dtype_mm,
        rngs=rngs,
    )
    self.dropout2 = nnx.Dropout(rate=self.dropout) # NNX Dropout is a Module

  def __call__(self, x: jax.Array, deterministic: bool):
    out = {}

    y = self.norm1(x)
    # MultiHeadDotProductAttention in NNX might require query, key, value as separate arguments.
    # If the input is the same for all, you pass it three times.
    y = out["sa"] = self.mha(y, y) #TODO (y,y,y) or (y,y)
    y = self.dropout1(y, deterministic=deterministic)
    x = out["+sa"] = x + y

    y = self.norm2(x)
    y = out["mlp"] = self.mlp(y, deterministic=deterministic)
    y = self.dropout2(y, deterministic=deterministic)
    x = out["+mlp"] = x + y
    return x, out
