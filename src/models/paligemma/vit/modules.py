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

from typing import Optional, Union

from flax import nnx
import jax
import jax.numpy as jnp


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      features: int,
      mlp_dim: Optional[int] = None, 
      dtype_mm: jnp.dtype = jnp.float32, 
      *,
      rngs: nnx.Rngs
      # sow_config: sow_lib.SowConfig = sow_lib.SowConfig()
  ):
    self.mlp_dim = mlp_dim
    self.dtype_mm = dtype_mm
    d = features # Input dimension for the first Dense layer

    self.linear1 = nnx.Linear(
        in_features=d,
        out_features=self.mlp_dim or 4 * d,
        dtype=self.dtype_mm,
        rngs=rngs,
    )

    self.linear2 = nnx.Linear(
        in_features=self.mlp_dim or 4 * d,
        out_features=d,
        dtype=self.dtype_mm,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array):
    """Applies Transformer MlpBlock module."""

    x = self.linear1(x)
    x = nnx.gelu(x)
    x = self.linear2(x)
    return x  

class Encoder1DBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  def __init__(
    self,
    hidden_size: int, # modification from Big Vision, requre hidden_size
    num_heads: int = 12,
    dtype_mm: jnp.dtype = jnp.float32,
    mlp_dim: Optional[int] = None,
    *,
    rngs: nnx.Rngs
    ):

    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dtype_mm = dtype_mm
    # Store features for MlpBlock if mlp_dim is None
    self.hidden_size = hidden_size

    # NNX layers are direct instances of modules
    self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
    self.attn = nnx.MultiHeadAttention(
      num_heads=self.num_heads,
      in_features=self.hidden_size,
      dtype=self.dtype_mm,
      rngs=rngs,
      deterministic=True,
      decode=False,
    )


    self.norm2 = nnx.LayerNorm(self.hidden_size, rngs=rngs)
    # Pass features to MlpBlock. If mlp_dim is None in MlpBlock, it will default
    self.mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        features=self.hidden_size,
        dtype_mm=self.dtype_mm,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array):
    out = {}

    y = self.norm1(x)
    # If the input is the same for all, you pass it three times.
    y = out["sa"] = self.attn(y) #TODO (y,y,y) or (y,y)
    x = out["+sa"] = x + y

    y = self.norm2(x)
    y = out["mlp"] = self.mlp(y)
    x = out["+mlp"] = x + y
    return x, out

class Encoder(nnx.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  def __init__(
  self,
  hidden_size: int, # modification from Big Vision, requre hidden_size
  *,
  depth: int,
  mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
  num_heads: int = 12,
  scan: bool = False,
  dtype_mm: str = "float32",
  rngs: nnx.Rngs = nnx.Rngs(0)):
    self.depth = depth
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.scan = scan
    self.dtype_mm = dtype_mm
    self.rngs = rngs

    @nnx.split_rngs(splits=self.depth)
    @nnx.vmap(in_axes=(None,None,None,0), out_axes=0)
    def create_encoderblock(hidden_size, mlp_dim, num_heads, rngs: nnx.Rngs):
      return Encoder1DBlock(
        hidden_size=hidden_size, 
        mlp_dim=mlp_dim, 
        num_heads=num_heads, 
        rngs=rngs)

    # print(f"Creating {self.depth} encoder blocks...")
    self.encoderblock = create_encoderblock(
      self.hidden_size, 
      self.mlp_dim, 
      self.num_heads, 
      self.rngs)
    self.encoder_norm = nnx.LayerNorm(self.hidden_size, rngs=rngs)


  def __call__(self, x):

    @nnx.split_rngs(splits=self.depth)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def forward_encoderblock(x, model: Encoder1DBlock):
      x = model(x)
      return x

    out = {}

    # print(f"Running {self.depth} encoder blocks...")
    x, scan_out = forward_encoderblock(x, self.encoderblock)
    # print(f"Finished running {self.depth} encoder blocks...")

    for lyr in range(self.depth):
      out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)

    out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return self.encoder_norm(x), out

  