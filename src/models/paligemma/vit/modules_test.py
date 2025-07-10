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
"""Tests for ViT modules."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import modules
import jax
import jax.numpy as jnp
import numpy as np


# TODO: Complete test cases
class MlpBlockTest(parameterized.TestCase):

  # Parameters for testing MlpBlock
  @parameterized.parameters(
      dict(
          features=2,
          mlp_dim=3,
          batch_size=2,
          dtype_mm=jnp.float32,
          expected_val=[5.8637934, 11.99979], # [11.72758674, 47.99916],
          expected_shape=(2, 1, 2),
      ),
  )
  def test_mlp_block(
      self,
      features,
      mlp_dim,
      batch_size,
      dtype_mm,
      expected_val,
      expected_shape
  ):
    # Create inputs with shape (batch_size, sequence_length, features)
    inputs = jnp.arange(1, batch_size+1)[:, None, None]
    inputs = jnp.repeat(inputs, features, axis=-1)

    mlp = modules.MlpBlock(
        features=features,
        mlp_dim=mlp_dim,
        dtype_mm=dtype_mm,
        rngs=nnx.Rngs(params=0),
    )

    # Manually set kernels to ones and biases to zeros for predictable output
    mlp.linear1.kernel.value = jnp.ones((features, mlp_dim), dtype=dtype_mm)
    mlp.linear1.bias.value = jnp.zeros((mlp_dim,), dtype=dtype_mm)
    mlp.linear2.kernel.value = jnp.ones((mlp_dim, features), dtype=dtype_mm)
    mlp.linear2.bias.value = jnp.zeros((features,), dtype=dtype_mm)

    with jax.default_matmul_precision('float32'):
      outputs = mlp(inputs)

    np.testing.assert_array_almost_equal(outputs[:, 0, 0], expected_val)
    self.assertEqual(outputs.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
