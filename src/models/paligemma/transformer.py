"""PaliGemma Transformer"""

# from typing import Optional
from __future__ import annotations
import functools
from typing import Any

from flax import nnx
import helpers
import jax.numpy as jnp
# from jaxtyping import Array

import params as params_lib
# import gemma.modules as gemma_modules
from gemma.transformer import Transformer as GemmaTransformer
from gemma.transformer import TransformerConfig as GemmaTransformerConfig



class PaliGemmaTransformerConfig(GemmaTransformerConfig):
  """Extended config for PaliGemma."""
  use_vision: bool = True

  @classmethod
  def from_params(cls, params: params_lib.Params) -> PaliGemmaTransformerConfig:

    pass



# TODO: verify this (using npz rather than linen)
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

def _assign_params_to_nnx_state(
    # TODO: verify necessity of this function
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any,
    transpose_gating_einsum: bool,
) -> dict[tuple[str, ...], Any]:
  """Splits and maybe transposes gate_proj."""
  if 'gate_proj' in mapped_path:
    if transpose_gating_einsum:
      val = jnp.swapaxes(val, 1, 2)
    state[mapped_path].value = val[0]
    state[mapped_path[:-2] + ('up_proj', 'kernel')].value = val[1]
  else:
    state[mapped_path].value = val
  return state

class PaliGemmaTransformer(GemmaTransformer):
  """PaliGemma Transformer model (multimodal extension)."""

  @classmethod
  def from_params(
    cls,
    params: params_lib.Params,
    config: None | PaliGemmaTransformerConfig = None
    # TODO: sow_config
  ) -> PaliGemmaTransformer:
    if config is None:
      config = PaliGemmaTransformerConfig.from_params(params)
    assign_val_fn = functools.partial(
      _assign_params_to_nnx_state,
      transpose_gating_einsum = config.transpose_gating_einsum
    )
    # TODO modify this to accomodate npz rather than linen variables
    return helpers.module_from_linen_variables(
      module_factory = lambda: cls(
        config, rngs=nnx.Rngs(params=0) #, sow_config=sow_config
      ),
      variables=params['params'], # TODO: determine key for params dict
      map_key_fn= _map_linen_var_names, # TODO: map npz var names
      assign_val_fn=assign_val_fn,
    )

  def __init__(
      self,
      config: PaliGemmaTransformerConfig,
      *,
      rngs: nnx.Rngs,
      sow_config=None,
  ):
    super().__init__(config, rngs=rngs, sow_config=sow_config)
    # TODO: Add PaLI-specific components

    


