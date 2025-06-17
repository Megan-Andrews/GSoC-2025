

"""Utils for loading PaliGemma params."""

from collections.abc import Mapping
import functools
from typing import Any

import jax.numpy as jnp



Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
  """Loads parameters from an NPZ file and formats them for compatibility.
  """
  print(f"Loading parameters from {path}...")
  params = load_params_from_npz(path) # <--- Modified: use the NPZ loader

  # Nest the parameters into the Flax/NNX PyTree structure
  nested_params = nest_params(params)
  return nested_params


# --- MODIFIED FUNCTION ---
# The original `load_params` used orbax.checkpoint.
# We need a new function or modify `load_params` to use np.load.
@functools.cache
def load_params_from_npz(path: str) -> Params:
  """Loads parameters from a .npz file."""
  print(f"Loading parameters from {path} using numpy.load...")
  with jnp.load(path) as data:
    loaded_params = {k: jnp.array(v) for k, v in data.items()}
  print(f"Loaded {len(loaded_params)} arrays.")
  return loaded_params


def nest_params(params: Params) -> Params:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param in params.items():
    parts = path.split('/')
    subdict = nested_params
    for i, key in enumerate(parts):
      if i == len(parts) - 1:  # Last part is the leaf (the actual parameter)
        subdict[key] = param
      else:
        subdict = subdict.setdefault(key, {})
  return nested_params
